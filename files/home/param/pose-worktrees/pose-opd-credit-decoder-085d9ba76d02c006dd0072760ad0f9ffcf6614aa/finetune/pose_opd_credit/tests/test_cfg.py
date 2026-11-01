"""Native silent-speech pose guidance and complete replay gradient checks."""
import contextlib
import copy
from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

from finetune.pose_opd_credit import runtime
from finetune.pose_opd_credit.config import TrainingConfig
from finetune.pose_opd_credit.tests.test_runtime import ToyPPM, dense_expected, fixture


class DenseGuidedDecoder(nn.Module):
    def __init__(self, decoder, scale):
        super().__init__()
        self.decoder, self.scale = decoder, scale

    def forward(self, hv, codes, flat, zb=None):
        conditional, labels = self.decoder(hv[0], codes, flat, zb=zb[0])
        unconditional, _ = self.decoder(hv[1], codes, flat, zb=zb[1])
        return self.scale * conditional + (1 - self.scale) * unconditional, labels


class DenseGuidedTeacher(nn.Module):
    """Independent reference for fixture rows with continuous valid audio."""
    def __init__(self, teacher, scale):
        super().__init__()
        self.teacher, self.scale = teacher, scale
        self.dec_p = DenseGuidedDecoder(teacher.dec_p, scale)

    def forward(self, batch, rot, *, role):
        conditional = self.teacher(batch, rot, role=role)
        # Fixture silence is local code zero; both lanes are completely valid.
        silent = torch.arange(32)[None, None, :] * 2048
        unconditional = self.teacher({**batch, 'depth': silent.expand_as(batch['depth'])}, rot, role=role)
        return dict(logits_p=self.scale * conditional['logits_p'] + (1 - self.scale) * unconditional['logits_p'],
                    hv_p=(conditional['hv_p'], unconditional['hv_p']),
                    zb_p=(conditional['zb_p'], unconditional['zb_p']))


class CFGTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(739)

    def test_silence_resets_at_gaps_and_packed_boundaries_preserving_other_lanes(self):
        offsets = torch.arange(32) * 2048
        depth = offsets.expand(2, 9, 32).clone() + 19
        depth[0, 2:4] = -1
        depth[1, :2] = -1
        batch = dict(depth=depth, ids=torch.arange(18).reshape(2, 9), pdepth=depth[0].clone(),
                     present=torch.ones(2, 9), segs=[5, 4])
        silence = np.tile(np.arange(5), (32, 1))
        actual = runtime.silent_pose_view(batch, silence)
        expected_local = [[0, 1, -1, -1, 0, 0, 1, 2, 3], [-1, -1, 0, 1, 2, 0, 1, 2, 3]]
        for channel in range(2):
            for position, local in enumerate(expected_local[channel]):
                expected = torch.full((32,), -1) if local < 0 else offsets + local
                torch.testing.assert_close(actual['depth'][channel, position], expected)
        torch.testing.assert_close(batch['depth'], depth)
        self.assertNotEqual(actual['depth'].data_ptr(), depth.data_ptr())
        for key in ('ids', 'pdepth', 'present', 'segs'):
            self.assertIs(actual[key], batch[key])
        with self.assertRaisesRegex(ValueError, 'shorter'):
            runtime.silent_pose_view(batch, silence[:, :2])

    def test_guidance_blends_logits_before_normalization(self):
        conditional = torch.tensor([[2., -1., .4]])
        unconditional = torch.tensor([[-1., .8, .1]])
        actual = runtime.guided_pose_logits(conditional, unconditional, 3).softmax(-1)
        expected = (3 * conditional - 2 * unconditional).softmax(-1)
        torch.testing.assert_close(actual, expected)
        self.assertIs(runtime.guided_pose_logits(conditional, unconditional, 1), conditional)

    def test_guided_chunked_replay_matches_independent_dense_all_parameter_gradients(self):
        models = [ToyPPM(), ToyPPM().requires_grad_(False), ToyPPM().requires_grad_(False)]
        reference = copy.deepcopy(models)
        reference[2] = DenseGuidedTeacher(reference[2], 3)
        bundle, latents = fixture()
        config = TrainingConfig(decoder_chunk_frames=3, text_chunk_frames=2, pose_teacher_cfg_scale=3,
                                pose_teacher_silence_path='fixture.npy', pose_teacher_silence_sha256='0' * 64)
        with mock.patch.object(runtime, 'load_silence_codes', return_value=np.zeros((32, 20), np.int64)), \
             mock.patch.object(runtime, 'resident_decoders', side_effect=lambda *_: contextlib.nullcontext()):
            metrics, actual_lp, sampled = runtime.replay_update(*models, bundle, latents, config, None,
                                                               torch.device('cpu'))
        expected, expected_lp, expected_mask = dense_expected(*reference, bundle, latents, config)
        expected.backward()
        self.assertAlmostEqual(metrics['loss/direct'], float(expected.detach()), places=6)
        torch.testing.assert_close(actual_lp[sampled], expected_lp.detach()[expected_mask], rtol=2e-6, atol=2e-6)
        for (name, actual), (_, dense) in zip(models[0].named_parameters(), reference[0].named_parameters()):
            with self.subTest(parameter=name):
                torch.testing.assert_close(actual.grad, dense.grad, rtol=3e-5, atol=3e-6)
        self.assertEqual(models[1].calls, [('audio_teacher', False)])
        self.assertEqual(models[2].calls, [('pose_teacher', False), ('pose_teacher', False)])
        self.assertEqual(len(models[2].dec_p.calls), 2 * len(models[1].dec_s.calls))
        self.assertTrue(all(p.grad is None for teacher in models[1:] for p in teacher.parameters()))

    def test_silence_file_identity_and_config_are_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'silence.npy'
            np.save(path, np.zeros((32, 9), np.int64))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(runtime.load_silence_codes(str(path), digest).shape, (32, 9))
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                runtime.load_silence_codes(str(path), '0' * 64)
        config = TrainingConfig(pose_teacher_cfg_scale=3)
        with self.assertRaisesRegex(ValueError, 'digital-silence'):
            config.validate()
        replace(config, pose_teacher_silence_path='fixture.npy', pose_teacher_silence_sha256='0' * 64).validate()

    def test_cpu_cache_is_exact_with_cfg_packed_rows_and_partial_chunks(self):
        bundles, features = zip(fixture(frames=6), fixture(frames=8))
        originals = [ToyPPM(), ToyPPM().requires_grad_(False), ToyPPM().requires_grad_(False)]
        for scale in (1, 3):
            with self.subTest(cfg_scale=scale):
                results = []
                for mode in ('none', 'cpu'):
                    models = copy.deepcopy(originals)
                    config = TrainingConfig(decoder_chunk_frames=3, text_chunk_frames=2,
                                            pose_teacher_cfg_scale=scale, teacher_depth_cache=mode)
                    residency = []

                    def resident(*modules):
                        residency.append(modules)
                        return contextlib.nullcontext()

                    with mock.patch.object(runtime, 'load_silence_codes', return_value=np.zeros((32, 20), np.int64)), \
                         mock.patch.object(runtime, 'resident_decoders', side_effect=resident):
                        metrics, lp, sampled = runtime.replay_update(*models, list(bundles), list(features), config,
                                                                    None, torch.device('cpu'))
                    results.append((metrics, lp, sampled, models))
                    if mode == 'cpu':
                        self.assertEqual(residency[2], (models[0].dec_s,))
                        self.assertEqual(residency[3], (models[0].dec_p,))
                before, after = results
                self.assertEqual(before[0], after[0])
                torch.testing.assert_close(before[1], after[1], rtol=0, atol=0)
                torch.testing.assert_close(before[2], after[2], rtol=0, atol=0)
                for left, right in zip(before[3][0].parameters(), after[3][0].parameters()):
                    torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
                for index, key in ((1, 'dec_s'), (2, 'dec_p')):
                    self.assertEqual(len(getattr(before[3][index], key).calls),
                                     2 * len(getattr(after[3][index], key).calls))


if __name__ == '__main__':
    unittest.main()
