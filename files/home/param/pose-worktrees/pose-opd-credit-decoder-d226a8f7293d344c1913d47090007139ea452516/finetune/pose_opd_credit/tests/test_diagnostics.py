"""Independent coordinate, distribution and opt-in diagnostic checks."""
import copy
import json
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from finetune.pose_opd_credit.diagnostics import PolicyAgreement, CategoricalAgreement, benchmark_prefix
from finetune.pose_opd_credit.data import replay_from_rollout
from finetune.pose_opd_credit.rollout import rollout_row
from finetune.pose_opd_credit.tests.test_rollout import seed_row, FakeCachedEngine
from finetune.pose_opd_credit.tests import test_text_head
from finetune.pose_opd_credit.text_head import chunked_text_objective


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def example(self, frames=31, prefix=7):
        seed = seed_row(frames=frames, prefix=prefix)
        seed.update(row_key=f'row-{prefix}', item=[0, prefix, 1])
        rollout = rollout_row(FakeCachedEngine(), seed, seed=1)
        bundle = replay_from_rollout(seed, rollout.text, rollout.audio, rollout.pose)
        lp = torch.as_tensor(np.concatenate((rollout.behavior_logprobs['text'][:, None],
                    rollout.behavior_logprobs['audio'], rollout.behavior_logprobs['pose']), 1)).clone()
        mask = torch.as_tensor(np.concatenate((rollout.sampled_masks['text'][:, None],
                    np.repeat(rollout.sampled_masks['audio'][:, None], 32, 1),
                    np.repeat(rollout.sampled_masks['pose'][:, None], 32, 1)), 1))
        return bundle, rollout, lp, mask

    def test_bounded_quantiles_worst_coordinates_and_codebooks(self):
        bundle, rollout, lp, mask = self.example()
        first = rollout.prefix + rollout.prefill_frames
        lp[first:first + 3, 0] += torch.tensor([-.1, .2, -.3])
        lp[first + 2, 33 + 14] -= 1.25
        lp[first + 5, 33 + 14] -= 99  # outside the retained first three pose frames
        report = PolicyAgreement(3)
        report.add([bundle], [rollout], lp, mask, row_offset=2)
        result = report.report()
        text = result['streams']['text']
        self.assertEqual(text['count'], 3)
        self.assertAlmostEqual(text['mean_abs'], .2, places=6)
        self.assertAlmostEqual(text['p50_abs'], .2, places=6)
        self.assertAlmostEqual(text['p95_abs'], .29, places=6)
        self.assertAlmostEqual(text['p99_abs'], .298, places=6)
        pose = result['streams']['pose']
        self.assertEqual(pose['count'], 3 * 32)
        self.assertEqual(pose['worst']['codebook'], 14)
        self.assertEqual(pose['worst']['row_slot'], 2)
        self.assertEqual(pose['worst']['model_frame'], 27)
        self.assertEqual(pose['worst']['physical_frame'], 25)
        self.assertEqual(pose['worst']['generated_frame'], 0)
        self.assertAlmostEqual(pose['worst']['signed_delta'], -1.25)
        self.assertEqual(pose['codebooks']['14']['count'], 3)
        json.dumps(result, allow_nan=False)

    def test_packed_rows_and_dummy_forced_positions_are_excluded(self):
        a, b = self.example(), self.example(prefix=11)
        b[0]['student']['dummy'] = True
        lp = torch.cat((a[2], b[2]))
        lp[:10] = 500  # forced prefix is excluded
        lp[len(a[2]):] = 900
        diag = PolicyAgreement(2)
        diag.add([a[0], b[0]], [a[1], b[1]], lp, torch.cat((a[3], torch.zeros_like(b[3]))))
        for result in diag.report()['streams'].values():
            self.assertEqual(result['max_abs'], 0.)
        empty = PolicyAgreement(2)
        empty.add([b[0]], [b[1]], b[2], torch.zeros_like(b[3]))
        self.assertIsNone(empty.report()['streams']['text']['mean_abs'])
        json.dumps(empty.report(), allow_nan=False)

    def test_explicit_prefix_preserves_conditioning_prefill_and_delayed_flush(self):
        source = seed_row(frames=200, prefix=11)
        source['wav'] = np.arange(200 * 1920, dtype=np.float32)
        original = copy.deepcopy(source)
        cropped = benchmark_prefix(source, 128)
        self.assertEqual(cropped['conversation_frames'], 128)
        self.assertEqual(len(cropped['wav']), 128 * 1920)
        self.assertEqual(cropped['segs'], [11 + 128 + 2])
        np.testing.assert_array_equal(cropped['pdepth'][:11 + 27], source['pdepth'][:11 + 27])
        generated = rollout_row(FakeCachedEngine(), cropped, seed=1)
        self.assertEqual(generated.sampled_masks['pose'].sum(), 128 - 25)
        self.assertFalse(generated.sampled_masks['text'][-2:].any())
        self.assertTrue(generated.sampled_masks['pose'][-2:].all())
        for key in ('ids', 'depth', 'pdepth', 'wav', 'pid'):
            np.testing.assert_array_equal(source[key], original[key])
        with self.assertRaises(ValueError):
            benchmark_prefix(source, 27)

    def test_full_categorical_probe_mapping_and_shift_invariant_kl(self):
        bundle, rollout, _, _ = self.example()
        position = rollout.prefix + 27
        actor = torch.arange(16).float() / 10
        rollout.policy_probes = ({'position': position, 'stream': 'pose', 'codebook': 14, 'logits': actor},
                                 {'position': position, 'stream': 'pose', 'codebook': 31, 'logits': actor})
        diag = CategoricalAgreement([bundle], [rollout], row_offset=3)
        logits = torch.zeros(2, 31, 16)
        logits[1, 13] = actor + 2  # identical categorical distribution
        logits[1, 30] = actor.flip(0)
        diag.depth('pose', torch.tensor([position - 1, position]), logits)
        report = diag.report()
        self.assertEqual(report['captured'], 2)
        self.assertEqual(report['missing'], [])
        one, two = report['probes']
        self.assertAlmostEqual(one['kl_actor_to_replay'], 0., places=12)
        self.assertAlmostEqual(one['logit_max_abs_difference'], 2., places=6)
        expected = (actor.double().softmax(0) * (actor.double().log_softmax(0) - actor.flip(0).double().log_softmax(0))).sum()
        self.assertAlmostEqual(two['kl_actor_to_replay'], float(expected), places=12)
        self.assertEqual(two['sampled_action'], 12)
        self.assertEqual(two['row_slot'], 3)

    def test_text_callback_captures_actual_special_replaced_chunks_without_gradient_change(self):
        fixture = test_text_head.TextHeadTests()
        fixture.setUp()
        heads, actions, direct, credit = fixture.fixture()
        reference = [value.detach().clone().requires_grad_() for value in heads]
        captured = []
        kwargs = dict(special_ids=(8, 9, 10), executable_vocab=11, chunk_size=3)
        loss, _, lp = chunked_text_objective(*heads, actions, direct, credit, **kwargs,
            logits_callback=lambda start, value: captured.append((start, value.clone())))
        expected, _, _ = chunked_text_objective(*reference, actions, direct, credit, **kwargs)
        self.assertEqual([start for start, _ in captured], [0, 3, 6])
        logits = torch.cat([value for _, value in captured])
        torch.testing.assert_close(logits, test_text_head.project(*heads[:3], (8, 9, 10), 11))
        torch.testing.assert_close(lp, logits.log_softmax(-1).gather(1, actions[:, None])[:, 0])
        loss.backward(); expected.backward()
        for actual, wanted in zip(heads[:3], reference[:3]):
            torch.testing.assert_close(actual.grad, wanted.grad)

    def test_packed_runtime_capture_uses_reference_pass_and_preserves_gradients(self):
        from finetune.pose_opd_credit import runtime
        from finetune.pose_opd_credit.config import TrainingConfig
        from finetune.pose_opd_credit.tests.test_runtime import ToyPPM, fixture

        class ProbePPM(ToyPPM):
            def forward(self, batch, rot, *, role, teacher=None):
                out = super().forward(batch, rot, role=role, teacher=teacher)
                if role == 'student' and batch.get('policy_probe_text') is not None:
                    batch['policy_probe_text'](0, out['text']['logits'])
                return out

        torch.manual_seed(715)
        models = [ProbePPM(), ProbePPM().requires_grad_(False), ProbePPM().requires_grad_(False)]
        baseline = copy.deepcopy(models)
        examples = [fixture(frames=7), fixture(frames=9)]
        bundles, latents = zip(*examples)
        config = TrainingConfig(decoder_chunk_frames=3)
        rollouts = []
        with torch.no_grad():
            for bundle, lat in examples:
                batch, _, _ = runtime.prepare(bundle, lat, config, torch.device('cpu'), world=1)
                raw, student = bundle['student'], batch['student']
                teacher = models[1](batch['audio_teacher'], None, role='audio_teacher')
                out = models[0](student, None, role='student', teacher=teacher['text'])
                probes = [{'position': int(raw['text_positions'][0]), 'stream': 'text', 'codebook': -1,
                           'logits': out['text']['logits'][0].clone()}]
                for stream, key, codes in (('audio', 's', student['depth'][1]), ('pose', 'p', student['pdepth'])):
                    position = int(raw['light'][0])
                    logits, _ = getattr(models[0], 'dec_' + key)(out['hv_' + key], codes,
                        torch.tensor([position]), zb=out.get('zb_' + key))
                    probes.append({'position': position, 'stream': stream, 'codebook': 0,
                                   'logits': out['logits_' + key][position].clone()})
                    for cb in (14, 31):
                        probes.append({'position': position, 'stream': stream, 'codebook': cb,
                                       'logits': logits[0, cb - 1].clone()})
                rollouts.append(SimpleNamespace(text=raw['ids'][1], audio=raw['depth'][1], pose=raw['pdepth'],
                    sampled_masks=raw['sampled_masks'], prefix=2, prefill_frames=2, policy_probes=probes))
        diag = CategoricalAgreement(bundles, rollouts)
        actual, _, _ = runtime.replay_update(*models, bundles, latents, config, None, torch.device('cpu'), policy_probes=diag)
        expected, _, _ = runtime.replay_update(*baseline, bundles, latents, config, None, torch.device('cpu'))
        self.assertEqual(actual, expected)
        for observed, wanted in zip(models[0].parameters(), baseline[0].parameters()):
            torch.testing.assert_close(observed.grad, wanted.grad, rtol=0, atol=0)
        report = diag.report()
        self.assertEqual(report['requested'], 14)
        self.assertEqual(report['captured'], 14)
        self.assertEqual(report['missing'], [])
        self.assertTrue(all(abs(probe['kl_actor_to_replay']) < 1e-12 for probe in report['probes']))


if __name__ == '__main__':
    unittest.main()
