"""Cached per-loss backward matches six independent dense gradient objectives."""
import contextlib
import copy
import types
import unittest
from unittest import mock

import torch
from torch import nn

from finetune.pose_opd_credit import runtime
from finetune.pose_opd_credit.config import TrainingConfig
from finetune.pose_opd_credit.loss_diagnostics import MODALITIES, STREAM_SLICES
from finetune.pose_opd_credit.tests.test_runtime import ToyPPM, fixture
from finetune.pose_opd_credit.text_head import chunked_text_objective


def dense_terms(student, audio_teacher, pose_teacher, bundle, latents, config):
    """Full-vocabulary/full-depth autograd and an explicit causal ancestor matrix."""
    b, weights, masks = runtime.prepare(bundle, latents, config, torch.device('cpu'), world=1)
    s = b['student']
    with torch.no_grad():
        audio = audio_teacher(b['audio_teacher'], None, role='audio_teacher')
        pose = pose_teacher(b['pose_teacher'], None, role='pose_teacher')
    out = student(s, None, role='student', teacher=audio['text'])
    costs, logp = torch.zeros_like(weights), torch.zeros_like(weights)

    def distributions(sl, tl, labels):
        p, q = sl.softmax(-1), tl.detach().softmax(-1)
        return (p * (p.log() - q.log())).sum(-1), p.log().gather(-1, labels[..., None]).squeeze(-1)

    k, lp = distributions(out['text']['logits'], audio['text']['logits'], s['y'])
    costs[s['text_positions'], 0], logp[s['text_positions'], 0] = k, lp
    for key, teacher, tout, codes, positions, offset in (
            ('s', audio_teacher, audio, s['depth'][1], s['light'], 1),
            ('p', pose_teacher, pose, s['pdepth'], s['plight'], 33)):
        k, lp = distributions(out['logits_' + key], tout['logits_' + key], codes[:, 0])
        costs[:, offset], logp[:, offset] = k, lp
        sl, labels = getattr(student, 'dec_' + key)(out['hv_' + key], codes, positions, zb=out.get('zb_' + key))
        with torch.no_grad():
            tl, _ = getattr(teacher, 'dec_' + key)(tout['hv_' + key], codes, positions, zb=tout.get('zb_' + key))
        k, lp = distributions(sl, tl, labels)
        costs[positions, offset + 1:offset + 32], logp[positions, offset + 1:offset + 32] = k, lp
    coordinates = runtime.event_layout(len(weights), weights.device, s['segs'])
    gap = coordinates.frames[None] - coordinates.frames[:, None]
    same = (gap == 0) & (
        ((coordinates.streams[:, None] == 0) & (coordinates.streams[None] != 0)) |
        ((coordinates.streams[:, None] == coordinates.streams[None]) &
         (coordinates.streams[:, None] != 0) &
         (coordinates.codebooks[:, None] < coordinates.codebooks[None])))
    preceding = config.lookback_frames - (config.window_mode == 'including_current')
    ancestors = (((gap >= 1) & (gap <= preceding)) | same) & masks.flatten()[:, None] & masks.flatten()[None]
    ancestors &= coordinates.rows[:, None] == coordinates.rows[None]
    result = {}
    for name, columns in zip(MODALITIES, STREAM_SLICES):
        weighted = torch.zeros_like(costs)
        weighted[:, columns] = costs[:, columns] * weights[:, columns]
        coefficient = ancestors.float() @ weighted.detach().flatten()
        result['direct/' + name] = weighted.sum()
        result['credit/' + name] = (coefficient * (logp.flatten() - logp.detach().flatten())).sum()
    return result


class LossGradientReplayTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(355)
        torch.set_num_threads(1)
        self.models = [ToyPPM(), ToyPPM().requires_grad_(False), ToyPPM().requires_grad_(False)]
        self.config = TrainingConfig(decoder_chunk_frames=3, text_chunk_frames=2, teacher_depth_cache='cpu')
        self.resident = mock.patch.object(runtime, 'resident_decoders', side_effect=lambda *_: contextlib.nullcontext())
        self.resident.start()
        self.addCleanup(self.resident.stop)

    def test_all_six_cached_gradients_match_dense_reference_without_rescoring(self):
        bundle, latents = fixture()
        reference = copy.deepcopy(self.models)
        terms = dense_terms(*reference, bundle, latents, self.config)
        parameters = list(reference[0].parameters())
        expected = {name: torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True)
                    for name, value in terms.items()}
        cache = {}
        runtime.replay_update(*self.models, bundle, latents, self.config, None, torch.device('cpu'),
                              loss_gradient_cache=cache)
        student, audio, pose = self.models
        normal_gradients = [None if p.grad is None else p.grad.clone() for p in student.parameters()]
        summed = [torch.zeros_like(p) for p in student.parameters()]
        temporal_calls = len(student.calls)
        with mock.patch.object(audio, 'forward', side_effect=AssertionError('Teacher replayed')), \
                mock.patch.object(pose, 'forward', side_effect=AssertionError('Teacher replayed')), \
                mock.patch.object(runtime, 'causal_credit_weights', side_effect=AssertionError('Credit recomputed')), \
                mock.patch.object(runtime, 'scored_categorical_statistics', side_effect=AssertionError('Student rescored')):
            for term in terms:
                student.zero_grad(set_to_none=True)
                before = [len(student.dec_s.calls), len(student.dec_p.calls)]
                runtime.replay_loss_gradient(student, cache, term, None, torch.device('cpu'))
                actual = [p.grad for p in student.parameters()]
                for total, (name, p), value, wanted in zip(summed, student.named_parameters(), actual, expected[term]):
                    value = torch.zeros_like(p) if value is None else value
                    wanted = torch.zeros_like(p) if wanted is None else wanted
                    total.add_(value)
                    torch.testing.assert_close(value, wanted, rtol=8e-5, atol=3e-6,
                        msg=lambda message: f'{term} / {name}: {message}')
                after = [len(student.dec_s.calls), len(student.dec_p.calls)]
                expected_calls = {'direct/text': [0, 0], 'direct/audio': [2, 0], 'direct/pose': [0, 2]}.get(term, [2, 2])
                self.assertEqual([a - b for a, b in zip(after, before)], expected_calls)
        self.assertEqual(len(student.calls) - temporal_calls, 6)
        for total, wanted in zip(summed, normal_gradients):
            if wanted is not None:
                torch.testing.assert_close(total, wanted, rtol=8e-5, atol=3e-6)
        self.assertTrue(all(grad for _, grad in student.calls[temporal_calls:]))
        self.assertTrue(all(p.grad is None for model in self.models[1:] for p in model.parameters()))

    def test_capture_does_not_change_normal_gradients_and_cache_has_no_graph(self):
        bundle, latents = fixture()
        original = copy.deepcopy(self.models)
        baseline = runtime.replay_update(*original, bundle, latents, self.config, None, torch.device('cpu'))
        cache = {}
        actual = runtime.replay_update(*self.models, bundle, latents, self.config, None, torch.device('cpu'),
                                       loss_gradient_cache=cache)
        self.assertEqual(baseline[0], actual[0])
        for left, right in zip(original[0].parameters(), self.models[0].parameters()):
            torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
        for left, right in zip(original, self.models):
            self.assertEqual(left.calls, right.calls)
            self.assertEqual(len(left.dec_s.calls), len(right.dec_s.calls))
            self.assertEqual(len(left.dec_p.calls), len(right.dec_p.calls))

        def check(value):
            if torch.is_tensor(value):
                self.assertFalse(value.requires_grad)
                self.assertIsNone(value.grad_fn)
            elif isinstance(value, dict):
                for item in value.values():
                    check(item)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    check(item)
        check(cache)

    def test_native_direct_branch_flags_skip_unused_towers_and_text_projection(self):
        class Input(nn.Module):
            def forward(self, *args):
                return torch.ones(3, 4)
        class Head(nn.Module):
            def forward(self, h, *args, **kwargs):
                return {'hn': h, 'loss': h.sum()}
        class Tower(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0
            def forward(self, ids, codes, hn, *args):
                self.calls += 1
                return hn, hn
        model = types.SimpleNamespace(inp=Input(), layers=[], head=Head(), tower_s=Tower(), tower_p=Tower(),
                                      decp_bb=nn.Identity(), decp_bb_alpha=1.)
        batch = {'ids': torch.zeros(2, 3), 'depth': torch.zeros(2, 3, 32), 'present': torch.ones(2, 3),
                 'pdepth': torch.zeros(3, 32), 'pid': torch.arange(3), 'segs': [3]}
        native = types.SimpleNamespace(Ctx=lambda *_: None, shift1=lambda x, *_: x,
                                      FO=types.SimpleNamespace(rms=lambda x, w, _: x * w))
        with mock.patch.object(runtime, 'M', native):
            for media, expected in (((), (0, 0)), (('s',), (1, 0)), (('p',), (1, 1))):
                runtime._ppm_forward(model, {**batch, 'diagnostic_media': media}, lambda *_: None,
                                     role='student', teacher={})
                self.assertEqual((model.tower_s.calls, model.tower_p.calls), expected)
            head = types.SimpleNamespace(norm=types.SimpleNamespace(weight=torch.ones(4), variance_epsilon=1e-6),
                                         proj=mock.Mock(side_effect=AssertionError('Unused text projection')))
            result = runtime._head_forward(head, torch.ones(3, 4), {'skip_text_loss': True}, 'student')
            self.assertEqual(float(result['loss']), 0.)
            head.proj.assert_not_called()

    def test_credit_only_text_backward_matches_full_objective_without_teacher_projection(self):
        heads = [torch.randn(shape, dtype=torch.float64, requires_grad=True)
                 for shape in ((5, 3), (8, 3), (1, 3), (5, 4), (8, 4), (1, 4))]
        actions = torch.arange(5)
        weights, credit = torch.zeros(5), torch.rand(5)
        expected = chunked_text_objective(*heads, actions, weights, credit,
            special_ids=(6,), executable_vocab=7, chunk_size=2)
        expected_grad = torch.autograd.grad(expected[0], heads[:3])
        # NaN teacher inputs make any accidentally used teacher projection/KL
        # visible while the exact score-only objective remains finite.
        teacher = [torch.full_like(value, float('nan')) for value in heads[3:]]
        actual = chunked_text_objective(*heads[:3], *teacher, actions, weights, credit,
            special_ids=(6,), executable_vocab=7, chunk_size=2, credit_only=True)
        actual_grad = torch.autograd.grad(actual[0], heads[:3])
        for left, right in zip(actual_grad, expected_grad):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
