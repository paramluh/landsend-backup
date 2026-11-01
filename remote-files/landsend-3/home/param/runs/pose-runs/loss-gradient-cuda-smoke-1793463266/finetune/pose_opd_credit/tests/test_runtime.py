"""CPU end-to-end replay proof using small, causally faithful PPM modules."""
import contextlib
import copy
import types
import unittest
from dataclasses import replace
from unittest import mock

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from finetune.pose_opd_credit import runtime
from finetune.pose_opd_credit.config import TrainingConfig
from finetune.pose_opd_credit.data import teacher_views


def shift(x, segments=None):
    if segments is not None:
        return torch.cat([shift(part) for part in x.split(segments)])
    return torch.cat((torch.zeros_like(x[:1]), x[:-1]))


class ToyDecoder(nn.Module):
    """Each target cbk consumes its own prefix cb0..k-1 and tower/link roots."""
    def __init__(self, hidden=4, vocab=5):
        super().__init__()
        self.vocab = vocab
        self.proj = nn.Linear(hidden, hidden)
        self.embeddings = nn.Parameter(torch.randn(32, vocab, hidden) * .15)
        self.heads = nn.Parameter(torch.randn(31, hidden, vocab) * .2)
        self.calls = []
        self.segment_history = []

    def forward(self, hv, depth, flat, zb=None):
        self.calls.append((flat.detach().clone(), torch.is_grad_enabled()))
        codes = depth[flat].remainder(2048).clamp_min(0)
        prefix = self.embeddings[torch.arange(32)[None], codes].cumsum(1)[:, :31]
        root = self.proj(hv[flat])
        if zb is not None:
            root = root + zb[flat]
        h = torch.tanh(root[:, None] + prefix)
        return torch.einsum('nch,chv->ncv', h, self.heads), codes[:, 1:]


class ToyPPM(nn.Module):
    def __init__(self, hidden=4, vocab=7, media_vocab=5):
        super().__init__()
        self.text_embedding = nn.Embedding(vocab, hidden)
        self.audio_embedding = nn.Embedding(media_vocab, hidden)
        self.pose_embedding = nn.Embedding(media_vocab, hidden)
        self.backbone = nn.Linear(hidden, hidden)
        self.speech_tower = nn.Linear(hidden, hidden)
        self.pose_tower = nn.Linear(hidden, hidden)
        self.link = nn.Linear(hidden, hidden)
        self.text_head = nn.Linear(hidden, vocab)
        self.speech_cb0 = nn.Linear(hidden, media_vocab)
        self.pose_cb0 = nn.Linear(hidden, media_vocab)
        self.dec_s = ToyDecoder(hidden, media_vocab)
        self.dec_p = ToyDecoder(hidden, media_vocab)
        self.calls = []

    @staticmethod
    def media_embedding(module, codes):
        local = codes.remainder(2048).clamp_min(0)
        return (module(local) * (codes >= 0)[..., None]).mean(1)

    def forward(self, b, rot, *, role, teacher=None):
        self.calls.append((role, torch.is_grad_enabled()))
        segments = b['segs']
        self.segment_history = getattr(self, 'segment_history', []) + [list(segments)]
        text = self.text_embedding(b['ids'][1])
        speech = self.media_embedding(self.audio_embedding, b['depth'][1])
        pose = self.media_embedding(self.pose_embedding, b['pdepth']) if 'pdepth' in b else 0.
        incoming = text + .2 * speech + .2 * pose
        if 'vlat' in b:
            # Shared external A conditioning affects both duplex models.
            a = b['vlat'].reshape(1, -1, 4, 1280).mean((0, 2, 3))
            incoming = incoming.index_add(0, b['vpos'], a[:, None].expand(-1, incoming.shape[1]))
        history = torch.cat([part.cumsum(0) / torch.arange(1, len(part) + 1)[:, None]
                             for part in incoming.split(segments)])
        h = torch.tanh(self.backbone(history))
        out = {'text': {}}
        if role != 'pose_teacher':
            logits = self.text_head(h[b['sel']])
            if role == 'audio_teacher':
                out['text'] = {'logits': logits.detach()}
            else:
                sl, tl = logits.log_softmax(-1), teacher['logits'].detach().log_softmax(-1)
                kl = (sl.exp() * (sl - tl)).sum(-1)
                lp = sl.gather(1, b['y'][:, None]).squeeze(1)
                loss = (b['text_weight'] * kl + b['text_credit'] * (lp - lp.detach())).sum()
                out['text'] = {'loss': loss, 'kl': kl.detach(), 'logp': lp.detach(), 'logits': logits}
            hv = torch.tanh(self.speech_tower(shift(h, segments) + text + .2 * shift(speech, segments)))
            out.update(hv_s=hv, logits_s=self.speech_cb0(hv))
        if role != 'audio_teacher':
            hv = torch.tanh(self.pose_tower(shift(h, segments) + text + .2 * shift(pose, segments)))
            out.update(hv_p=hv, logits_p=self.pose_cb0(hv), zb_p=shift(self.link(h), segments))
        return out


def fixture(dummy=False, frames=6):
    generator = np.random.default_rng(81)
    prefix, prefill, delay = 2, 2, 2
    length = prefix + frames + delay
    masks = {name: (np.arange(length) >= prefix + prefill + d) &
                   (np.arange(length) < prefix + frames + d)
             for name, d in (('text', 0), ('audio', delay), ('pose', delay))}
    offsets = np.arange(32)[None] * 2048
    ids = generator.integers(0, 7, (2, length), dtype=np.int64)
    depth = generator.integers(0, 5, (2, length, 32), dtype=np.int64) + offsets
    pose = generator.integers(0, 5, (length, 32), dtype=np.int64) + offsets
    positions = np.flatnonzero(masks['text'])
    raw = dict(ids=ids, depth=depth, pdepth=pose, present=np.ones((2, length), np.float32),
               pid=np.arange(length, dtype=np.int64), vin=np.array([prefix, frames]),
               segs=[length], kind='pose', dummy=dummy, sampled_masks=masks,
               text_positions=positions, sel=positions - 1, y=ids[1, positions],
               light=np.flatnonzero(masks['audio']), plight=np.flatnonzero(masks['pose']))
    return teacher_views(raw, pad_id=0), torch.randn(1, 4 * frames, 1280) * .05


def dense_expected(student, audio_teacher, pose_teacher, bundle, latents, config, normalization=None):
    """All depth frames retained together; independent dense ancestry matrix."""
    b, weights, masks = runtime.prepare(bundle, latents, config, torch.device('cpu'), world=1,
                                        normalization=normalization)
    s = b['student']
    with torch.no_grad():
        audio = audio_teacher(b['audio_teacher'], None, role='audio_teacher')
        pose = pose_teacher(b['pose_teacher'], None, role='pose_teacher')
    out = student(s, None, role='student', teacher=audio['text'])
    costs, logp = torch.zeros_like(weights), torch.zeros_like(weights)

    def distributions(sl, tl, ids):
        sl, tl = sl.log_softmax(-1), tl.detach().log_softmax(-1)
        return (sl.exp() * (sl - tl)).sum(-1), sl.gather(-1, ids[..., None]).squeeze(-1)

    text_kl, text_lp = distributions(out['text']['logits'], audio['text']['logits'], s['y'])
    costs[s['text_positions'], 0], logp[s['text_positions'], 0] = text_kl, text_lp
    for key, teacher, tout, codes, flat, offset in (
        ('s', audio_teacher, audio, s['depth'][1], s['light'], 1),
        ('p', pose_teacher, pose, s['pdepth'], s['plight'], 33)):
        k, lp = distributions(out['logits_' + key], tout['logits_' + key], codes[:, 0])
        costs[:, offset], logp[:, offset] = k, lp
        sl, ids = getattr(student, 'dec_' + key)(out['hv_' + key], codes, flat, zb=out.get('zb_' + key))
        with torch.no_grad():
            tl, _ = getattr(teacher, 'dec_' + key)(tout['hv_' + key], codes, flat, zb=tout.get('zb_' + key))
        k, lp = distributions(sl, tl, ids)
        costs[flat, offset + 1:offset + 32], logp[flat, offset + 1:offset + 32] = k, lp
    coordinates = runtime.event_layout(len(weights), weights.device, s['segs'])
    gap = coordinates.frames[None] - coordinates.frames[:, None]
    previous = config.lookback_frames - (config.window_mode == 'including_current')
    same = (gap == 0) & (
        ((coordinates.streams[:, None] == 0) & (coordinates.streams[None] != 0)) |
        ((coordinates.streams[:, None] == coordinates.streams[None]) &
         (coordinates.streams[:, None] != 0) &
         (coordinates.codebooks[:, None] < coordinates.codebooks[None])))
    ancestors = (((gap >= 1) & (gap <= previous)) | same) & masks.flatten()[:, None] & masks.flatten()[None]
    ancestors &= coordinates.rows[:, None] == coordinates.rows[None]
    weighted = costs * weights
    coefficient = ancestors.float() @ weighted.detach().flatten()
    return weighted.sum() + (coefficient * (logp.flatten() - logp.detach().flatten())).sum(), logp, masks


class ReplayTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(498)
        torch.set_num_threads(1)
        self.config = TrainingConfig(decoder_chunk_frames=3, text_chunk_frames=2)
        self.models = [ToyPPM(), ToyPPM().requires_grad_(False), ToyPPM().requires_grad_(False)]
        self.resident = mock.patch.object(runtime, 'resident_decoders', side_effect=lambda *_: contextlib.nullcontext())
        self.resident.start()
        self.addCleanup(self.resident.stop)

    def test_two_pass_chunk_backward_matches_independent_dense_objective(self):
        bundle, latents = fixture()
        reference_models = copy.deepcopy(self.models)
        before = {name: p.detach().clone() for name, p in self.models[0].named_parameters()}
        metrics, replay_lp, sampled = runtime.replay_update(*self.models, bundle, latents, self.config,
                                                           None, torch.device('cpu'))
        expected, expected_lp, expected_masks = dense_expected(*reference_models, bundle, latents, self.config)
        expected.backward()
        self.assertAlmostEqual(metrics['loss/direct'], float(expected.detach()), places=6)
        torch.testing.assert_close(replay_lp[sampled], expected_lp.detach()[expected_masks], rtol=2e-6, atol=2e-6)
        for (name, actual), (_, reference) in zip(self.models[0].named_parameters(), reference_models[0].named_parameters()):
            with self.subTest(parameter=name):
                torch.testing.assert_close(actual.grad, reference.grad, rtol=3e-5, atol=3e-6)
                torch.testing.assert_close(actual.detach(), before[name], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for t in self.models[1:] for p in t.parameters()))
        self.assertEqual(self.models[0].calls, [('student', False), ('student', True)])
        # CB14's decoder output and its prefix embeddings both receive gradient.
        self.assertGreater(float(self.models[0].dec_p.heads.grad[13].abs().sum()), 0.)
        self.assertGreater(float(self.models[0].dec_p.embeddings.grad[13].abs().sum()), 0.)

    def test_padding_and_chunk_geometry_do_not_change_gradients(self):
        bundle, latents = fixture()
        results = []
        for chunk in (1, 3, 9):
            models = copy.deepcopy(self.models)
            config = replace(self.config, decoder_chunk_frames=chunk)
            metrics, _, _ = runtime.replay_update(*models, bundle, latents, config, None, torch.device('cpu'))
            results.append((metrics, {name: p.grad.clone() for name, p in models[0].named_parameters()}))
        for metrics, gradients in results[1:]:
            self.assertAlmostEqual(metrics['loss/direct'], results[0][0]['loss/direct'], places=6)
            for name, gradient in gradients.items():
                torch.testing.assert_close(gradient, results[0][1][name], rtol=3e-5, atol=3e-6)

    def test_dummy_row_runs_collectives_but_has_zero_gradients(self):
        bundle, latents = fixture(dummy=True)
        metrics, _, sampled = runtime.replay_update(*self.models, bundle, latents, self.config,
                                                    None, torch.device('cpu'))
        self.assertEqual(metrics['loss/direct'], 0.)
        self.assertFalse(bool(sampled.any()))
        self.assertTrue(self.models[0].dec_s.calls and self.models[0].dec_p.calls)
        for param in self.models[0].parameters():
            self.assertIsNotNone(param.grad)
            torch.testing.assert_close(param.grad, torch.zeros_like(param.grad), rtol=0, atol=0)

    def test_row_and_position_mean_normalization(self):
        bundle, _ = fixture()
        raw = bundle['student']
        config = replace(self.config, modality_weights={'text': .2, 'audio': .3, 'pose': .5})
        for reduction in ('row_mean', 'position_mean'):
            with self.subTest(reduction=reduction):
                cfg = replace(config, reduction=reduction)
                weights, sampled = runtime.loss_weights(raw, cfg, torch.device('cpu'))
                torch.testing.assert_close(weights[:, 0].sum(), torch.tensor(.2))
                torch.testing.assert_close(weights[:, 1:33].sum(), torch.tensor(.3))
                torch.testing.assert_close(weights[:, 33:].sum(), torch.tensor(.5))
                self.assertTrue(bool((weights[~sampled] == 0).all()))
        # Simulate another rank with twice as many generated positions. Global
        # position means and means of row means must differ for unequal lengths.
        cfg = replace(config, reduction='position_mean')
        with mock.patch.object(runtime.dist, 'is_initialized', return_value=True), \
             mock.patch.object(runtime.dist, 'all_reduce', side_effect=lambda x, **_: x.mul_(3)):
            w, _ = runtime.loss_weights(raw, cfg, torch.device('cpu'), world=2)
        self.assertAlmostEqual(float(w[:, 0].sum()), .2 * 2 / 3, places=6)
        cfg = replace(config, reduction='row_mean')
        with mock.patch.object(runtime.dist, 'is_initialized', return_value=True), \
             mock.patch.object(runtime.dist, 'all_reduce', side_effect=lambda x, **_: x.fill_(2)):
            w, _ = runtime.loss_weights(raw, cfg, torch.device('cpu'), world=2)
        self.assertAlmostEqual(float(w[:, 0].sum()), .2, places=6)

    def test_chunk_indices_align_rank_counts_and_pad_with_valid_codes(self):
        positions = torch.tensor([4, 5, 6, 7])
        with mock.patch.object(runtime.dist, 'is_initialized', return_value=True), \
             mock.patch.object(runtime.dist, 'all_reduce', side_effect=lambda x, **_: x.fill_(4)):
            chunks = list(runtime.chunk_indices(positions, 3))
        self.assertEqual([valid for _, valid in chunks], [3, 1, 0, 0])
        self.assertTrue(all(flat.shape == (3,) for flat, _ in chunks))
        torch.testing.assert_close(chunks[1][0], torch.tensor([7, 4, 4]))
        torch.testing.assert_close(chunks[2][0], torch.tensor([4, 4, 4]))
        with self.assertRaises(ValueError):
            list(runtime.chunk_indices(torch.zeros(0, dtype=torch.int64), 3))

    def test_pose_depth_two_to_one_keeps_modalities_equal_and_scores_every_code(self):
        bundle, _ = fixture()
        config = replace(self.config, pose_codebook_weights=(1 / 3,) + (2 / 93,) * 31)
        for reduction in ('row_mean', 'position_mean'):
            with self.subTest(reduction=reduction):
                weights, sampled = runtime.loss_weights(bundle['student'], replace(config, reduction=reduction),
                                                       torch.device('cpu'))
                torch.testing.assert_close(weights[:, 0].sum(), torch.tensor(1 / 3))
                torch.testing.assert_close(weights[:, 1:33].sum(), torch.tensor(1 / 3))
                torch.testing.assert_close(weights[:, 33:].sum(), torch.tensor(1 / 3))
                torch.testing.assert_close(weights[:, 1].sum(), torch.tensor(1 / 6))
                torch.testing.assert_close(weights[:, 2:33].sum(), torch.tensor(1 / 6))
                torch.testing.assert_close(weights[:, 33].sum(), torch.tensor(1 / 9))
                torch.testing.assert_close(weights[:, 34:].sum(), torch.tensor(2 / 9))
                self.assertTrue(bool((weights[sampled] > 0).all()))
                self.assertTrue(bool((weights[~sampled] == 0).all()))

    def test_same_a_conditioning_and_native_empty_monoplex(self):
        bundle, latents = fixture()
        prepared, _, _ = runtime.prepare(bundle, latents, self.config, torch.device('cpu'), world=1)
        self.assertIs(prepared['student']['vlat'], prepared['audio_teacher']['vlat'])
        torch.testing.assert_close(prepared['student']['vpos'], prepared['audio_teacher']['vpos'])
        self.assertNotIn('vlat', prepared['pose_teacher'])
        self.assertNotIn('pdepth', prepared['audio_teacher'])
        self.assertTrue(bool((prepared['pose_teacher']['depth'][0] == -1).all()))

    def test_native_decoder_enters_layer_hooks_and_checkpoints_gradients(self):
        # FSDP gathers parameters through module __call__ hooks. Calling a
        # layer's unbound implementation would bypass them, even if CPU math
        # happened to work. Exercise the actual decoder driver, not ToyDecoder.
        class Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 4)

            def forward(self, x, cos, sin):
                return self.linear(x.cumsum(1)).sin() + x

        class Decoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(4, 4)
                self.cb0_emb = nn.Embedding(2048, 4)
                self.model = nn.Module()
                self.model.embed_tokens = nn.Embedding(31 * 2048, 4)
                self.model.layers = nn.ModuleList([Layer(), Layer()])
                self.model.norm = nn.Module()
                self.model.norm.weight = nn.Parameter(torch.ones(4))
                self.model.norm.variance_epsilon = 1e-6
                self.model.rotary_emb = lambda x, _: (torch.ones_like(x), torch.zeros_like(x))
                self.heads = nn.Linear(4, 31 * 2048, bias=False)

        decoder = Decoder()
        called = [0, 0]

        def hook(index):
            def on_enter(*_):
                called[index] += 1
            return on_enter

        for i, layer in enumerate(decoder.model.layers):
            layer.register_forward_pre_hook(hook(i))
        hv = torch.randn(2, 4, requires_grad=True)
        depth = (torch.arange(32) * 2048)[None].expand(2, -1)
        flat = torch.tensor([0, 1])
        with mock.patch.object(runtime, 'M', types.SimpleNamespace(checkpoint=checkpoint)):
            with torch.no_grad():
                eager, labels = runtime._decoder_forward(decoder, hv, depth, flat)
            output, actual_labels = runtime._decoder_forward(decoder, hv, depth, flat)
            self.assertEqual(called, [2, 2])
            torch.testing.assert_close(output, eager)
            torch.testing.assert_close(actual_labels, labels)
            output.square().mean().backward()
        self.assertTrue(all(n >= 3 for n in called))
        self.assertTrue(bool(torch.isfinite(hv.grad).all()))
        self.assertGreater(float(hv.grad.abs().sum()), 0.)
        expected_gradients = [parameter.grad.clone() for parameter in decoder.parameters()]
        expected_hv = hv.grad.clone()
        decoder.zero_grad(set_to_none=True)
        hv.grad = None
        decoder._opd_checkpoint_depth_decoder = False
        called[:] = [0, 0]
        forbidden = mock.Mock(side_effect=AssertionError('Depth checkpoint called while disabled'))
        with mock.patch.object(runtime, 'M', types.SimpleNamespace(checkpoint=forbidden)):
            direct, _ = runtime._decoder_forward(decoder, hv, depth, flat)
            direct.square().mean().backward()
        self.assertEqual(called, [1, 1])
        forbidden.assert_not_called()
        torch.testing.assert_close(direct, output, rtol=0, atol=0)
        torch.testing.assert_close(hv.grad, expected_hv, rtol=0, atol=0)
        for parameter, expected in zip(decoder.parameters(), expected_gradients):
            torch.testing.assert_close(parameter.grad, expected, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
