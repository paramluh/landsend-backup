"""Full-support categorical CDF, independent RNG tables and cached histories."""
import types
import unittest

import numpy as np
import torch

from finetune.pose_opd_credit.rollout import inverse_cdf_sample, make_fastppm_class, resize_depth_cache


class Decoder:
    def __init__(self, rows):
        self.dec = types.SimpleNamespace(cb0_emb=torch.nn.Embedding(8, 2))
        self.K = [torch.zeros(rows, 64, 1, 2)]
        self.V = [torch.zeros_like(self.K[0])]
        self.seen = []
    def slot0(self, hv, zb):
        return hv if zb is None else hv + zb
    def step(self, x, position):
        self.seen.append(x.clone())
        return x[:, -1]
    def head(self, hidden, codebook):
        return torch.arange(8)[None].float() * (1 + hidden[:, :1] / 20 + codebook / 32)
    def emb(self, code, codebook):
        return code[:, None, None].expand(-1, 1, 2).float()


class Base:
    def __init__(self, **kwargs):
        self.NI, self.pose, self.V, self.H, self.lmax = kwargs['n_inst'], True, 8, 2, 100
        self.d0 = self.dl = self.dp = torch.device('cpu')
        self.Tv = {self.dl: torch.full((self.NI,), 3, dtype=torch.long)}
        self.arI = {self.dl: torch.arange(self.NI)}
        self.m = types.SimpleNamespace(inp=torch.nn.Identity())
        self.dec_s, self.dec_p = Decoder(self.NI), Decoder(self.NI)
        self.gen_t = tuple(torch.Generator() for _ in range(self.NI))
        self.gen_a = tuple(torch.Generator() for _ in range(self.NI))
        self.gen_p = tuple(torch.Generator() for _ in range(self.NI))
        self.depB = torch.full((self.NI, self.lmax, 32), -1, dtype=torch.long)
        self.pdepB = torch.full_like(self.depB, -1)
        self.pk = {self.dl: torch.zeros(self.NI, 33, dtype=torch.long)}
        self.pkp = {self.dp: torch.zeros(self.NI, 32, dtype=torch.long)}
        self.off_t = self.offp = torch.arange(32) * 2048
        self.bblink = False
        self.adapt = torch.zeros(self.NI, 5)
        self.hn_prev = torch.zeros(self.NI, 2)
        self.zt, self.probs, self.ptok = torch.zeros(self.NI, 8), torch.zeros(self.NI, 2), torch.zeros(self.NI, 2)
        self.laneB = torch.zeros(self.NI, self.lmax, dtype=torch.long)
        self.tok_now = torch.zeros(self.NI, dtype=torch.long)
        self.ring = types.SimpleNamespace(push=lambda _: None)
    def text_logits(self, _hidden):
        return torch.tensor([[0., 1., 2., 3., 4., 5., 1000., 1000.]]).repeat(self.NI, 1)
    def start(self, instances, seeds, params, f0):
        for row, seed in enumerate(seeds):
            self.gen_t[row].manual_seed(seed + 1)
            self.gen_a[row].manual_seed(seed)
            self.gen_p[row].manual_seed(seed + 2)
    def _speech(self, full):
        assert not full
        self.hv_s = torch.zeros(self.NI, 2)
        self.z0 = torch.arange(8).float()[None].expand(self.NI, -1)
    def _pose(self, full, cfg):
        assert not full and not cfg
        self.hv_p = torch.zeros(self.NI, 2)
        self.zp0 = torch.arange(8).float()[None].expand(self.NI, -1)


def make_engine(rows, **kwargs):
    def forbidden(*_):
        raise AssertionError('Fast sampler must not call per-row multinomial')
    cls = make_fastppm_class(types.SimpleNamespace(FastPPM=Base, row_multinomial=forbidden, PAD=1, NW=2))
    return cls(n_inst=rows, executable_vocab_size=8, sampler_backend='inverse_cdf', depth_cache_positions=32, **kwargs)


class SamplingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_exact_support_boundaries_and_unnormalized_weights(self):
        weights = torch.tensor([[0., 2., 0., 6., 0.]]).expand(5, -1)
        uniforms = torch.tensor([0., .24999, .25, .9, 1. - 2 ** -53], dtype=torch.float64)
        self.assertEqual(inverse_cdf_sample(weights, uniforms).tolist(), [1, 1, 3, 3, 3])
        # This positive category disappears in a float32 cumulative sum.
        tiny = torch.tensor([[1., 2 ** -30, 1.]], dtype=torch.float32)
        target = torch.tensor([(1 + 2 ** -31) / (2 + 2 ** -30)], dtype=torch.float64)
        self.assertEqual(inverse_cdf_sample(tiny, target).item(), 1)
        full = torch.zeros(1, 151682)
        full[0, -1] = 1
        self.assertEqual(inverse_cdf_sample(full, torch.tensor([.5], dtype=torch.float64)).item(), 151681)

    def test_uniform_grid_recovers_categorical_mass(self):
        n = 10000
        weights = torch.tensor([[.1, .2, .7]]).expand(n, -1)
        uniforms = (torch.arange(n, dtype=torch.float64) + .5) / n
        self.assertEqual(torch.bincount(inverse_cdf_sample(weights, uniforms)).tolist(), [1000, 2000, 7000])

    def test_uniforms_are_reproducible_and_independent_of_padded_teammates(self):
        alone, batch = make_engine(1), make_engine(2)
        short = {'ids': np.zeros((2, 10), np.int64)}
        long = {'ids': np.zeros((2, 90), np.int64)}
        try:
            alone.start([short], [81])
            batch.start([short, long], [81, 17])
            saved = {}
            for stream in ('text', 'audio', 'pose'):
                a, b = alone._opd_uniforms[stream], batch._opd_uniforms[stream]
                torch.testing.assert_close(a[0, :10], b[0, :10], rtol=0, atol=0)
                self.assertTrue(bool(((b >= 0) & (b < 1)).all()))
                self.assertEqual(b.dtype, torch.float64)
                saved[stream] = b.clone()
            batch.start([short, long], [81, 17])
            for stream, expected in saved.items():
                torch.testing.assert_close(batch._opd_uniforms[stream], expected, rtol=0, atol=0)
        finally:
            alone.close(); batch.close()

    def test_media_draws_follow_own_prefix_and_record_actual_log_probabilities(self):
        engine = make_engine(2)
        try:
            engine.start([{'ids': np.zeros((2, 10), np.int64)}] * 2, [81, 17])
            for stream in ('audio', 'pose'):
                decoder = engine.dec_s if stream == 'audio' else engine.dec_p
                getattr(engine, '_speech' if stream == 'audio' else '_pose')(True)
                codes = (engine.depB if stream == 'audio' else engine.pdepB)[:, 3] - torch.arange(32) * 2048
                self.assertTrue(bool(((codes >= 0) & (codes < 8)).all()))
                lp0 = torch.arange(8).float().log_softmax(0)[codes[:, 0]]
                torch.testing.assert_close(engine._opd_history[stream][:, 3, 0], lp0)
                for cb in range(1, 32):
                    logits = decoder.head(decoder.seen[cb - 1][:, -1], cb)
                    expected = logits.log_softmax(-1).gather(1, codes[:, cb:cb + 1])[:, 0]
                    torch.testing.assert_close(engine._opd_history[stream][:, 3, cb], expected)
                    if cb > 1:
                        torch.testing.assert_close(decoder.seen[cb - 1][:, 0, 0], codes[:, cb - 1].float())
        finally:
            engine.close()

    def test_text_cdf_masks_padded_vocabulary_and_preserves_forcing(self):
        engine = make_engine(2)
        try:
            engine._opd_executable = 6
            engine._opd_uniforms['text'][:, 3] = torch.tensor([0., .9999], dtype=torch.float64)
            engine._text()
            self.assertEqual(engine.tok_now.tolist(), [0, 5])
            expected = torch.arange(6).float().log_softmax(0)[torch.tensor([0, 5])]
            torch.testing.assert_close(engine._opd_history['text'][:, 3], expected)
            engine._opd_lane_force[1] = 1
            engine._text()
            self.assertEqual(engine.tok_now.tolist(), [0, 1])
            self.assertEqual(engine._opd_history['text'][1, 3].item(), 0.)
        finally:
            engine.close()

    def test_cache_keeps_every_reachable_position_and_rejects_short_capacity(self):
        decoder = Decoder(2)
        decoder.K[0].copy_(torch.arange(decoder.K[0].numel()).reshape_as(decoder.K[0]))
        expected = decoder.K[0][:, :32].clone()
        resize_depth_cache(decoder, 32)
        torch.testing.assert_close(decoder.K[0], expected, rtol=0, atol=0)
        self.assertTrue(decoder.K[0].is_contiguous())
        self.assertEqual(decoder.K[0].shape[1], 32)
        for invalid in (0, 31, 33, True):
            with self.assertRaises(ValueError):
                resize_depth_cache(decoder, invalid)
        with self.assertRaisesRegex(ValueError, 'cannot cover'):
            resize_depth_cache(decoder, 64)


if __name__ == '__main__':
    unittest.main()
