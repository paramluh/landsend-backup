"""CPU dependency checks for split sampling, committed replay, and bounded rewind.

The cached stand-in implements the native append-at-T/read-through-T contract.
It exercises the actual natural actor orchestration, without claiming CUDA
kernel parity (the integration benchmark must establish that separately).
"""
import types
import unittest

import numpy as np
import torch

from finetune.pose_opd_credit.rollout import make_fastppm_class
from finetune.pose_opd_credit.tests.test_sampling import Base


class Ring:
    def __init__(self, rows):
        self.ring = torch.full((rows, 16), -1, dtype=torch.long)
        self.pidx = self.ring.clone()
        self.ptr = torch.zeros(rows, dtype=torch.long)

    def push(self, value):
        rows = torch.arange(len(value))
        self.ring[rows, self.ptr % 16] = value
        self.pidx[rows, self.ptr % 16] = self.ptr
        self.ptr += 1


class CachedAudioBase(Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pose = False
        self.dec_p = None
        del self.pdepB
        self.R = self.NI
        self.P = [5] * self.NI
        self.Pv = {self.d0: torch.full((self.NI,), 5, dtype=torch.long)}
        self.segs = [(self.d0, [0])]
        self.sdevs = [self.d0]
        self.graphs = {}
        self.ring = Ring(self.NI)
        self.bb_cache = torch.zeros(self.NI, self.lmax, self.H)
        self.tower_cache = torch.zeros_like(self.bb_cache)
        self.head_calls = self.light_calls = self.tower_calls = self.backbone_calls = 0

    def set_f(self, frame):
        self.Tv[self.d0].copy_(self.Pv[self.d0] + frame)

    def text_logits(self, hidden):
        self.head_calls += 1
        return super().text_logits(hidden)

    def _speech(self, full):
        assert not full  # Natural inverse-CDF implementation owns light draws.
        self.tower_calls += 1
        positions, rows = self.Tv[self.dl], self.arI[self.dl]
        self.hv_s = self.hn_prev + self.tok_now[:, None].float()
        self.tower_cache[rows, positions] = self.hv_s
        self.z0 = torch.arange(8).float()[None] * (1 + self.hv_s[:, :1] / 20)
        self.pk[self.dl][:, 0].copy_(self.tok_now)
        self.pk[self.dl][:, 1:].copy_(self.depB[rows, positions])

    def _seg(self, index):
        self.backbone_calls += 1
        positions, rows = self.Tv[self.d0], self.arI[self.d0]
        own = self.pk[self.d0][:, [0, 1]].float() / 100
        self.bb_cache[rows, positions] = self.m.inp(own)
        # Stale suffixes may contain arbitrary values; never read beyond T.
        for row, position in enumerate(positions.tolist()):
            self.hn_prev[row].copy_(self.bb_cache[row, :position + 1].sum(0))


def engine(rows=2, rewind=8):
    native = types.SimpleNamespace(FastPPM=CachedAudioBase, PAD=1, NW=2)
    actor = make_fastppm_class(native, require_pose=False)(
        n_inst=rows, executable_vocab_size=8, sampler_backend='inverse_cdf', depth_cache_positions=32)
    actor.enable_rewind(rewind)
    actor.start([{'ids': np.zeros((2, 40), np.int64)}] * rows, list(range(rows)), f0=2)
    return actor


def committed(actor, frame):
    return {'lane': torch.full((actor.NI,), frame % 8, dtype=torch.long),
            'codes': (torch.arange(32) * 2048 + frame % 8).expand(actor.NI, -1).clone()}


class PartnerActorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_audio_only_has_no_pose_allocations_and_keeps_natural_audio_sampling(self):
        actor = engine()
        try:
            self.assertEqual(set(actor._opd_uniforms), {'text', 'audio'})
            self.assertEqual(set(actor._opd_history), {'text', 'audio'})
            before = actor.hn_prev.clone()
            actor.sample_outputs(2, graphs=False)
            torch.testing.assert_close(actor.hn_prev, before, rtol=0, atol=0)
            self.assertEqual(actor.backbone_calls, 0)
            self.assertGreater(len(actor.dec_s.seen), 0)
            actor.set_audio_frame(2, torch.full((actor.NI, actor.H), 3.))
            actor.advance_backbone(2, graphs=False)
            self.assertTrue(bool((actor.hn_prev >= 3.).all()))
            self.assertEqual(actor.backbone_calls, 1)
        finally:
            actor.close()

    def test_rewind_corrected_audio_matches_fresh_replay_without_resampling_committed_outputs(self):
        actor, fresh = engine(), engine()
        try:
            # Same committed utterance, but the speculative run saw silence at f4.
            fresh.set_audio_frame(4, torch.full((fresh.NI, fresh.H), .25))
            for frame in range(2, 9):
                for current in (actor, fresh):
                    current.sample_outputs(frame, force=committed(current, frame), sample_text=False, graphs=False)
                    current.advance_backbone(frame, graphs=False)
            prefix = actor.bb_cache[:, :9].clone()  # slot P+4=9 is first changed.
            actor.rewind(4)
            actor.set_audio_frame(4, torch.full((actor.NI, actor.H), .25))
            count = actor.tower_calls
            actor.advance_backbone(4, force=committed(actor, 4), graphs=False)
            self.assertEqual(actor.tower_calls, count)  # first tower remains valid.
            for frame in range(5, 9):
                actor.sample_outputs(frame, force=committed(actor, frame), sample_text=False, graphs=False)
                actor.advance_backbone(frame, graphs=False)
            torch.testing.assert_close(actor.hn_prev, fresh.hn_prev, rtol=0, atol=0)
            torch.testing.assert_close(actor.bb_cache, fresh.bb_cache, rtol=0, atol=0)
            torch.testing.assert_close(actor.tower_cache, fresh.tower_cache, rtol=0, atol=0)
            torch.testing.assert_close(actor.bb_cache[:, :9], prefix, rtol=0, atol=0)
            torch.testing.assert_close(actor.ring.ring, fresh.ring.ring, rtol=0, atol=0)
            torch.testing.assert_close(actor.ring.ptr, fresh.ring.ptr, rtol=0, atol=0)
            self.assertEqual(actor.head_calls, 0)
            self.assertEqual(actor.dec_s.seen, [])  # no light forwards on forced replay.
        finally:
            actor.close()
            fresh.close()

    def test_sequential_contract_rejects_stale_cache_skips_and_expires_rewind_slots(self):
        actor = engine(rewind=3)
        try:
            with self.assertRaisesRegex(ValueError, 'Expected unsampled'):
                actor.sample_outputs(3)
            actor.sample_outputs(2, graphs=False)
            with self.assertRaisesRegex(ValueError, 'pending backbone'):
                actor.sample_outputs(3)
            actor.rewind(2)  # pending speculative sample can be discarded.
            for frame in range(2, 6):
                actor.step(frame, graphs=False)
            with self.assertRaisesRegex(ValueError, 'No retained'):
                actor.rewind(2)
            actor.rewind(4)
            with self.assertRaisesRegex(ValueError, 'No retained'):
                actor.rewind(5)  # previously computed suffix is now invalid.
            with self.assertRaisesRegex(ValueError, 'no matching'):
                actor.advance_backbone(4)
        finally:
            actor.close()

    def test_forced_replay_uses_tower_graph_and_never_full_sampling_graph(self):
        actor = engine()
        calls = []

        class Graph:
            def __init__(self, name, fn):
                self.name, self.fn = name, fn
            def replay(self):
                calls.append(self.name)
                return self.fn()

        actor.graphs = {name: Graph(name, fn) for name, fn in {
            'text': actor._text, 'speech': lambda: actor._speech(False),
            'speech_full': lambda: actor._speech(True), 'seg0': lambda: actor._seg(0)}.items()}
        try:
            actor.sample_outputs(2, force=committed(actor, 2), sample_text=False)
            actor.advance_backbone(2)
            self.assertEqual(calls, ['speech', 'seg0'])
            self.assertEqual(actor.head_calls, 0)
            self.assertEqual(actor.dec_s.seen, [])
        finally:
            actor.close()

    def test_mixed_prefill_overrides_only_forced_rows_and_clears_their_credit(self):
        actor = engine()
        try:
            actor.sample_outputs(2, graphs=False)
            positions, rows = actor.Tv[actor.dl], actor.arI[actor.dl]
            sampled = actor.depB[rows, positions].clone()
            likelihood = actor._opd_history['audio'][rows, positions].clone()
            tower = actor.tower_cache.clone()
            forced = (torch.arange(32) * 2048).expand(actor.NI, -1).clone()
            actor.force_media_rows(2, mask=torch.tensor([True, False]), codes=forced)
            torch.testing.assert_close(actor.depB[0, positions[0]], forced[0], rtol=0, atol=0)
            torch.testing.assert_close(actor.depB[1, positions[1]], sampled[1], rtol=0, atol=0)
            torch.testing.assert_close(actor.pk[actor.dl][0, 1:], forced[0], rtol=0, atol=0)
            self.assertTrue(bool((actor._opd_history['audio'][0, positions[0]] == 0).all()))
            torch.testing.assert_close(actor._opd_history['audio'][1, positions[1]], likelihood[1], rtol=0, atol=0)
            torch.testing.assert_close(actor.tower_cache, tower, rtol=0, atol=0)
            actor.advance_backbone(2, graphs=False)
            with self.assertRaisesRegex(ValueError, 'awaiting'):
                actor.force_media_rows(2, mask=[True, False], codes=forced)
        finally:
            actor.close()


if __name__ == '__main__':
    unittest.main()
