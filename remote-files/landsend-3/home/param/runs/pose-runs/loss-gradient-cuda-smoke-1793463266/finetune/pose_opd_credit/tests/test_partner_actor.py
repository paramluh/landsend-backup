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

    def reset(self):
        self.ring.fill_(-1)
        self.pidx.fill_(-1)
        self.ptr.zero_()


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

    def start(self, instances, seeds, params, f0):
        """Model native start's prefix overwrite, leaving stale KV suffixes allocated.

        Actual FastPPM.start replays each prefix, resets the text ring and RNG,
        and clears token arrays. It intentionally does not clear the growing KV
        allocation, so a repeated-batch test must not hide bugs by zeroing it.
        """
        super().start(instances, seeds, params, f0)
        self.ring.reset()
        self.laneB.fill_(1)
        self.depB.fill_(-1)
        self.P = [int(instance.get('P', 5)) for instance in instances]
        self.Pv[self.d0].copy_(torch.tensor(self.P))
        for row, (instance, prefix) in enumerate(zip(instances, self.P)):
            stop = prefix + f0
            ids = torch.as_tensor(instance['ids'][1, :stop], dtype=torch.long)
            depth = torch.as_tensor(instance.get('depth', np.zeros((2, instance['ids'].shape[1], 32), np.int64))
                                    [1, :stop], dtype=torch.long)
            self.laneB[row, :stop].copy_(ids)
            self.depB[row, :stop].copy_(depth)
            own = torch.stack((ids, depth[:, 0]), -1).float() / 100
            current = own + self._opd_audio[row, :stop].float()
            self.bb_cache[row, :stop].copy_(current)
            cumulative = current.cumsum(0)
            previous = torch.cat((torch.zeros_like(cumulative[:1]), cumulative[:-1]), 0)
            self.tower_cache[row, :stop].copy_(previous + ids[:, None])
            self.hn_prev[row].copy_(cumulative[-1])
        self.set_f(f0)

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

    def test_repeated_batch_after_pending_teacher_draw_matches_fresh_actor(self):
        # Native teacher rollouts finish with an emitted frame whose backbone
        # remains pending. Start must establish the new session before a rewind
        # ring can be configured, and must invalidate every old boundary tag.
        torch.manual_seed(811)
        actor = engine()
        torch.manual_seed(811)
        fresh = engine()
        try:
            for frame in range(2, 8):
                actor.step(frame, graphs=False)
            actor.sample_outputs(8, graphs=False)
            self.assertEqual(actor._opd_pending_frame, 8)
            with self.assertRaisesRegex(ValueError, 'completed backbone boundary'):
                actor.enable_rewind(8)  # the old, incorrect lifecycle ordering
            rewind_storage = {name: value.data_ptr() for name, value in actor._opd_rewind_states.items()}
            uniform_storage = {name: value.data_ptr() for name, value in actor._opd_uniforms.items()}
            old_uniforms = actor._opd_uniforms['audio'].clone()
            actor._opd_history['text'].fill_(123.)
            actor._opd_history['audio'].fill_(123.)
            # Stale suffix poison must not leak across changed prefix lengths.
            actor.bb_cache[:, 20:].fill_(float('nan'))
            actor.tower_cache[:, 20:].fill_(float('nan'))
            instances = []
            for row, prefix in enumerate((7, 11)):
                ids = np.full((2, 40), row + 2, np.int64)
                depth = np.full((2, 40, 32), row + 3, np.int64)
                instances.append({'P': prefix, 'ids': ids, 'depth': depth})
            for current in (actor, fresh):
                # The production frontend fills a new audio tensor before start.
                for row, prefix in enumerate((7, 11)):
                    current.set_audio_features(torch.full((20, current.H), .125 * (row + 1)),
                                               prefix=prefix, instance=row)
                current.start(instances, [111, 222], f0=3)
                if current._opd_rewind_window < 8:
                    current.enable_rewind(8)
            self.assertIsNone(actor._opd_pending_frame)
            self.assertEqual(actor._opd_next_frame, 3)
            self.assertEqual([value for value in actor._opd_rewind_frames if value is not None], [3])
            with self.assertRaisesRegex(ValueError, 'No retained'):
                actor.rewind(2)
            self.assertTrue(bool(torch.isnan(actor._opd_history['text']).all()))
            self.assertTrue(bool(torch.isnan(actor._opd_history['audio']).all()))
            self.assertEqual(rewind_storage, {name: value.data_ptr() for name, value in actor._opd_rewind_states.items()})
            self.assertEqual(uniform_storage, {name: value.data_ptr() for name, value in actor._opd_uniforms.items()})
            self.assertFalse(torch.equal(old_uniforms, actor._opd_uniforms['audio']))
            for stream in actor._opd_uniforms:
                torch.testing.assert_close(actor._opd_uniforms[stream], fresh._opd_uniforms[stream], rtol=0, atol=0)
            for frame in range(3, 11):
                for current in (actor, fresh):
                    current.step(frame, graphs=False)
                torch.testing.assert_close(actor.hn_prev, fresh.hn_prev, rtol=0, atol=0)
                positions = actor.Tv[actor.dl]
                rows = actor.arI[actor.dl]
                torch.testing.assert_close(actor.laneB[rows, positions], fresh.laneB[rows, positions], rtol=0, atol=0)
                torch.testing.assert_close(actor.depB[rows, positions], fresh.depB[rows, positions], rtol=0, atol=0)
                torch.testing.assert_close(actor.tower_cache[rows, positions], fresh.tower_cache[rows, positions], rtol=0, atol=0)
                for stream in ('text', 'audio'):
                    torch.testing.assert_close(actor._opd_history[stream][rows, positions],
                                               fresh._opd_history[stream][rows, positions], rtol=0, atol=0)
            torch.testing.assert_close(actor.ring.ring, fresh.ring.ring, rtol=0, atol=0)
            actor.rewind(6)
            fresh.rewind(6)
            torch.testing.assert_close(actor.hn_prev, fresh.hn_prev, rtol=0, atol=0)
            torch.testing.assert_close(actor.ring.ring, fresh.ring.ring, rtol=0, atol=0)
        finally:
            actor.close()
            fresh.close()

    def test_same_seed_restart_reproduces_draws_without_reallocating_cached_state(self):
        actor = engine()
        instances = [{'ids': np.zeros((2, 40), np.int64)}] * actor.NI
        try:
            observed = []
            for _ in range(3):
                actor.start(instances, [81, 17], f0=2)
                if actor._opd_rewind_window < 8:
                    actor.enable_rewind(8)
                draws = []
                for frame in range(2, 7):
                    actor.sample_outputs(frame, graphs=False)
                    positions, rows = actor.Tv[actor.dl], actor.arI[actor.dl]
                    draws.append((actor.laneB[rows, positions].clone(), actor.depB[rows, positions].clone(),
                                  actor._opd_history['audio'][rows, positions].clone()))
                    if frame < 6:
                        actor.advance_backbone(frame, graphs=False)
                self.assertEqual(actor._opd_pending_frame, 6)
                observed.append(draws)
            for repeat in observed[1:]:
                for original, repeated in zip(observed[0], repeat):
                    for want, got in zip(original, repeated):
                        torch.testing.assert_close(got, want, rtol=0, atol=0)
        finally:
            actor.close()


if __name__ == '__main__':
    unittest.main()
