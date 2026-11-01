"""Bounded categorical diagnostics and per-actor attention precision routing."""
import types
import unittest

import numpy as np
import torch

from finetune.pose_opd_credit.rollout import make_fastppm_class
from finetune.pose_opd_credit.tests.test_sampling import Base, make_engine


class PolicyProbeTests(unittest.TestCase):
    def test_probe_snapshots_and_tail_schedule_are_per_row_and_bounded(self):
        engine = make_engine(2, policy_probe_frames=(3,))
        try:
            # The CPU test decoder has eight categories; production has2048.
            for stream, decoder in (('audio', engine.dec_s), ('pose', engine.dec_p)):
                engine._opd_depth_probe[stream] = torch.empty(2, 2, 8)
                decoder._opd_probe_logits = {14: engine._opd_depth_probe[stream][:, 0],
                                             31: engine._opd_depth_probe[stream][:, 1]}
            engine.start([{'P': 2, 'ids': np.zeros((2, 10), np.int64)},
                          {'P': 2, 'ids': np.zeros((2, 12), np.int64)}], [81, 17], f0=0)
            self.assertEqual(engine._opd_active_probe_frames, {3, 5, 7, 9})
            engine.Tv[engine.dl].fill_(5)
            engine._text(); engine._speech(True); engine._pose(True)
            engine._record_policy_probe(3)
            for row in range(2):
                records = engine.policy_probes(row)
                self.assertEqual(len(records), 7)
                self.assertTrue(all(record['position'] == 5 for record in records))
                for stream, decoder in (('audio', engine.dec_s), ('pose', engine.dec_p)):
                    for cb in (14, 31):
                        record = next(r for r in records if r['stream'] == stream and r['codebook'] == cb)
                        expected = decoder.head(decoder.seen[cb - 1][:, -1], cb)[row].numpy()
                        np.testing.assert_array_equal(record['logits'], expected)
            saved = engine.policy_probes(0)[0]['logits'].copy()
            engine.zt.fill_(123)
            np.testing.assert_array_equal(engine.policy_probes(0)[0]['logits'], saved)
            # f5 is only row0's text end. Row1 must not acquire extra probes.
            engine._record_policy_probe(5)
            self.assertEqual(len(engine.policy_probes(0)), 14)
            self.assertEqual(len(engine.policy_probes(1)), 7)
            # f9 is row1's media end, so it records six media events, no text.
            engine._record_policy_probe(9)
            self.assertEqual(len(engine.policy_probes(1)), 13)
        finally:
            engine.close()

    def test_fp32_override_routes_only_its_decoder_to_unrounded_attention(self):
        calls = []
        def norm(res, add, weight, eps):
            res = res if add is None else res + add
            return res, res * weight
        def attn(qkv, *args, round_p):
            calls.append(round_p)
            return qkv
        lx = types.SimpleNamespace(add_rms_f32=norm, attn_exact=attn,
                                   final_norm_f32=lambda r, a, w, e: norm(r, a, w, e)[1])
        lk = types.SimpleNamespace(silu_mul_gu=lambda value: value.chunk(2, -1)[0] * value.chunk(2, -1)[1])
        native = types.SimpleNamespace(FastPPM=Base, LX=lx, LK=lk)
        cls = make_fastppm_class(native)
        with self.assertRaisesRegex(ValueError, 'attention precision'):
            cls(attention_precision='unknown')
        engine = cls(executable_vocab_size=8, attention_precision='fp32')
        other = cls(executable_vocab_size=8)
        try:
            decoder = engine.dec_s
            decoder.S = 1
            decoder.cos, decoder.sin = torch.ones(64, 2), torch.zeros(64, 2)
            decoder.nh, decoder.kv, decoder.D = 1, 1, 2
            decoder.nw, decoder.ne = torch.ones(2), 1e-6
            decoder.lin1 = lambda x, w: x @ w.T
            decoder.layers = [dict(ln1=torch.ones(2), e1=1e-6, wqkv=torch.eye(2), qn=None, kn=None,
                                   eqk=1e-6, sc=1., wo=torch.eye(2), ln2=torch.ones(2), e2=1e-6,
                                   wgu=torch.cat((torch.eye(2), torch.eye(2))), wd=torch.eye(2))]
            output = decoder.step(torch.ones(1, 2, 2), 0)
            self.assertEqual(tuple(output.shape), (1, 2))
            self.assertEqual(calls, [0])
            self.assertIn('step', decoder.__dict__)
            self.assertNotIn('step', other.dec_s.__dict__)
            self.assertFalse(hasattr(lx, 'ROUND_P'))  # no global precision mutation
        finally:
            engine.close(); other.close()
        self.assertNotIn('step', decoder.__dict__)


if __name__ == '__main__':
    unittest.main()
