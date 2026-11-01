"""The optional M16 backend changes per-instance launches, never shared code."""
from contextlib import nullcontext
import types
import unittest
from unittest.mock import patch

import torch

from finetune.pose_opd_credit.rollout import make_fastppm_class, tuned_m16_linear
from finetune.pose_opd_credit.tests.test_sampling import Base


class ActorLinearTests(unittest.TestCase):
    def test_known_m16_shapes_keep_native_kernel_and_explicit_configs(self):
        calls = []
        class Kernel:
            def __getitem__(self, grid):
                def launch(x, weight, result, rows, outputs, inputs, **options):
                    calls.append((grid, rows, outputs, inputs, options))
                    result.fill_(7)
                return launch
        kernel = Kernel()
        native = types.SimpleNamespace(LK=types.SimpleNamespace(_gemv_k=types.SimpleNamespace(fn=kernel)))
        def fallback(*args):
            self.fail('Known M16 matrices must use the explicit native kernel')
        for outputs, inputs, bk, warps, stages in (
            (4096,1024,128,2,4), (2048,1024,128,2,4), (6144,1024,128,2,4),
            (1024,2048,256,4,3), (1024,3072,256,4,3)):
            x = torch.zeros(1,16,inputs,dtype=torch.bfloat16)
            weight = torch.zeros(outputs,inputs,dtype=torch.bfloat16)
            with patch('torch.cuda.device', return_value=nullcontext()):
                result = tuned_m16_linear(native, fallback, x, weight)
            self.assertEqual(result.shape, (1,16,outputs))
            self.assertTrue(bool((result == 7).all()))
            self.assertEqual(calls[-1], ((outputs//16,),16,outputs,inputs,
                dict(BM=16,BN=16,BK=bk,num_warps=warps,num_stages=stages)))
        self.assertIs(native.LK._gemv_k.fn, kernel)

    def test_other_geometry_and_dtype_keep_native_dispatch(self):
        calls = []
        fallback = lambda x,w: calls.append((x.shape,w.shape)) or 'native'
        native = types.SimpleNamespace()  # touching its kernel would fail
        for rows, outputs, inputs, dtype in ((8,1024,3072,torch.bfloat16),
                                              (32,1024,3072,torch.bfloat16),
                                              (16,16,16,torch.bfloat16),
                                              (16,1024,3072,torch.float32)):
            self.assertEqual(tuned_m16_linear(native, fallback,
                torch.zeros(rows,inputs,dtype=dtype), torch.zeros(outputs,inputs,dtype=dtype)), 'native')
        self.assertEqual(len(calls),4)

    def test_constructor_validates_and_override_is_instance_local(self):
        native = types.SimpleNamespace(FastPPM=Base)
        cls = make_fastppm_class(native)
        with self.assertRaisesRegex(ValueError, 'linear backend'):
            cls(linear_backend='bad')
        # The CPU fixture's minimal decoder does not otherwise need lin1.
        from finetune.pose_opd_credit.tests import test_sampling
        with patch.object(test_sampling.Decoder, 'lin1', staticmethod(lambda x,w:'native'), create=True):
            engine = cls(executable_vocab_size=8, linear_backend='tuned_m16')
            other = cls(executable_vocab_size=8)
            decoder = engine.dec_s
            try:
                self.assertIn('lin1', decoder.__dict__)
                self.assertNotIn('lin1', other.dec_s.__dict__)
                self.assertEqual(decoder.lin1(torch.zeros(1,2),torch.zeros(2,2)), 'native')
            finally:
                engine.close();other.close()
            self.assertNotIn('lin1', decoder.__dict__)


if __name__ == '__main__':
    unittest.main()
