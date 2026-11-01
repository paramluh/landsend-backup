"""CPU construction controls and exact tiny-model trim/gradient comparison."""

from contextlib import ExitStack
import copy
import unittest
from unittest.mock import patch

import torch

from distillation import bimodal_net
from distillation.full_gt_dmd import FullGTStabilizer
from distillation.test_joint_layout import small_net, rows_and_noise
from distillation.test_rollout import attention_cpu
from scripts.compile_probe import variant_context


class CompileProbeTests(unittest.TestCase):
    def test_shared_dynamic_compiles_once_and_restores(self):
        factory, cached = bimodal_net.compiled_whole_block, dict(bimodal_net._WHOLE)
        with patch('torch.compile', side_effect=lambda fn, **kw: (fn, kw)) as compile_mock:
            with variant_context('block_dynamic') as metadata:
                first = bimodal_net.compiled_whole_block()
                second = bimodal_net.compiled_whole_block()
                self.assertIs(first, second)
                self.assertTrue(first[1]['dynamic'])
                self.assertEqual(compile_mock.call_count, 1)
                self.assertTrue(metadata['whole_block_dynamic_shapes'])
        self.assertIs(bimodal_net.compiled_whole_block, factory)
        self.assertEqual(bimodal_net._WHOLE, cached)

    def test_role_isolation_clones_code_objects_without_changing_behavior(self):
        with patch('torch.compile', side_effect=lambda fn, **kw: fn):
            with variant_context('block_role_isolated') as metadata:
                functions = [bimodal_net.compiled_whole_block() for _ in range(3)]
                self.assertEqual(len({id(fn.__code__) for fn in functions}), 3)
                for fn in functions:
                    self.assertEqual(fn.__code__.co_code, bimodal_net.block_forward.__code__.co_code)
                    self.assertIs(fn.__globals__, bimodal_net.block_forward.__globals__)
                self.assertEqual([entry['role'] for entry in metadata['compiled_wrappers_created']],
                                 ['teacher', 'student', 'critic'])

    def test_variant_restore_after_exception(self):
        original = FullGTStabilizer.__init__
        with self.assertRaises(RuntimeError):
            with variant_context('block_untrimmed'):
                self.assertFalse(FullGTStabilizer().trim)
                raise RuntimeError('injected')
        self.assertIs(FullGTStabilizer.__init__, original)
        self.assertTrue(FullGTStabilizer().trim)

    def test_untrimmed_model_outputs_and_parameter_gradients_match(self):
        old_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, old_threads)
        torch.set_num_threads(1)
        with ExitStack() as stack:
            stack.enter_context(patch('distillation.bimodal_net.flash_attn_func', attention_cpu))
            stack.enter_context(patch('models.pose.audio.flash_attn_func', attention_cpu))
            stack.enter_context(patch.object(torch.Tensor, 'bfloat16', lambda tensor, *a, **kw: tensor.float()))
            torch.manual_seed(707)
            first = small_net()
            second = copy.deepcopy(first)
            first.configure(False, 'block')
            second.configure(False, 'block')
            source, noise = rows_and_noise()
            source = dict(source, latents=source['latents'][:, :, :8], pose=source['pose'][:, :, :8],
                          ofi=source['ofi'][:, :8])
            noise = noise[:, :, :8].detach()
            trimmed = FullGTStabilizer(band_steps=(1, 1, 1, 1), trim=True)
            padded = FullGTStabilizer(band_steps=(1, 1, 1, 1), trim=False)
            _, context = trimmed.prepare_context(source, noise=noise)
            expected, _ = trimmed.generate(first, source, noise=noise, context=context)
            actual, _ = padded.generate(second, source, noise=noise, context=context)
            torch.testing.assert_close(expected, actual, atol=2e-6, rtol=2e-5)
            expected.square().mean().backward()
            actual.square().mean().backward()
            for (name, parameter), (_, other) in zip(first.named_parameters(), second.named_parameters()):
                with self.subTest(parameter=name):
                    if parameter.grad is None:
                        self.assertIsNone(other.grad)
                    else:
                        torch.testing.assert_close(parameter.grad, other.grad, atol=2e-7, rtol=2e-4)


if __name__ == '__main__':
    unittest.main()
