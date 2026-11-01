"""Selective reference recomputation preserves real BimodalNet gradients."""
import copy
from contextlib import ExitStack
import unittest
from unittest.mock import patch

import torch

from distillation.full_gt_dmd import FullGTStabilizer, generate_resolver
from distillation.test_joint_layout import clone_rows, compare, rows_and_noise, small_net
from distillation.test_rollout import attention_cpu


class ReferenceCheckpointTests(unittest.TestCase):
    def setUp(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous)
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch('distillation.bimodal_net.flash_attn_func', attention_cpu))
        stack.enter_context(patch('models.pose.audio.flash_attn_func', attention_cpu))
        stack.enter_context(patch.object(torch.Tensor, 'bfloat16', lambda tensor, *args, **kwargs: tensor.float()))

    def test_full_gt_resolver_outputs_all_parameter_and_reference_gradients(self):
        torch.manual_seed(619)
        baseline = small_net()
        baseline.configure(False, 'none')
        selective = copy.deepcopy(baseline)
        selective.reference_checkpoint = 'block'
        rows, noise = rows_and_noise()
        for key in ('latents', 'pose'):
            rows[key] = rows[key][:, :, :8].detach().requires_grad_(True)
        rows['ofi'] = rows['ofi'][:, :8]
        copied = clone_rows(rows)
        rollout = FullGTStabilizer(history_dropout=(0., 0., 0., 0.))
        fixed_noise = noise[:, :, :8].detach()
        _, context = rollout.prepare_context(rows, fixed_noise)
        _, copied_context = rollout.prepare_context(copied, fixed_noise)
        expected, _ = generate_resolver(baseline, rows, context, fixed_noise[:, :, -1:])
        policies = []
        run = selective._run_block
        def record(*args):
            policies.append((selective.checkpoint, args[3]))  # heads=0 during reference prefill
            return run(*args)
        with patch.object(selective, '_run_block', record):
            actual, _ = generate_resolver(selective, copied, copied_context, fixed_noise[:, :, -1:])
            compare(self, expected, actual, 2e-6, 2e-5)
            self.assertEqual(selective.checkpoint, 'none')
            self.assertEqual(policies, [('block', 0), ('block', 0), ('none', 1), ('none', 1)])
            expected.square().mean().backward()
            actual.square().mean().backward()
        for (name, original), (other_name, optimized) in zip(baseline.named_parameters(), selective.named_parameters()):
            self.assertEqual(name, other_name)
            with self.subTest(parameter=name):
                compare(self, original.grad, optimized.grad, 2e-7, 2e-4)
        compare(self, rows['refs'].grad, copied['refs'].grad, 2e-7, 2e-4)
        self.assertGreater(float(copied['refs'].grad.abs().sum()), 0.)
        self.assertGreater(float(selective.ref_patch_embedding.weight.grad.abs().sum()), 0.)
        self.assertIsNone(selective._reference_cache)
        self.assertIsNone(selective._references)
        self.assertEqual(selective.checkpoint, 'none')

    def test_prefill_failure_restores_reference_state_and_target_policy(self):
        net = small_net()
        net.configure(False, 'none')
        net.reference_checkpoint = 'block'
        rows, _ = rows_and_noise()
        with patch.object(net, '_prefill_references', side_effect=RuntimeError('prefill failed')):
            with self.assertRaisesRegex(RuntimeError, 'prefill failed'):
                with net.references(rows['refs'], rows['ref_keep']):
                    self.fail('prefill should fail before target execution')
        self.assertEqual(net.checkpoint, 'none')
        self.assertIsNone(net._references)
        self.assertIsNone(net._reference_cache)


if __name__ == '__main__':
    unittest.main()
