"""Pool-only causal trimming changes execution shapes, not targets/context."""
import ast
import copy
from contextlib import ExitStack, nullcontext
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from distillation.full_gt_dmd import FullGTStabilizer
from distillation.test_joint_layout import rows_and_noise, small_net
from distillation.test_rollout import attention_cpu


class FullGTPoolTrimTests(unittest.TestCase):
    def test_real_bimodal_pool_trim_output_context_and_rng_parity(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous)
        with ExitStack() as stack:
            stack.enter_context(patch('distillation.bimodal_net.flash_attn_func', attention_cpu))
            stack.enter_context(patch('models.pose.audio.flash_attn_func', attention_cpu))
            stack.enter_context(patch.object(torch.Tensor, 'bfloat16', lambda tensor, *args, **kwargs: tensor.float()))
            torch.manual_seed(731)
            net = small_net()
            rows, noise = rows_and_noise()
            for key in ('latents', 'pose'):
                rows[key] = rows[key][:, :, :8].detach()
            rows['ofi'] = rows['ofi'][:, :8]
            rows['history_keep'] = torch.tensor([[True, False, True, True, False, True],
                                                [False, True, True, False, True, True]])
            padded = FullGTStabilizer(trim=False)
            trimmed = FullGTStabilizer(trim=True)
            noise, context = padded.prepare_context(rows, noise[:, :, :8].detach())
            retained = {key: value.clone() for key, value in context.items()}
            state = torch.get_rng_state().clone()
            with torch.no_grad():
                expected, original_context = padded.generate(net, rows, noise, context)
                actual, pool_context = trimmed.generate(net, rows, noise, context)
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
            self.assertIs(original_context, context)
            self.assertIs(pool_context, context)
            self.assertTrue(torch.equal(state, torch.get_rng_state()))
            for key, value in retained.items():
                torch.testing.assert_close(context[key], value)
            self.assertIsNone(net._reference_cache)
            self.assertIsNone(net._references)

    def test_trainer_selects_pool_layout_only_and_prepares_context_once(self):
        source = ast.parse((Path(__file__).resolve().parents[1] / 'learn_dmd_joint.py').read_text())
        main = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        fn = next(node for node in main.body if isinstance(node, ast.FunctionDef) and node.name == 'generate')
        calls = []
        noise = torch.ones(2, 1, 8, 1, 1)
        context = {'keep': torch.ones(2, 6, dtype=torch.bool)}
        def prepare(rows):
            calls.append('context')
            return noise, context
        def target(label, net, rows, **kwargs):
            self.assertIs(kwargs['noise'], noise)
            self.assertIs(kwargs['context'], context)
            calls.append(label)
            return label, context
        ns = dict(full_gt=True, timed=lambda name: nullcontext(),
                  full_gt_rollout=SimpleNamespace(prepare_context=prepare,
                      generate=lambda *args, **kwargs: target('student', *args, **kwargs)),
                  full_gt_pool_rollout=SimpleNamespace(
                      generate=lambda *args, **kwargs: target('pool', *args, **kwargs)),
                  sums=SimpleNamespace(add=lambda *args: None), nets={'student': object()},
                  training_metrics=None)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), '<generate>', 'exec'), ns)
        rows = {'weight': torch.ones(2)}
        self.assertEqual(ns['generate'](rows, 'stabilizer', critic_pool=True)[0], 'pool')
        self.assertEqual(ns['generate'](rows, 'stabilizer')[0], 'student')
        self.assertEqual(calls, ['context', 'pool', 'context', 'student'])


if __name__ == '__main__':
    unittest.main()
