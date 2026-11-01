"""Rollout observations reuse existing velocities without changing training."""
import ast
import copy
from contextlib import ExitStack, nullcontext
import math
from pathlib import Path
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import numpy as np
import torch

from distillation.full_gt_dmd import FullGTStabilizer, generate_resolver
from distillation.test_joint_layout import clone_rows, rows_and_noise, small_net
from distillation.test_rollout import attention_cpu
from distillation.training_metrics import TrainingMetrics


class StudentRolloutMetricTests(unittest.TestCase):
    def setUp(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous)
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch('distillation.bimodal_net.flash_attn_func', attention_cpu))
        stack.enter_context(patch('models.pose.audio.flash_attn_func', attention_cpu))
        stack.enter_context(patch.object(torch.Tensor, 'bfloat16', lambda tensor, *a, **kw: tensor.float()))

    def test_observation_preserves_outputs_gradients_rng_and_network_call_counts(self):
        torch.manual_seed(743)
        original = small_net()
        rows, _ = rows_and_noise()
        for name in ('latents', 'pose'):
            rows[name] = rows[name][:, :, :8].detach().requires_grad_()
        rows['ofi'] = rows['ofi'][:, :8]
        rows['weight'] = torch.tensor([1., .25])

        def run(kind, checkpoint, trim, observe):
            random.seed(91)
            np.random.seed(92)
            torch.manual_seed(93)
            net = copy.deepcopy(original)
            net.configure(False, checkpoint)
            batch = clone_rows(rows)
            rollout = FullGTStabilizer(trim=trim)
            metrics = TrainingMetrics('cpu', channels=2)
            observed_refs, observed_squares = [], []

            def observer(velocity):
                self.assertFalse(velocity.requires_grad)
                self.assertIsNone(velocity.grad_fn)
                self.assertEqual(velocity.shape[2], 1)
                observed_refs.append(weakref.ref(velocity))
                observed_squares.append(float(
                    (velocity.square().sum((1, 2, 3, 4), dtype=torch.float64) * batch['weight']).sum()))
                metrics.add_student_rollout(kind, velocity, batch['weight'])

            with patch.object(net, 'forward_pass', wraps=net.forward_pass) as forward:
                if kind == 'stabilizer':
                    output, context = rollout.generate(net, batch, observe_velocity=observer if observe else None)
                else:
                    _, context = rollout.prepare_context(batch)
                    output, context = generate_resolver(net, batch, context,
                                                        observe_velocity=observer if observe else None)
                loss = output.square().mean()
                loss.backward()
                calls = forward.call_count
            report, _ = metrics.report({'stabilizer': .5, 'resolver': .5}, reduce=False)
            if observe:
                expected_observations = 8 if kind == 'stabilizer' else 1
                self.assertEqual(len(observed_refs), expected_observations)
                self.assertTrue(all(ref() is None for ref in observed_refs))
                count = expected_observations * batch['weight'].sum().item() * output[0].numel()
                self.assertAlmostEqual(report[f'velocity_probes/{kind}/student_rollout_rms'],
                                       math.sqrt(sum(observed_squares) / count))
                self.assertFalse(metrics.values.requires_grad)
                self.assertIsNone(metrics.values.grad_fn)
            else:
                self.assertEqual(report, {})
            return (output.detach(), context,
                    {name: value.grad for name, value in net.named_parameters()},
                    {name: value.grad for name, value in batch.items()},
                    calls, torch.rand(4), random.random(), np.random.rand(4))

        for kind in ('stabilizer', 'resolver'):
            for checkpoint in ('none', 'block'):
                for trim in ((False, True) if kind == 'stabilizer' else (False,)):
                    with self.subTest(kind=kind, checkpoint=checkpoint, trim=trim):
                        expected = run(kind, checkpoint, trim, False)
                        observed = run(kind, checkpoint, trim, True)
                        torch.testing.assert_close(observed[:6], expected[:6], rtol=0, atol=0)
                        self.assertEqual(observed[6], expected[6])
                        np.testing.assert_array_equal(observed[7], expected[7])

    def test_trainer_observes_student_rollouts_only(self):
        source = ast.parse((Path(__file__).resolve().parents[1] / 'learn_dmd_joint.py').read_text())
        main = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        fn = next(node for node in main.body if isinstance(node, ast.FunctionDef) and node.name == 'generate')
        metrics = TrainingMetrics('cpu', channels=1)
        calls = []
        noise = torch.ones(2, 1, 8, 1, 1)
        context = {'keep': torch.ones(2, 6, dtype=torch.bool)}
        velocity = torch.tensor([2., 6.]).view(2, 1, 1, 1, 1).requires_grad_()

        def target(net, rows, context=None, noise=None, observe_velocity=None):
            calls.append(observe_velocity is not None)
            if observe_velocity is not None:
                observe_velocity(velocity.detach())
            return velocity, context

        namespace = dict(full_gt=True, timed=lambda name: nullcontext(),
                         full_gt_rollout=SimpleNamespace(prepare_context=lambda rows: (noise, context),
                                                        generate=target),
                         full_gt_pool_rollout=SimpleNamespace(generate=target),
                         generate_full_gt_resolver=target, training_metrics=metrics,
                         sums=SimpleNamespace(add=lambda *args: None), nets={'student': object()})
        exec(compile(ast.Module(body=[fn], type_ignores=[]), '<generate>', 'exec'), namespace)
        rows = {'weight': torch.tensor([1., .5])}
        for kind in ('stabilizer', 'resolver'):
            namespace['generate'](rows, kind, critic_pool=True)
            self.assertNotIn(f'velocity_probes/{kind}/student_rollout_rms',
                             metrics.report({'stabilizer': .5, 'resolver': .5}, reduce=False)[0])
            namespace['generate'](rows, kind)
        self.assertEqual(calls, [False, True, False, True])
        online, local = metrics.report({'stabilizer': .5, 'resolver': .5}, reduce=False)
        for kind in ('stabilizer', 'resolver'):
            self.assertAlmostEqual(online[f'velocity_probes/{kind}/student_rollout_rms'],
                                   math.sqrt((4. + .5 * 36.) / 1.5))
            self.assertEqual(local[f'velocity_probes/{kind}/student_rollout_square']['count'], 1.5)


if __name__ == '__main__':
    unittest.main()
