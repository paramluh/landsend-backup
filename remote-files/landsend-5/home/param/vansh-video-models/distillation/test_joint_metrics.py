"""CPU evidence that richer diagnostics preserve training and rollout algebra."""
import ast
import copy
from contextlib import ExitStack, contextmanager
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from distillation.joint_dmd import generate_resolver, generate_stabilizer
from distillation.joint_metrics import (add_dmd_diagnostics, add_resolver_history_dropout,
    add_timing_aliases, finalize_diagnostics, project_training_metrics,
    TRAINING_METRIC_MAP, TRAINING_METRIC_ALIASES, TRAINING_METADATA_FIELDS)
from distillation.test_joint_layout import clone_rows, rows_and_noise, small_net
from distillation.test_rollout import attention_cpu


class CPUSums:
    def __init__(self):
        self.values = {}

    def add(self, name, total, count):
        assert not total.requires_grad and not count.requires_grad
        total, count = total.double(), count.double()
        previous = self.values.get(name, (0, 0))
        self.values[name] = (previous[0] + total, previous[1] + count)

    def report(self):
        return {name: (total / count).item() for name, (total, count) in self.values.items() if count > 0}


class JointMetricsTests(unittest.TestCase):
    def test_prior_formulas_are_weighted_detached_and_padding_safe(self):
        torch.manual_seed(91)
        tensors = [torch.randn(3, 2, 4, 1, 2, requires_grad=True) for _ in range(4)]
        weights = torch.tensor([1., .25, 0.])
        # Nonfinite padding must not pollute valid-row diagnostic totals.
        with torch.no_grad():
            for tensor in tensors:
                tensor[2] = float('nan')
        y, vt, vf, target = tensors
        sums = CPUSums()
        rng = torch.get_rng_state().clone()
        add_dmd_diagnostics(sums, 'stabilizer', *tensors, weights)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertTrue(all(tensor.grad is None for tensor in tensors))
        result = finalize_diagnostics(sums.report(), learning_rates={}, warming=False)
        w = weights[:2]
        mean = lambda rows: (rows * w).sum().item() / w.sum().item()
        expected = {
            'dmd_stabilizer/normalizer': mean((vt[:2] - target[:2]).abs().mean((1, 2, 3, 4))),
            'dmd_stabilizer/difference': mean((vt[:2] - vf[:2]).abs().mean((1, 2, 3, 4))),
            'diagnostic/stabilizer/teacher_critic_velocity_cosine': mean(
                F.cosine_similarity(vt[:2].flatten(1), vf[:2].flatten(1), dim=1)),
            'diagnostic/stabilizer/teacher_critic_velocity_rms': mean(
                (vt[:2] - vf[:2]).square().mean((1, 2, 3, 4))) ** .5,
            'diagnostic/stabilizer/output_rms': mean(y[:2].square().mean((1, 2, 3, 4))) ** .5,
        }
        for key, value in expected.items():
            self.assertAlmostEqual(result[key], value, places=6, msg=key)

    def test_global_reduction_precedes_rms_and_ignores_empty_rank(self):
        y = torch.tensor([1., 3., 100.]).view(3, 1, 1, 1, 1).expand(-1, 1, 4, 1, 1)
        vt, vf = y * 2, y * 0
        weights = torch.tensor([1., 1., 0.])
        whole, partitioned = CPUSums(), CPUSums()
        add_dmd_diagnostics(whole, 'resolver', y, vt, vf, vt, weights)
        for index in range(3):
            part = slice(index, index + 1)
            add_dmd_diagnostics(partitioned, 'resolver', y[part], vt[part], vf[part], vt[part], weights[part])
        self.assertEqual(whole.report(), partitioned.report())
        result = finalize_diagnostics(partitioned.report(), learning_rates={}, warming=False)
        self.assertEqual(result['dmd_resolver/normalizer'], 0.)  # Never clamp logging to epsilon.
        self.assertAlmostEqual(result['diagnostic/resolver/output_rms'], 5 ** .5)
        self.assertNotEqual(result['diagnostic/resolver/output_rms'], 2.)  # Not mean(row RMS).

    def test_diagnostics_do_not_change_existing_dmd_loss_or_gradients(self):
        source = ast.parse((Path(__file__).resolve().parents[1] / 'learn_dmd_joint.py').read_text())
        function = next(node for node in source.body if isinstance(node, ast.FunctionDef)
                        and node.name == 'dmd_frames')
        namespace = {'torch': torch}
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'learn_dmd_joint.py', 'exec'), namespace)
        torch.manual_seed(34)
        y, vt, vf, target = [torch.randn(2, 2, 4, 1, 2) for _ in range(4)]
        original, observed = y.clone().requires_grad_(), y.clone().requires_grad_()
        expected = namespace['dmd_frames'](original, vt, vf, target, 1e-8)
        actual = namespace['dmd_frames'](observed, vt, vf, target, 1e-8)
        add_dmd_diagnostics(CPUSums(), 'resolver', observed, vt, vf, target, torch.tensor([1., 0.]))
        expected[0].mean().backward()
        actual[0].mean().backward()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(observed.grad, original.grad, rtol=0, atol=0)

    def test_resolver_history_counts_repeated_attention_slots_and_padding(self):
        ofi = torch.arange(11)[None].repeat(3, 1)
        ofi[0, 5] = -1  # First shared parent appears once.
        ofi[1, 6] = -1  # Next parent appears twice.
        ofi[2, 5:10] = -1  # Padded row contributes neither numerator nor denominator.
        sums = CPUSums()
        add_resolver_history_dropout(sums, ofi, torch.tensor([1., .5, 0.]), target_start=7)
        self.assertAlmostEqual(sums.report()['data/resolver_history_slots_dropped'], 2 / 12)
        shifted = torch.cat((torch.zeros(3, 2, dtype=ofi.dtype), ofi), 1)
        other = CPUSums()
        add_resolver_history_dropout(other, shifted, torch.tensor([1., .5, 0.]), target_start=9)
        self.assertEqual(sums.report(), other.report())

    def test_loss_aliases_lr_next_and_phase_flags(self):
        metrics = {'critic/stabilizer': 2., 'student/resolver': 3.,
                   **{f'critic/stabilizer/frame_{i}': i + .5 for i in range(4)}}
        result = finalize_diagnostics(metrics, learning_rates={'student': 1e-6, 'critic': 2e-6}, warming=True)
        self.assertEqual(result['fm_stabilizer/loss'], 2.)
        self.assertEqual(result['dmd_resolver/loss'], 3.)
        for i in range(4):
            self.assertEqual(result[f'fm_stabilizer_frame/{i}'], i + .5)
        self.assertEqual(result['student/lr_next'], 1e-6)
        self.assertEqual((result['phase/critic_warmup'], result['phase/joint']), (1, 0))
        self.assertNotIn('fm_stabilizer/loss', metrics)

    def test_timing_modes_sum_before_max_rank_reduction(self):
        def rank(stabilizer, resolver):
            from distillation.joint_metrics import TIMING_ALIASES
            base = {f'seconds/{name}': 0. for names in TIMING_ALIASES.values() for name in names}
            base.update({'seconds/student_forward/stabilizer': stabilizer,
                         'seconds/student_forward/resolver': resolver})
            return add_timing_aliases(base)
        a, b = rank(10., 1.), rank(1., 10.)
        self.assertEqual(max(a['seconds/student_forward'], b['seconds/student_forward']), 11.)
        self.assertNotEqual(max(a['seconds/student_forward/stabilizer'], b['seconds/student_forward/stabilizer'])
                            + max(a['seconds/student_forward/resolver'], b['seconds/student_forward/resolver']), 11.)


class RolloutTimingHookTests(unittest.TestCase):
    def setUp(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch('distillation.bimodal_net.flash_attn_func', attention_cpu))
        stack.enter_context(patch('models.pose.audio.flash_attn_func', attention_cpu))
        stack.enter_context(patch.object(torch.Tensor, 'bfloat16', lambda tensor, *a, **kw: tensor.float()))

    def test_hooks_preserve_outputs_all_gradients_rng_and_model_calls(self):
        for objective, prefix in (('stabilizer', 3), ('stabilizer', 6), ('resolver', 6)):
            with self.subTest(objective=objective, prefix=prefix):
                torch.manual_seed(314)
                original = small_net()
                hooked = copy.deepcopy(original)
                original.configure(False, 'block')
                hooked.configure(False, 'block')
                rows, _ = rows_and_noise(batch=1)
                copied = clone_rows(rows)
                events = []

                @contextmanager
                def timer(stage):
                    events.append(('enter', stage))
                    try:
                        yield
                    finally:
                        events.append(('exit', stage))

                def run(net, inputs, timing):
                    torch.manual_seed(781)
                    with patch.object(net, 'forward_pass', wraps=net.forward_pass) as calls:
                        if objective == 'stabilizer':
                            output, context = generate_stabilizer(net, inputs, prefix_frames=prefix,
                                history_dropout=(.1, .2, .3), timed=timing)
                        else:
                            output, context = generate_resolver(net, inputs, timed=timing), {}
                        output.square().mean().backward()
                        return output, context, calls.call_count, torch.get_rng_state().clone()

                expected, expected_context, expected_calls, expected_rng = run(original, rows, None)
                actual, context, calls, rng = run(hooked, copied, timer)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(calls, expected_calls)
                self.assertTrue(torch.equal(rng, expected_rng))
                for key in context:
                    torch.testing.assert_close(context[key], expected_context[key], rtol=0, atol=0)
                for (name, parameter), (_, reference) in zip(hooked.named_parameters(), original.named_parameters()):
                    if reference.grad is None:
                        self.assertIsNone(parameter.grad, name)
                    else:
                        torch.testing.assert_close(parameter.grad, reference.grad, rtol=0, atol=0, msg=name)
                for key in rows:
                    if rows[key].grad is None:
                        self.assertIsNone(copied[key].grad, key)
                    else:
                        torch.testing.assert_close(copied[key].grad, rows[key].grad, rtol=0, atol=0, msg=key)
                self.assertEqual(collections_counter(events, 'enter'), collections_counter(events, 'exit'))
                self.assertEqual(set(stage for event, stage in events),
                                 {'context', 'forward'} if objective == 'stabilizer' else {'input_generation', 'forward'})


class TrainingMetricProjectionTests(unittest.TestCase):
    def full_row(self):
        row = {source: float(index + 1) for index, source in enumerate(TRAINING_METRIC_MAP.values())}
        row.update({'phase': 'joint', 'phase/joint': 1, 'phase/critic_warmup': 0})
        row.update({alias: row[source] for alias, source in TRAINING_METRIC_ALIASES.items()})
        return row

    def test_full_contract_has_nine_core_fields_and_no_uncovered_raw_scalars(self):
        row = self.full_row()
        before = copy.deepcopy(row)
        rng = torch.get_rng_state().clone()
        projected = project_training_metrics(row)
        self.assertEqual(len(projected), 98)
        self.assertEqual(sum(key.startswith('train/') for key in projected), 9)
        self.assertEqual(len(row) - len(TRAINING_METADATA_FIELDS), 123)
        self.assertEqual(set(row), set(TRAINING_METRIC_MAP.values()) |
                         set(TRAINING_METRIC_ALIASES) | TRAINING_METADATA_FIELDS)
        for key, source in TRAINING_METRIC_MAP.items():
            self.assertEqual(projected[key], row[source])
        self.assertEqual(row, before)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertNotIn('dmd_stabilizer/loss', projected)
        self.assertNotIn('diagnostic/stabilizer_output_rms', projected)

    def test_historical_missing_diagnostics_stay_missing_and_throughput_is_actual(self):
        result = project_training_metrics({'rounds': 25, 'phase': 'critic_warmup', 'critic/loss': .5,
            'seconds/wall': 2., 'global_valid_rows': 3,
            'global_objective_targets': {'stabilizer': 12, 'resolver': 12}})
        self.assertEqual(result['train/round'], 25)
        self.assertEqual(result['phase/critic_warmup'], 1)
        self.assertEqual(result['throughput/unique_rows_per_second'], 1.5)
        self.assertEqual(result['throughput/objective_targets_per_second'], 12.)
        self.assertFalse(any('normalizer' in key or 'diagnostics/' in key or 'next' in key for key in result))
        self.assertNotIn('train/student_loss', result)

    def test_collisions_unmapped_scalars_and_conflicting_aliases_fail_visibly(self):
        cases = (
            ({'global_objective_rows': {'resolver': 1}, 'global_objective_rows/resolver': 1}, 'collision'),
            ({'unexpected/new_metric': .5}, 'Unmapped training scalar'),
            ({'student/stabilizer': 1., 'dmd_stabilizer/loss': 2.}, 'Conflicting local metric alias'),
            ({'phase': 'joint', 'phase/joint': 0}, 'phase indicator disagrees'),
        )
        for row, message in cases:
            with self.subTest(row=row):
                with self.assertRaisesRegex(ValueError, message):
                    project_training_metrics(row)


def collections_counter(events, event):
    from collections import Counter
    return Counter(stage for action, stage in events if action == event)


if __name__ == '__main__':
    unittest.main()
