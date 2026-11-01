import math
from datetime import timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from distillation.training_metrics import BUCKETS, OBJECTIVES, TRAINING_METRIC_KEYS, TrainingMetrics


COEFFICIENTS = {'stabilizer': .5, 'resolver': .5}


def observations(kind, batch=2, channels=2):
    times = torch.tensor([[.4, .6, .8, 1.] if kind == 'stabilizer' else [.05, .1, .15, .2]]).repeat(batch, 1)
    frames = torch.arange(1, batch * 4 + 1, dtype=torch.float32).view(batch, 4)
    teacher = torch.arange(1, batch * channels * 4 * 2 + 1, dtype=torch.float32).view(batch, channels, 4, 1, 2)
    critic = teacher * .25 + 1
    student = teacher * 2
    target = teacher - .5
    return frames, times, teacher, critic, student, target, torch.ones(batch)


def observe_student(metrics, kind, velocities, weights):
    for velocity in velocities.split(1, dim=2):
        metrics.add_student_rollout(kind, velocity, weights)


def distributed_metrics_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=Path(rendezvous).as_uri(), rank=rank,
                            world_size=2, timeout=timedelta(seconds=30))
    try:
        metrics, whole = TrainingMetrics('cpu', channels=2), TrainingMetrics('cpu', channels=2)
        frames, times, teacher, critic, student, target, weights = observations('stabilizer', batch=3)
        weights[:] = torch.tensor([1., .25, 0.])
        part = slice(0, 1) if rank == 0 else slice(1, 3)
        observe_student(metrics, 'stabilizer', student[part], weights[part])
        observe_student(whole, 'stabilizer', student, weights)
        metrics.add_dmd('stabilizer', frames[part], times[part], teacher[part], critic[part],
                       target[part], weights[part], 1e-8)
        whole.add_dmd('stabilizer', frames, times, teacher, critic, target, weights, 1e-8)
        assert metrics.report(COEFFICIENTS) == whole.report(COEFFICIENTS, reduce=False)
        metrics.reset()
        if rank == 0:
            metrics.add_fm('resolver', frames[:1, :1], torch.tensor([[.05]]), critic[:1, :, :1],
                          target[:1, :, :1], weights[:1])
        online, local = metrics.report(COEFFICIENTS)
        assert len(online) == 3, online
        assert online['train/critic_fm_resolver_loss'] == 1.
        assert local['critic_fm_resolver/timewise_endpoint_loss/t_0.00_0.05']['count'] == 1.
        assert 'train/critic_fm_loss' not in online
    finally:
        dist.destroy_process_group()


class TrainingMetricsTests(unittest.TestCase):
    def test_exact_54_series_schema_and_balanced_totals(self):
        metrics = TrainingMetrics('cpu', channels=2)
        for kind in OBJECTIVES:
            frames, times, teacher, critic, student, target, weights = observations(kind)
            factor = 2 if kind == 'stabilizer' else 3
            metrics.add_fm(kind, frames * factor, times, critic, target, weights)
            observe_student(metrics, kind, student, weights)
            metrics.add_dmd(kind, frames, times, teacher, critic, target, weights, 1e-8)
        metrics.add_lr('student', 1e-5)
        for rate in (1e-4, 2e-4, 3e-4, 4e-4):
            metrics.add_lr('critic', rate)
        result, local = metrics.report(COEFFICIENTS, reduce=False)
        self.assertEqual(len(TRAINING_METRIC_KEYS), 54)
        self.assertEqual(set(result), TRAINING_METRIC_KEYS)
        self.assertEqual(result['train/dmd_loss'], 4.5)
        self.assertEqual(result['train/critic_fm_loss'], 11.25)
        self.assertAlmostEqual(result['train/critic_lr'], 2.5e-4)
        self.assertEqual(result['train/student_lr'], 1e-5)
        self.assertEqual(local['train/critic_lr'], {'sum': .001, 'count': 4.})
        self.assertNotIn('train/step', result)

    def test_upper_inclusive_bucket_boundaries_and_padding(self):
        metrics = TrainingMetrics('cpu', channels=2)
        for kind in OBJECTIVES:
            frames, times, teacher, critic, student, target, weights = observations(kind, batch=3)
            weights[:] = torch.tensor([1., .25, 0.])
            for tensor in (frames, times, teacher, critic, student, target):
                tensor[2] = float('nan')
            metrics.add_fm(kind, frames, times, critic, target, weights)
            observe_student(metrics, kind, student, weights)
            metrics.add_dmd(kind, frames, times, teacher, critic, target, weights, 1e-8)
            result, local = metrics.report(COEFFICIENTS, reduce=False)
            for index, bucket in enumerate(BUCKETS[kind]):
                expected = (index + 1 + .25 * (index + 5)) / 1.25
                self.assertAlmostEqual(result[f'dmd_{kind}/timewise_loss/{bucket}'], expected)
                self.assertEqual(local[f'dmd_{kind}/timewise_loss/{bucket}']['count'], 1.25)
            expected_raw = ((critic[:2] - target[:2]).square().mean((1, 3, 4), dtype=torch.float64) * weights[:2, None]).sum() / 5
            raw = [result[f'critic_fm_{kind}/timewise_velocity_loss/{bucket}'] for bucket in BUCKETS[kind]]
            self.assertAlmostEqual(sum(raw) / 4, expected_raw.item())

    def test_rms_variance_and_channel_bias_reduce_before_nonlinearity(self):
        metrics = TrainingMetrics('cpu', channels=2)
        teacher = torch.tensor([[1., 2.], [3., 6.]]).view(2, 2, 1, 1, 1)
        delta = torch.tensor([[2., -2.], [4., -4.]]).view_as(teacher)
        student, critic = teacher * 2, teacher + delta
        weights = torch.tensor([1., .5])
        metrics.add_student_rollout('resolver', student, weights)
        metrics.add_dmd('resolver', torch.ones(2, 1), torch.full((2, 1), .1), teacher,
                        critic, teacher - .25, weights, 1e-8)
        result, local = metrics.report(COEFFICIENTS, reduce=False)
        prefix = 'velocity_probes/resolver'
        self.assertAlmostEqual(result[f'{prefix}/student_rollout_rms'], math.sqrt((20 + .5 * 180) / 3))
        self.assertNotIn(f'{prefix}/student_rms', result)
        self.assertEqual(result[f'{prefix}/critic_minus_teacher_mean'], 0)
        self.assertAlmostEqual(result[f'{prefix}/critic_minus_teacher_variance'], (8 + .5 * 32) / 3)
        self.assertAlmostEqual(result[f'{prefix}/critic_minus_teacher_channel_bias_rms'], 4 / 1.5)
        self.assertEqual(local[f'{prefix}/delta']['count'], 3)
        self.assertAlmostEqual(result[f'{prefix}/teacher_critic_cosine'], (3 / math.sqrt(5 * 9)
            + .5 * 33 / math.sqrt(45 * 53)) / 1.5)

    def test_effective_normalizer_floor_and_zero_cosine_counts(self):
        metrics = TrainingMetrics('cpu', channels=1)
        frames, times = torch.ones(3, 1), torch.full((3, 1), .1)
        teacher = torch.tensor([0., 1., 1.]).view(3, 1, 1, 1, 1)
        critic = torch.tensor([1., 0., 1.]).view_as(teacher)
        target = teacher - torch.tensor([0., 0., .25]).view_as(teacher)
        metrics.add_dmd('resolver', frames, times, teacher, critic, target, torch.tensor([1., 2., 1.]), .1)
        result, local = metrics.report(COEFFICIENTS, reduce=False)
        base = 'dmd_resolver/timewise_'
        self.assertAlmostEqual(result[base + 'normalizer/t_0.05_0.10'], .1375, places=7)
        self.assertAlmostEqual(local[base + 'raw_normalizer/t_0.05_0.10']['sum'], .25)
        self.assertEqual(local[base + 'normalizer_floor_hits/t_0.05_0.10'], {'sum': 3., 'count': 4.})
        self.assertEqual(result['velocity_probes/resolver/teacher_critic_cosine'], 1.)
        self.assertEqual(local['velocity_probes/resolver/teacher_critic_cosine']['count'], 1.)
        self.assertEqual(local['velocity_probes/resolver/zero_norm_cosine_targets']['sum'], 3.)

    def test_empty_round_and_warmup_missing_student_metrics(self):
        metrics = TrainingMetrics('cpu', channels=2)
        self.assertEqual(metrics.report(COEFFICIENTS, reduce=False)[0], {})
        for kind in OBJECTIVES:
            frames, times, _, critic, _, target, weights = observations(kind)
            times[:] = .3 if kind == 'stabilizer' else .03
            metrics.add_fm(kind, frames, times, critic, target, weights)
        metrics.add_lr('critic', 1e-4)
        result, _ = metrics.report(COEFFICIENTS, reduce=False)
        self.assertEqual(len(result), 8)
        self.assertFalse(any('dmd' in key or 'student' in key or 'velocity_probes' in key for key in result))
        self.assertNotIn('critic_fm_resolver/timewise_endpoint_loss/t_0.05_0.10', result)
        metrics.reset()
        self.assertEqual(metrics.report(COEFFICIENTS, reduce=False)[0], {})

    def test_single_packed_global_reduction_with_unequal_rank_counts(self):
        first, second, whole = [TrainingMetrics('cpu', channels=2) for _ in range(3)]
        frames, times, teacher, critic, student, target, weights = observations('stabilizer', batch=3)
        weights[:] = torch.tensor([1., .25, 0.])
        frames[2] = float('nan')
        for metric, part in ((first, slice(0, 1)), (second, slice(1, 3)), (whole, slice(None))):
            observe_student(metric, 'stabilizer', student[part], weights[part])
            metric.add_dmd('stabilizer', frames[part], times[part], teacher[part], critic[part],
                           target[part], weights[part], 1e-8)
        expected, expected_local = whole.report(COEFFICIENTS, reduce=False)
        before = first.values.clone()
        with patch('distillation.training_metrics.dist.is_initialized', return_value=True), \
                patch('distillation.training_metrics.dist.all_reduce', side_effect=lambda values, op: values.add_(second.values)) as reduce:
            actual, actual_local = first.report(COEFFICIENTS)
        reduce.assert_called_once()
        self.assertEqual(actual, expected)
        self.assertEqual(actual_local, expected_local)
        torch.testing.assert_close(first.values, before, rtol=0, atol=0)
        empty = TrainingMetrics('cpu', channels=2)
        self.assertEqual(empty.values.shape, first.values.shape)

    def test_gloo_reduction_with_unequal_counts_and_empty_rank(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(distributed_metrics_worker, args=(str(Path(directory) / 'rendezvous'),), nprocs=2, join=True)

    def test_no_rng_or_graph_changes(self):
        torch.manual_seed(34)
        metrics = TrainingMetrics('cpu', channels=2)
        frames, times, teacher, critic, student, target, weights = observations('resolver')
        for tensor in (frames, teacher, critic, student, target):
            tensor.requires_grad_()
        rng = torch.get_rng_state().clone()
        metrics.add_fm('resolver', frames, times, critic, target, weights)
        observe_student(metrics, 'resolver', student, weights)
        metrics.add_dmd('resolver', frames, times, teacher, critic, target, weights, 1e-8)
        metrics.report(COEFFICIENTS, reduce=False)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertFalse(metrics.values.requires_grad)
        self.assertIsNone(metrics.values.grad_fn)
        self.assertTrue(all(tensor.grad is None for tensor in (frames, teacher, critic, student, target)))
        frames.sum().backward()
        torch.testing.assert_close(frames.grad, torch.ones_like(frames))

    def test_invalid_valid_row_data_fails_but_zero_weight_rows_do_not(self):
        for case in ('weights', 'times', 'velocity'):
            with self.subTest(case=case):
                metrics = TrainingMetrics('cpu', channels=2)
                frames, times, _, critic, _, target, weights = observations('stabilizer')
                if case == 'weights':
                    weights[0] = -1
                elif case == 'times':
                    times[0, 0] = .2
                else:
                    critic[0] = float('nan')
                metrics.add_fm('stabilizer', frames, times, critic, target, weights)
                with self.assertRaises((ValueError, FloatingPointError)):
                    metrics.report(COEFFICIENTS, reduce=False)

    def test_resolver_includes_zero_and_all_zero_cosines_are_missing(self):
        metrics = TrainingMetrics('cpu', channels=1)
        frames, times, velocity = torch.ones(1, 1), torch.zeros(1, 1), torch.zeros(1, 1, 1, 1, 1)
        metrics.add_dmd('resolver', frames, times, velocity, velocity, velocity, torch.ones(1), 1e-8)
        result, _ = metrics.report(COEFFICIENTS, reduce=False)
        self.assertEqual(result['dmd_resolver/timewise_loss/t_0.00_0.05'], 1.)
        self.assertNotIn('velocity_probes/resolver/teacher_critic_cosine', result)
        self.assertEqual(result['velocity_probes/resolver/critic_rms'], 0.)


if __name__ == '__main__':
    unittest.main()
