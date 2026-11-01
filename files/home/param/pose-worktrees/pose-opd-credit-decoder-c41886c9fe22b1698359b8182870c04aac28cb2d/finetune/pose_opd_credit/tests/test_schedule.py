from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import unittest

from finetune.pose_opd_credit.config import TrainingConfig
from finetune.pose_opd_credit.schedule import resolve_schedule, multiplier, apply_learning_rate
from finetune.pose_opd_credit.train import make_plan, rollout_seed


class ScheduleTests(unittest.TestCase):
    def test_amu_boundaries_and_last_applied_floor(self):
        cfg = TrainingConfig(learning_rate_schedule='amu_opd')
        schedule = resolve_schedule(cfg, 129)
        self.assertEqual((schedule['warmup_updates'], schedule['constant_updates'], schedule['decay_updates']), (7, 109, 13))
        self.assertEqual(multiplier(schedule, 0), .1)
        self.assertAlmostEqual(multiplier(schedule, 6), .1 + .9 * 6 / 7)
        self.assertEqual(multiplier(schedule, 7), 1.)
        self.assertEqual(multiplier(schedule, 115), 1.)
        self.assertEqual(multiplier(schedule, 116), 1.)
        self.assertAlmostEqual(multiplier(schedule, 117), .925)
        self.assertAlmostEqual(multiplier(schedule, 128), .1)
        self.assertAlmostEqual(multiplier(schedule, 129), .1)
        values = [multiplier(schedule, x) for x in range(129)]
        self.assertTrue(all(.0999999 <= x <= 1 for x in values))

    def test_source_linear_matches_native_lambda_indexing(self):
        cfg = TrainingConfig(learning_rate_schedule='source_linear')
        schedule = resolve_schedule(cfg, 129)
        self.assertEqual(schedule['warmup_updates'], 4)
        source = Path(__file__).parents[2] / 'duplex_distill/production/lib/sched_s3q.py'
        spec = importlib.util.spec_from_file_location('native_schedule_reference', source)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        reference = module.make_lambda(True, dict(warmup=4, horizon=129, min_lr_rate=0,
            s3_sched='wsd', wsd_decay_start=4, wsd_decay_end=129, wsd_floor=0, wsd_shape='linear'))
        for step in range(131):
            self.assertEqual(multiplier(schedule, step), reference(step))
        self.assertGreater(multiplier(schedule, 128), 0.)
        self.assertEqual(multiplier(schedule, 129), 0.)

    def test_resume_reconstructs_next_lr_without_epoch_or_benchmark_reset(self):
        cfg = TrainingConfig(learning_rate_schedule='amu_opd')
        schedule = resolve_schedule(cfg, 129)
        optimizer = SimpleNamespace(param_groups=[{'lr': 999.}, {'lr': -1.}])
        uninterrupted = [apply_learning_rate(optimizer, 3e-6, schedule, step) for step in range(129)]
        for cursor in (0, 7, 43, 60, 86, 120, 128):
            resumed = SimpleNamespace(param_groups=[{'lr': 999.}])
            self.assertEqual(apply_learning_rate(resumed, 3e-6, schedule, cursor), uninterrupted[cursor])
            self.assertEqual(resumed.param_groups[0]['lr'], uninterrupted[cursor])

    def test_invalid_and_short_schedules(self):
        for changes in ({'learning_rate_schedule': 'unknown'},
                        {'learning_rate_warmup_steps': 2},
                        {'learning_rate_schedule': 'amu_opd', 'learning_rate_warmup_steps': True}):
            with self.assertRaises(ValueError):
                replace(TrainingConfig(), **changes).validate()
        with self.assertRaises(ValueError):
            resolve_schedule(TrainingConfig(learning_rate_schedule='amu_opd', learning_rate_warmup_steps=129), 129)
        with self.assertRaises(ValueError):
            resolve_schedule(TrainingConfig(learning_rate_schedule='source_linear'), 1)
        self.assertAlmostEqual(multiplier(resolve_schedule(TrainingConfig(learning_rate_schedule='amu_opd'), 1), 0), .1)

    def test_three_epoch_coverage_changes_shuffle_and_rollout_seeds(self):
        items = [(0, row, 0) for row in range(13566)]
        plan = make_plan(items, world=40, epochs=3, seed=20261031, rows_per_rank=8,
                         frame_lengths={item: 1500 for item in items})
        self.assertEqual(len(plan['updates']), 129)
        orders = []
        for epoch in range(3):
            rows = [tuple(x) for batch in plan['updates'][epoch*43:(epoch+1)*43] for x in batch if x is not None]
            self.assertCountEqual(rows, items); orders.append(rows)
        self.assertNotEqual(orders[0], orders[1]); self.assertNotEqual(orders[1], orders[2])
        self.assertTrue(all(sum(item is not None for item in batch[:8]) >= 4 for batch in plan['updates']))
        seeds = [rollout_seed(20261031, step, 0, 0) for step in (0, 43, 86)]
        self.assertEqual(len(set(seeds)), 3)
        self.assertEqual(seeds[1], rollout_seed(20261031, 43, 0, 0))


if __name__ == '__main__':
    unittest.main()
