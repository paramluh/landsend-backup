"""Maximum-shape synthetic stress is distinct from actor-generated training."""
import contextlib
import copy
import io
import json
import unittest

import numpy as np
import torch

from finetune.pose_opd_credit.benchmark import cache_identity, replay_candidates
from finetune.pose_opd_credit.config import TrainingConfig
from finetune.pose_opd_credit.data import replay_from_rollout
from finetune.pose_opd_credit.shape_benchmark import worst_shape_plan, validate_shape_seed, synthetic_rollout
from finetune.pose_opd_credit.tests.test_rollout import seed_row
from finetune.pose_opd_credit.train import args_parser


class ShapeBenchmarkTests(unittest.TestCase):
    def test_adversarial_plan_duplicates_joint_maximum_without_changing_production_sampler(self):
        specs = [dict(item=[0, 1, 0], conversation_frames=7000, prefix_frames=134, total_frames=7136),
                 dict(item=[0, 2, 0], conversation_frames=7500, prefix_frames=134, total_frames=7636)]
        plan = worst_shape_plan(specs, world=40, rows_per_rank=4)
        self.assertEqual(len(plan['updates']), 1)
        self.assertEqual(plan['updates'][0], [[0, 2, 0]] * 160)
        self.assertTrue(plan['synthetic_shape_only'])
        self.assertEqual(plan['independent_maxima'], {'conversation_frames': 7500, 'prefix_frames': 134})
        changed = copy.deepcopy(specs)
        changed[1].update(prefix_frames=5, total_frames=7507)
        with self.assertRaisesRegex(ValueError, 'both maximum'):
            worst_shape_plan(changed, world=40, rows_per_rank=4)

    def test_synthetic_tokens_use_only_prefill_and_leave_inputs_and_forced_values_exact(self):
        row = seed_row(frames=90, prefix=11)
        row['wav'] = np.arange(90 * 1920, dtype=np.float32)
        before = copy.deepcopy(row)
        spec = dict(conversation_frames=90, prefix_frames=11, total_frames=103)
        self.assertEqual(validate_shape_seed(row, spec), spec)
        rollout = synthetic_rollout(row)
        self.assertEqual(rollout.behavior_logprobs, {})
        self.assertEqual(rollout.policy_probes, ())
        for stream, values, original, delay in (('text', rollout.text, row['ids'][1], 0),
                                               ('audio', rollout.audio, row['depth'][1], 2),
                                               ('pose', rollout.pose, row['pdepth'], 2)):
            mask = row['sampled_masks'][stream]
            known = original[11 + delay:11 + delay + 25]
            np.testing.assert_array_equal(values[mask], known[np.arange(mask.sum()) % 25])
            np.testing.assert_array_equal(values[~mask], original[~mask])
        for key in ('ids', 'depth', 'pdepth', 'present', 'wav'):
            np.testing.assert_array_equal(row[key], before[key])
        bundle = replay_from_rollout(row, rollout.text, rollout.audio, rollout.pose)
        np.testing.assert_array_equal(bundle['student']['ids'][0], before['ids'][0])
        np.testing.assert_array_equal(bundle['student']['wav'], before['wav'])
        with self.assertRaisesRegex(ValueError, 'maximum shape'):
            validate_shape_seed(row, dict(spec, prefix_frames=134))

    def test_shape_cli_forces_full_unsaved_warm_single_update_and_no_policy_claims(self):
        base = ['--config', 'config.json', '--run-dir', '/tmp/shape']
        allowed = ['--benchmark-shape-only', '--warm-optimizer', '--no-save', '--max-updates', '1']
        args = args_parser(base + allowed + ['--profile-replay-stages'])
        self.assertTrue(args.benchmark_shape_only)
        for flags in (['--benchmark-shape-only'],
                      ['--benchmark-shape-only', '--no-save', '--max-updates', '1'],
                      allowed + ['--benchmark-prefix-frames', '750'],
                      allowed + ['--profile-policy-agreement'], allowed + ['--profile-actor-graphs'],
                      allowed + ['--resume', '/tmp/checkpoint']):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                args_parser(base + flags)

    def test_replay_variant_matrix_is_learner_only_and_each_variant_uses_base_defaults(self):
        base = ['--config', 'config.json', '--run-dir', '/tmp/variants', '--no-save', '--max-updates', '1']
        variants = [{'physical_microbatch': 2, 'decoder_gradient_sync': 'per_replay', 'checkpoint_depth_decoder': False},
                    {'physical_microbatch': 2, 'decoder_chunk_frames': 256, 'text_chunk_frames': 128}]
        args = args_parser(base + ['--benchmark-replay-variants', json.dumps(variants)])
        self.assertEqual(args.benchmark_replay_variants, variants)
        config = TrainingConfig(effective_batch_per_rank=4)
        parameter = torch.nn.Parameter(torch.ones(2))
        optimizer = torch.optim.AdamW([parameter])
        observed = []
        for candidate in replay_candidates(config, None, optimizer, variants=variants):
            self.assertIsNone(parameter.grad)
            observed.append(candidate)
            parameter.grad = torch.ones_like(parameter)
        self.assertFalse(observed[0].checkpoint_depth_decoder)
        self.assertTrue(observed[1].checkpoint_depth_decoder)
        self.assertEqual(observed[1].decoder_gradient_sync, 'per_chunk')
        for value in ('[]', '{}', '[{}]', '[{"learning_rate": 0.1}]', 'bad'):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                args_parser(base + ['--benchmark-replay-variants', value])
        identity = {'method': config.to_dict(), 'world': 2, 'benchmark_shape_only': False}
        real = cache_identity(identity, update=0, rank=0)
        identity['benchmark_shape_only'] = True
        self.assertNotEqual(real, cache_identity(identity, update=0, rank=0))


if __name__ == '__main__':
    unittest.main()
