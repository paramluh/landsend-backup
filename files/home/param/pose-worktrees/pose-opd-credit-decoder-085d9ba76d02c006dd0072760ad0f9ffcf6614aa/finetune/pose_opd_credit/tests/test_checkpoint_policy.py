"""Checkpoint scheduling and export retention contract without GPU allocation."""
from dataclasses import replace
import unittest

from finetune.pose_opd_credit.config import TrainingConfig
from finetune.pose_opd_credit.train import checkpoint_deadline, checkpoint_due, checkpoint_export_config


class CheckpointPolicyTests(unittest.TestCase):
    def test_absolute_step_boundaries_and_final_short_epoch(self):
        cfg = TrainingConfig(checkpoint_interval_seconds=None, checkpoint_interval_steps=60).validate()
        self.assertIsNone(checkpoint_deadline(cfg, 1000.))
        due = lambda update, limit=130: checkpoint_due(cfg, completed_update=update, limit=limit,
                                                      now=1e20, deadline=None)
        self.assertEqual([i for i in range(1, 131) if due(i)], [60, 120, 130])
        self.assertFalse(due(59)); self.assertTrue(due(60)); self.assertFalse(due(61))
        self.assertEqual([i for i in range(1, 44) if due(i, 43)], [43])
        # Resume at59 still uses absolute update60, not60 updates after resume.
        self.assertEqual([i for i in range(59, 63) if due(i)], [60])

    def test_time_and_step_policies_are_combined_and_deadline_resets(self):
        cfg = TrainingConfig(checkpoint_interval_seconds=10., checkpoint_interval_steps=60).validate()
        deadline = checkpoint_deadline(cfg, 100.)
        self.assertEqual(deadline, 110.)
        def due(step, now):
            return checkpoint_due(cfg, completed_update=step, limit=100, now=now, deadline=deadline)
        self.assertFalse(due(1, 109.9))
        self.assertTrue(due(1, 110.))
        self.assertTrue(due(60, 101.))
        self.assertTrue(due(100, 101.))
        self.assertEqual(checkpoint_deadline(cfg, 112.), 122.)

    def test_invalid_policies_rejected_and_configuration_roundtrips(self):
        for changes in ({'checkpoint_interval_seconds': None},
                        *({'checkpoint_interval_steps': x} for x in (0, -1, 1.5, True)),
                        *({'checkpoint_interval_seconds': x} for x in (0, -1, float('inf'), True)),
                        {'keep_latest_optimizer_state': 1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(TrainingConfig(), **changes).validate()
        cfg = TrainingConfig(checkpoint_interval_seconds=None, checkpoint_interval_steps=60,
                             keep_latest_optimizer_state=True).validate()
        self.assertEqual(TrainingConfig.from_dict(cfg.to_dict()), cfg)

    def test_saved_config_uses_retention_flag_and_preserves_reconstruction(self):
        spec = {'config': {'original': 'metadata', 'keep_latest_optimizer_state': False},
                'vadapter': {'norm': 'none'}, 'decp_bb_alpha': 1.,
                'layers': 64, 'tower_layers': 8, 'dec_layers': 6, 'proj_r': 1024, 'tower_vocab': 2048}
        for keep in (False, True):
            cfg = TrainingConfig(keep_latest_optimizer_state=keep)
            saved = checkpoint_export_config(cfg, spec, identity={'run': 1}, data_config={'voice_frac': 1.},
                                             next_update=43, plan_digest='plan')
            self.assertIs(saved['keep_latest_optimizer_state'], keep)
            self.assertEqual(saved['_runtime_state'], {'next_update': 43, 'plan_digest': 'plan'})
            self.assertEqual(saved['original'], 'metadata')
            self.assertEqual(saved['layers'], 64)
            self.assertEqual(saved['opd_identity'], {'run': 1})
        self.assertIs(spec['config']['keep_latest_optimizer_state'], False)


if __name__ == '__main__':
    unittest.main()
