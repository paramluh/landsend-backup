import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from distillation.checkpoint_retention import checkpoint_compatibility, enforce_checkpoint_retention
from distillation.checkpoints import complete_checkpoint


class CheckpointRetentionTests(unittest.TestCase):
    def checkpoint(self, root, step, *, complete=True, foreign=False, signature=None):
        path = root / f'step-{step:06d}-fixture'
        path.mkdir()
        for name in ('student.safetensors', 'critic.safetensors', 'model_ema.safetensors', 'rank-00000.pt'):
            (path / name).write_bytes((name + str(step)).encode())
        if complete:
            complete_checkpoint(path, {'world_size': 1, 'optimizer_state_retained': True,
                'counters': {'rounds': step}, 'compatibility': signature if signature is not None else
                {'dataset': 'foreign' if foreign else 'ours'}})
        return path

    def latest(self, root, path):
        meta = json.loads((path / 'metadata.json').read_text())
        (root / 'latest.json').write_text(json.dumps({'checkpoint': str(path), 'counters': meta['counters']}))

    def test_latest_eight_keep_all_three_weights_and_one_optimizer(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = [self.checkpoint(root, x) for x in range(1, 12)]
            self.latest(root, paths[-1])
            pending = self.checkpoint(root, 12, complete=False)
            (root / '.step-000013.partial').mkdir()
            result = enforce_checkpoint_retention(paths[-1], 8)
            self.assertEqual(len(result['checkpoint_prunes']), 3)
            self.assertEqual(len(result['optimizer_prunes']), 7)
            for path in paths[:3]:
                self.assertFalse(path.exists())
            for path in paths[3:]:
                for name in ('student.safetensors', 'critic.safetensors', 'model_ema.safetensors'):
                    self.assertTrue((path / name).is_file())
                self.assertEqual((path / 'rank-00000.pt').exists(), path == paths[-1])
            self.assertTrue((pending / 'rank-00000.pt').exists())
            self.assertTrue((root / '.step-000013.partial').exists())
            # Persisted policy survives a different default on resume.
            resumed = enforce_checkpoint_retention(paths[-1], 1)
            self.assertEqual(resumed['keep_last'], 8)
            self.assertFalse(resumed['checkpoint_prunes'])

    def test_foreign_checkpoints_and_symlinks_are_not_pruned(self):
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as outside:
            root = Path(temp)
            foreign = self.checkpoint(root, 1, foreign=True)
            old = self.checkpoint(root, 2)
            newest = self.checkpoint(root, 3)
            external = self.checkpoint(Path(outside), 1)
            (root / 'step-000000-link').symlink_to(external, target_is_directory=True)
            self.latest(root, newest)
            result = enforce_checkpoint_retention(newest, 1)
            self.assertFalse(old.exists())
            self.assertTrue((foreign / 'rank-00000.pt').exists())
            self.assertTrue((external / 'rank-00000.pt').exists())
            self.assertEqual(len(result['skipped']), 2)

    def test_older_resume_or_missing_latest_shard_never_authorizes_deletion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old, newest = [self.checkpoint(root, x) for x in (1, 2)]
            self.latest(root, newest)
            with self.assertRaisesRegex(ValueError, 'latest.json'):
                enforce_checkpoint_retention(old, 1)
            (newest / 'rank-00000.pt').unlink()
            with self.assertRaises(ValueError):
                enforce_checkpoint_retention(newest, 1)
            self.assertTrue((old / 'rank-00000.pt').exists())

    def joint_signature(self):
        return {'schema': 'joint-nine-ref-v2-paired-batches', 'world_size': 1,
                'teacher_sha256': 'teacher', 'data': {'csv': 'studio', 'bank': 'verified'},
                'epoch_steps': 168284, 'training': {'save_every_n_steps': 200,
                    'checkpoint_keep_last': 8, 'optimizer': {'lr': 2e-6},
                    'model': {'num_refs': 9}, 'global_batch_size': 160,
                    'critic_updates': 4, 'lr_warmup_steps': 100}}

    def test_mixed_old_1200_and_normalized_1314_share_immutable_run_policy(self):
        with tempfile.TemporaryDirectory() as temp:
            root, original = Path(temp), self.joint_signature()
            plan = root / 'epoch-plan.json'
            plan.write_text(json.dumps({'signature': original}))
            older = self.checkpoint(root, 1000, signature=original)
            old = self.checkpoint(root, 1200, signature=original)
            self.latest(root, old)
            enforce_checkpoint_retention(old, 8)
            plan_bytes, policy_bytes = plan.read_bytes(), (root / 'checkpoint-retention.json').read_bytes()
            current = copy.deepcopy(original)
            current['training']['save_every_n_steps'] = 1000
            newest = self.checkpoint(root, 1314, signature=checkpoint_compatibility(current))
            latest_metadata = (newest / 'metadata.json').read_bytes()
            self.latest(root, newest)
            result = enforce_checkpoint_retention(newest, 8)
            self.assertEqual(set(result['kept']), {str(older), str(old), str(newest)})
            self.assertEqual(len(result['optimizer_prunes']), 1)
            self.assertTrue(all(row['reason'] == 'symlink_or_incomplete'
                                and not Path(row['path']).is_dir() for row in result['skipped']))
            self.assertTrue((newest / 'rank-00000.pt').is_file())
            self.assertFalse((old / 'rank-00000.pt').exists())
            self.assertEqual((newest / 'metadata.json').read_bytes(), latest_metadata)
            self.assertEqual(plan.read_bytes(), plan_bytes)
            self.assertEqual((root / 'checkpoint-retention.json').read_bytes(), policy_bytes)
            self.assertEqual(read_training(old)['save_every_n_steps'], 200)

    def test_joint_training_identity_changes_fail_before_retention_mutates_files(self):
        changes = ((('teacher_sha256',), 'other'), (('world_size',), 2),
                   (('data',), {'csv': 'other'}), (('epoch_steps',), 168285),
                   (('training', 'optimizer', 'lr'), 1e-3),
                   (('training', 'model', 'num_refs'), 8),
                   (('training', 'global_batch_size'), 80),
                   (('training', 'critic_updates'), 1),
                   (('training', 'lr_warmup_steps'), 200))
        for keys, value in changes:
            with self.subTest(keys=keys), tempfile.TemporaryDirectory() as temp:
                root, original = Path(temp), self.joint_signature()
                (root / 'epoch-plan.json').write_text(json.dumps({'signature': original}))
                old = self.checkpoint(root, 1200, signature=original)
                changed = copy.deepcopy(original)
                target = changed
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = value
                newest = self.checkpoint(root, 1314, signature=checkpoint_compatibility(changed))
                self.latest(root, newest)
                with self.assertRaisesRegex(ValueError, 'does not belong to this run'):
                    enforce_checkpoint_retention(newest, 8)
                self.assertTrue((old / 'rank-00000.pt').exists())
                self.assertTrue((newest / 'rank-00000.pt').exists())
                self.assertFalse((root / 'checkpoint-retention.json').exists())

    def test_non_joint_retention_does_not_ignore_another_trainers_cadence(self):
        original = {'schema': 'other-trainer', 'training': {'save_every_n_steps': 200,
                                                         'checkpoint_keep_last': 8}}
        changed = copy.deepcopy(original)
        changed['training']['save_every_n_steps'] = 1000
        self.assertNotEqual(checkpoint_compatibility(original), checkpoint_compatibility(changed))
        changed = copy.deepcopy(original)
        changed['training']['checkpoint_keep_last'] = 4
        self.assertEqual(checkpoint_compatibility(original), checkpoint_compatibility(changed))

    def test_copied_foreign_run_policy_is_rejected_even_for_equal_training_identity(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            source, target = Path(first), Path(second)
            old = self.checkpoint(source, 1200, signature=self.joint_signature())
            self.latest(source, old)
            enforce_checkpoint_retention(old, 8)
            newest = self.checkpoint(target, 1314, signature=self.joint_signature())
            self.latest(target, newest)
            (target / 'checkpoint-retention.json').write_bytes((source / 'checkpoint-retention.json').read_bytes())
            with self.assertRaisesRegex(ValueError, 'policy identity differs'):
                enforce_checkpoint_retention(newest, 8)
            self.assertTrue((newest / 'rank-00000.pt').exists())

    def test_newer_completed_generation_prevents_all_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = [self.checkpoint(root, x) for x in (1, 2, 3)]
            self.latest(root, paths[1])
            with self.assertRaisesRegex(ValueError, 'newer completed'):
                enforce_checkpoint_retention(paths[1], 1)
            self.assertTrue(all((p / 'rank-00000.pt').exists() for p in paths))
            self.assertFalse((root / 'checkpoint-retention.json').exists())

    def test_quarantine_cleanup_recovers_after_interruption(self):
        import distillation.checkpoint_retention as retention
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old, newest = [self.checkpoint(root, x) for x in (1, 2)]
            self.latest(root, newest)
            with patch.object(retention, '_finish_quarantine', side_effect=OSError('interrupted')):
                with self.assertRaises(OSError):
                    enforce_checkpoint_retention(newest, 1)
            trash, = root.glob('.retired-checkpoint-*')
            (trash / 'student.safetensors').unlink()  # Simulated interrupted cleanup.
            result = enforce_checkpoint_retention(newest, 1)
            self.assertEqual(len(result['checkpoint_prunes']), 1)
            self.assertFalse(trash.exists())
            self.assertTrue((newest / 'rank-00000.pt').exists())

    def test_changed_latest_during_retention_prevents_old_checkpoint_delete(self):
        import distillation.checkpoints as checkpoints
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old, newest = [self.checkpoint(root, x) for x in (1, 2)]
            self.latest(root, newest)
            real_write = checkpoints.write_json_atomic
            def mutate(path, value):
                real_write(path, value)
                if path.name == 'checkpoint-retention.json':
                    (root / 'latest.json').write_text('{}')
            with patch.object(checkpoints, 'write_json_atomic', side_effect=mutate):
                with self.assertRaisesRegex(ValueError, 'changed during retention'):
                    enforce_checkpoint_retention(newest, 1)
            self.assertTrue((old / 'student.safetensors').exists())
            self.assertTrue((newest / 'rank-00000.pt').exists())

    def interrupted_retirement(self, root, *, anchored):
        import distillation.checkpoint_retention as retention
        old, newest = [self.checkpoint(root, x) for x in (1, 2)]
        self.latest(root, newest)
        if anchored:
            real_unlink = Path.unlink

            def interrupt(file, *args, **kwargs):
                if file.parent.name.startswith('.retired-checkpoint-'):
                    raise OSError('interrupted after durable identity anchor')
                return real_unlink(file, *args, **kwargs)

            with patch.object(Path, 'unlink', interrupt):
                with self.assertRaises(OSError):
                    enforce_checkpoint_retention(newest, 1)
        else:
            with patch.object(retention, '_finish_quarantine', side_effect=OSError('interrupted')):
                with self.assertRaises(OSError):
                    enforce_checkpoint_retention(newest, 1)
        trash, = root.glob('.retired-checkpoint-*')
        receipt, = (root / 'checkpoint-retention-history').glob('*.json')
        self.assertEqual((trash / '.retirement-identity.json').exists(), anchored)
        return newest, trash, receipt

    def test_changed_generation_blocks_quarantine_replay_before_and_after_anchor(self):
        for anchored in (False, True):
            for name in ('metadata.json', 'checkpoint.complete.json'):
                with self.subTest(anchored=anchored, identity_file=name), tempfile.TemporaryDirectory() as temp:
                    newest, trash, receipt = self.interrupted_retirement(Path(temp), anchored=anchored)
                    identity_file = trash / name
                    original_size = identity_file.stat().st_size
                    value = json.loads(identity_file.read_text())
                    value['checkpoint_id'] = '0' * 32
                    identity_file.write_text(json.dumps(value, indent=2) + '\n')
                    self.assertEqual(identity_file.stat().st_size, original_size)
                    with self.assertRaisesRegex(ValueError, 'generation differs'):
                        enforce_checkpoint_retention(newest, 1)
                    self.assertTrue((trash / 'student.safetensors').exists())
                    self.assertTrue((newest / 'rank-00000.pt').exists())
                    self.assertEqual(json.loads(receipt.read_text())['status'], 'planned')

    def test_anchor_allows_recovery_after_identity_file_was_deleted(self):
        with tempfile.TemporaryDirectory() as temp:
            newest, trash, _ = self.interrupted_retirement(Path(temp), anchored=True)
            (trash / 'metadata.json').unlink()
            result = enforce_checkpoint_retention(newest, 1)
            self.assertEqual(len(result['checkpoint_prunes']), 1)
            self.assertFalse(trash.exists())
            self.assertTrue((newest / 'rank-00000.pt').exists())

    def test_receipt_token_cannot_be_replayed_for_another_quarantine(self):
        with tempfile.TemporaryDirectory() as temp:
            newest, trash, receipt = self.interrupted_retirement(Path(temp), anchored=False)
            record = json.loads(receipt.read_text())
            record['retirement_id'] = '0' * 32
            receipt.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'receipt identity differs'):
                enforce_checkpoint_retention(newest, 1)
            self.assertTrue((trash / 'student.safetensors').exists())
            self.assertTrue((newest / 'rank-00000.pt').exists())


def read_training(path):
    return json.loads((path / 'metadata.json').read_text())['compatibility']['training']


if __name__ == '__main__':
    unittest.main()
