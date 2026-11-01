import json
from pathlib import Path
import shutil
import tempfile
import unittest
import uuid

from distillation.archive_checkpoint import inspect_checkpoint, retire_source, scan_pruned, snapshot_checkpoint, verify_copy, verify_source, worker


class ArchiveCheckpointTests(unittest.TestCase):
    def make_checkpoint(self, parent, name='critic-warmup', retained=True, student_updates=0):
        path = Path(parent) / name
        path.mkdir()
        checkpoint_id = uuid.uuid4().hex
        meta = {'checkpoint_id': checkpoint_id, 'critic_warmup_complete': True,
                'counters': {'student_updates': student_updates}, 'optimizer_state_retained': retained, 'world_size': 1}
        (path / 'metadata.json').write_text(json.dumps(meta))
        (path / 'student.safetensors').write_bytes(b'student weights')
        (path / 'critic.safetensors').write_bytes(b'critic weights')
        if retained:
            (path / 'rank-00000.pt').write_bytes(b'full optimizer state')
        marker = {'checkpoint_id': checkpoint_id,
                  'files': [{'path': file.name, 'size': file.stat().st_size} for file in sorted(path.iterdir())]}
        (path / 'checkpoint.complete.json').write_text(json.dumps(marker))
        return path, checkpoint_id

    def test_verified_copy_publish_and_retirement(self):
        with tempfile.TemporaryDirectory(prefix='checkpoint-archive-') as root:
            source, checkpoint_id = self.make_checkpoint(root)
            snapshot = snapshot_checkpoint(source, checkpoint_id)
            destination = Path(root) / 'archive' / 'critic-warmup'
            temporary = Path(worker('prepare', {'path': str(destination)})['temporary'])
            shutil.copytree(source, temporary, dirs_exist_ok=True)
            verify_copy(temporary, snapshot)
            verify_source(snapshot)
            receipt = {'checkpoint_id': checkpoint_id, 'destination': str(destination)}
            worker('publish', {'path': str(destination), 'temporary': str(temporary), 'receipt': receipt})
            retire_source(snapshot, receipt)
            self.assertFalse(source.exists())
            self.assertFalse(temporary.exists())
            self.assertTrue(destination.is_dir())
            self.assertEqual(json.loads(source.with_name(source.name + '.archive.json').read_text()), receipt)
            verify_copy(destination, snapshot)

    def test_corrupt_copy_or_changed_source_cannot_be_accepted(self):
        with tempfile.TemporaryDirectory(prefix='checkpoint-archive-') as root:
            source, checkpoint_id = self.make_checkpoint(root)
            snapshot = snapshot_checkpoint(source, checkpoint_id)
            destination = Path(root) / 'copy'
            shutil.copytree(source, destination)
            (destination / 'critic.safetensors').write_bytes(b'CRITIC weights')
            with self.assertRaises(ValueError):
                verify_copy(destination, snapshot)
            (source / 'critic.safetensors').write_bytes(b'CRITIC weights')
            with self.assertRaises(ValueError):
                retire_source(snapshot, {})
            self.assertTrue(source.exists())

    def test_incomplete_and_wrong_generation_rejected(self):
        with tempfile.TemporaryDirectory(prefix='checkpoint-archive-') as root:
            source, checkpoint_id = self.make_checkpoint(root)
            with self.assertRaises(ValueError):
                snapshot_checkpoint(source, 'wrong-generation')
            (source / 'partial.tmp').write_bytes(b'pending writer')
            with self.assertRaises(ValueError):
                snapshot_checkpoint(source, checkpoint_id)

    def test_move_restricted_to_immutable_snapshots(self):
        with tempfile.TemporaryDirectory(prefix='checkpoint-archive-') as root:
            source, checkpoint_id = self.make_checkpoint(root, 'step-000350', student_updates=200)
            active = snapshot_checkpoint(source, checkpoint_id)
            self.assertFalse(active['move_eligible'])
            with self.assertRaises(ValueError):
                retire_source(active, {})
            pruned, checkpoint_id = self.make_checkpoint(root, 'step-000550', retained=False, student_updates=400)
            self.assertTrue(inspect_checkpoint(pruned, checkpoint_id)['move_eligible'])

    def test_monitor_requires_completed_newer_full_checkpoint(self):
        with tempfile.TemporaryDirectory(prefix='checkpoint-archive-') as root:
            old, _ = self.make_checkpoint(root, 'step-000350', retained=False, student_updates=200)
            self.assertFalse(scan_pruned(root)['candidates'])
            latest, _ = self.make_checkpoint(root, 'step-000550', student_updates=400)
            status = scan_pruned(root)
            self.assertEqual([item['path'] for item in status['candidates']], [str(old)])
            self.assertEqual(status['latest_full']['path'], str(latest))
            (latest / 'checkpoint.complete.json').unlink()
            status = scan_pruned(root)
            self.assertFalse(status['candidates'])
            self.assertEqual(status['pending'], ['step-000550'])

    def test_monitor_waits_for_old_optimizer_pruning(self):
        with tempfile.TemporaryDirectory(prefix='checkpoint-archive-') as root:
            old, _ = self.make_checkpoint(root, 'step-000350', student_updates=200)
            latest, _ = self.make_checkpoint(root, 'step-001200', student_updates=1050)
            status = scan_pruned(root)
            self.assertFalse(status['candidates'])
            self.assertEqual(status['unpruned_older'], [str(old)])
            self.assertEqual(status['latest_full']['path'], str(latest))


if __name__ == '__main__':
    unittest.main()
