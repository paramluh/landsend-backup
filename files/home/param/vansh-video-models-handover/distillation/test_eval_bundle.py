import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from distillation.archive_checkpoint import digest
from distillation.compare import snapshot
from distillation.eval_bundle import MARKER, SOURCE_MARKER, create_bundle, read_bundle
from distillation import test_archive_checkpoint


class EvaluationBundleTests(unittest.TestCase):
    def test_bundle_copies_only_student_and_preserves_source_identity(self):
        with tempfile.TemporaryDirectory() as root:
            source, checkpoint_id = test_archive_checkpoint.ArchiveCheckpointTests().make_checkpoint(root, 'step-000350', student_updates=200)
            original = {path.name: digest(path) for path in source.iterdir()}
            destination = Path(root) / 'evaluation'
            real_open = Path.open

            def guarded_open(path, *args, **kwargs):
                if path.name.startswith('rank-') or path.name == 'critic.safetensors':
                    raise AssertionError('Evaluation bundling must not read optimizer or critic contents')
                return real_open(path, *args, **kwargs)

            with patch.object(Path, 'open', guarded_open):
                receipt = create_bundle(source, destination, checkpoint_id)
                student, meta = snapshot(destination)
            self.assertEqual({path.name for path in destination.iterdir()},
                             {'student.safetensors', 'metadata.json', SOURCE_MARKER, MARKER})
            self.assertEqual(student.read_bytes(), (source / 'student.safetensors').read_bytes())
            self.assertEqual(meta['checkpoint_id'], checkpoint_id)
            self.assertTrue(meta['optimizer_state_retained'])  # Original metadata is not rewritten.
            self.assertFalse(receipt['training_resume_supported'])
            self.assertEqual(receipt['source']['directory'], str(source))
            self.assertEqual(original, {path.name: digest(path) for path in source.iterdir()})
            with self.assertRaises(FileExistsError):
                create_bundle(source, destination, checkpoint_id)
            from distillation.checkpoints import read_checkpoint_metadata
            with self.assertRaises(FileNotFoundError):
                read_checkpoint_metadata(destination)

    def test_bundle_rejects_same_size_corruption_fake_full_marker_and_changed_source(self):
        with tempfile.TemporaryDirectory() as root:
            source, checkpoint_id = test_archive_checkpoint.ArchiveCheckpointTests().make_checkpoint(root)
            destination = Path(root) / 'evaluation'
            create_bundle(source, destination, checkpoint_id)
            student = destination / 'student.safetensors'
            student.write_bytes(b'STUDENT weights')
            with self.assertRaisesRegex(ValueError, 'content differs'):
                read_bundle(destination)
            student.write_bytes((source / 'student.safetensors').read_bytes())
            (destination / 'checkpoint.complete.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'unexpected'):
                snapshot(destination)
            second = Path(root) / 'other'
            with patch('distillation.eval_bundle.verify_source', side_effect=ValueError('source changed')):
                with self.assertRaisesRegex(ValueError, 'source changed'):
                    create_bundle(source, second, checkpoint_id)
            self.assertFalse(second.exists())
            self.assertFalse(list(Path(root).glob('.other.partial-*')))

    def test_full_reader_rejects_missing_optimizer_and_bundle_requires_original_generation(self):
        with tempfile.TemporaryDirectory() as root:
            source, checkpoint_id = test_archive_checkpoint.ArchiveCheckpointTests().make_checkpoint(root)
            with self.assertRaisesRegex(ValueError, 'generation differs'):
                create_bundle(source, Path(root) / 'evaluation', 'wrong')
            (source / 'rank-00000.pt').unlink()
            with self.assertRaises(ValueError):
                snapshot(source)


if __name__ == '__main__':
    unittest.main()
