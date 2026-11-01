import tempfile
import unittest
import hashlib
import json
from pathlib import Path

from distillation.joint_runtime import (acknowledge_save_request, claim_save_request,
    prune_previous_checkpoints, training_indices, validate_reference_bank)


class JointRuntimeTests(unittest.TestCase):
    def make_reference_bank(self, directory):
        latent = directory / 'bank/actor/property_1/latents.safetensors'
        latent.parent.mkdir(parents=True)
        latent.write_bytes(b'validated reference tensor')
        (directory / 'bank/bank.csv').write_text('actor,property\nactor,1\n')
        (directory / 'bank/SHA256SUMS').write_text('fixture nested manifest\n')
        selection = directory / 'selection.parquet'
        selection.write_bytes(b'validated selection')
        digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        report = {'complete': False, 'artifact_valid': True, 'available_work_complete': True,
                  'source_complete': False, 'sources': {'errors': []}, 'latents': {'errors': []},
                  'selection_sha256': digest(selection), 'vae_sha256': 'pinned-vae',
                  'encoder_fingerprint': {'sha256': 'pinned-implementation'}}
        (directory / 'coverage.json').write_text(json.dumps(report))
        manifest = ''.join(f'{digest(path)}  {path.relative_to(directory)}\n'
                           for path in sorted(directory.rglob('*')) if path.is_file())
        (directory / 'SHA256SUMS').write_text(manifest)
        marker = {'schema_version': 1, 'artifact_valid': True, 'source_complete': False,
                  'available_work_complete': True, 'selection_sha256': digest(selection),
                  'vae_sha256': 'pinned-vae', 'encoder_fingerprint_sha256': 'pinned-implementation',
                  'coverage_sha256': digest(directory / 'coverage.json'),
                  'manifest_sha256': digest(directory / 'SHA256SUMS')}
        (directory / 'artifact-validation.json').write_text(json.dumps(marker))
        return latent

    def test_bank_identity_requires_final_marker_and_detects_changed_tensor(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            latent = self.make_reference_bank(directory)
            identity = validate_reference_bank(directory)
            self.assertFalse(identity['source_complete'])  # Honest source gaps do not mean corrupt latents.
            self.assertEqual(identity['verified_files'], 5)
            latent.write_bytes(b'replacement at identical path')
            with self.assertRaisesRegex(ValueError, 'checksum differs'):
                validate_reference_bank(directory)
            (directory / 'artifact-validation.json').unlink()
            with self.assertRaises(FileNotFoundError):
                validate_reference_bank(directory)

    def test_bank_gate_rejects_partial_work_and_unmanifested_latents(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            latent = self.make_reference_bank(directory)
            extra = latent.with_name('unverified.safetensors')
            extra.write_bytes(b'unverified')
            with self.assertRaisesRegex(ValueError, 'latent files differ'):
                validate_reference_bank(directory)
            extra.unlink()
            marker_path = directory / 'artifact-validation.json'
            marker = json.loads(marker_path.read_text())
            marker['available_work_complete'] = False
            marker_path.write_text(json.dumps(marker))
            with self.assertRaisesRegex(ValueError, 'not completed artifact validation'):
                validate_reference_bank(directory)

    def test_touch_save_claim_survives_failure_and_preserves_new_request(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / 'save').touch()
            request = claim_save_request(directory)
            self.assertEqual(request, {'save': True, 'quit': False, 'markers': ['.save.pending']})
            self.assertFalse((directory / 'save').exists())
            self.assertEqual(claim_save_request(directory), request)
            (directory / 'save').touch()  # Arrives while the previous save is running.
            acknowledge_save_request(directory, request)
            self.assertTrue((directory / 'save').exists())
            self.assertTrue(claim_save_request(directory)['save'])

    def test_save_q_takes_priority_and_acknowledges_both_requests(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / 'save').touch()
            (directory / 'save_q').touch()
            request = claim_save_request(directory)
            self.assertTrue(request['save'])
            self.assertTrue(request['quit'])
            acknowledge_save_request(directory, request)
            self.assertFalse(claim_save_request(directory)['save'])

    def test_resume_plan_preserves_one_full_epoch_and_separate_warmup(self):
        warm, joint = training_indices(3, {'warmup_step': 2, 'epoch_step': 0}, 5)
        self.assertEqual(warm, [2, 0, 1])
        self.assertEqual(list(joint), [0, 1, 2])
        warm, joint = training_indices(3, {'warmup_step': 5, 'epoch_step': 1}, 5)
        self.assertEqual(warm, [])
        self.assertEqual(list(joint), [1, 2])

    def test_benchmark_always_exercises_full_joint_rounds(self):
        warm, joint = training_indices(3, {'warmup_step': 0, 'epoch_step': 0}, 150, 5)
        self.assertEqual(warm, [])
        self.assertEqual(joint, [0, 1, 2, 0, 1])

    def make_checkpoint(self, parent, step, complete=True):
        from distillation.checkpoints import complete_checkpoint
        directory = parent / f'step-{step:06d}'
        directory.mkdir()
        (directory / 'student.safetensors').write_bytes(b'student weights')
        (directory / 'critic.safetensors').write_bytes(b'critic weights')
        (directory / 'rank-00000.pt').write_bytes(b'optimizer state')
        if complete:
            complete_checkpoint(directory, {'world_size': 1, 'optimizer_state_retained': True,
                                            'counters': {'rounds': step}})
        return directory

    def test_only_completed_latest_checkpoint_authorizes_optimizer_pruning(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            old = self.make_checkpoint(directory, 200)
            pending = self.make_checkpoint(directory, 400, complete=False)
            with self.assertRaises(FileNotFoundError):
                prune_previous_checkpoints(pending)
            self.assertTrue((old / 'rank-00000.pt').exists())
            latest = self.make_checkpoint(directory, 600)
            receipts = prune_previous_checkpoints(latest)
            self.assertEqual(len(receipts), 1)
            self.assertFalse((old / 'rank-00000.pt').exists())
            self.assertTrue((old / 'student.safetensors').exists())
            self.assertTrue((pending / 'rank-00000.pt').exists())
            self.assertTrue((latest / 'rank-00000.pt').exists())

    def test_resuming_older_generation_cannot_prune_newer_optimizer(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            oldest = self.make_checkpoint(directory, 200)
            older = self.make_checkpoint(directory, 400)
            newest = self.make_checkpoint(directory, 600)
            with self.assertRaisesRegex(ValueError, 'newer checkpoint'):
                prune_previous_checkpoints(older)
            for checkpoint in (oldest, older, newest):
                self.assertTrue((checkpoint / 'rank-00000.pt').exists())


if __name__ == '__main__':
    unittest.main()
