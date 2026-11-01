"""CPU checks for the one explicitly authorized full-state guidance transition."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from distillation.checkpoint_retention import checkpoint_compatibility, enforce_checkpoint_retention
from distillation.checkpoints import complete_checkpoint, read_checkpoint_metadata
from distillation.resume_transition import (TRANSITIONS_FILE, accepted_run_signatures,
    record_resume_transition, validate_resume_compatibility)


def signature(guidance=1.0):
    return {'schema': 'full-gt-nine-ref-v1-single-target', 'world_size': 1,
            'data': {'csv_sha256': 'studio', 'bank': 'strict'}, 'teacher_sha256': 'teacher',
            'latent_hw': [30, 16], 'epoch_steps': 8000,
            'training': {'training_method': 'full_gt_single_target',
                         'teacher_guidance_scale': guidance, 'global_batch_size': 160,
                         'critic_updates': 4, 'critic_warmup_steps': 150,
                         'optimizer': {'lr': 2e-6}, 'ema_decay': 0.99,
                         'lr_warmup_steps': 100, 'critic_lr_warmup_steps': 200,
                         'student_steps': 8000, 'seed': 42}}


class GuidanceResumeTests(unittest.TestCase):
    def checkpoint(self, root, step, identity):
        path = root / f'step-{step:06d}-fixture'
        path.mkdir()
        for name in ('student.safetensors', 'critic.safetensors', 'model_ema.safetensors', 'rank-00000.pt'):
            (path / name).write_bytes((name + str(step)).encode())
        complete_checkpoint(path, {'world_size': 1, 'optimizer_state_retained': True,
            'counters': {'rounds': step}, 'compatibility': identity})
        return path

    def latest(self, root, checkpoint):
        metadata = read_checkpoint_metadata(checkpoint)
        (root / 'latest.json').write_text(json.dumps({'checkpoint': str(checkpoint),
                                                    'counters': metadata['counters']}))

    def plan(self, root):
        (root / 'epoch-plan.json').write_text(json.dumps({'signature': signature(),
                                                       'totals': {'student': 8000, 'critic': 32600}}))

    def enroll(self, source, requested=None, expected=None):
        return record_resume_transition(source, requested or signature(2.5),
            expected_checkpoint_id=expected or read_checkpoint_metadata(source)['checkpoint_id'],
            allow_teacher_guidance_change=True)

    def test_guidance_is_strict_by_default_and_opt_in_is_not_training_identity(self):
        old, new = signature(), signature(2.5)
        old_before, new_before = copy.deepcopy(old), copy.deepcopy(new)
        self.assertNotEqual(checkpoint_compatibility(old), checkpoint_compatibility(new))
        with self.assertRaisesRegex(ValueError, 'differs'):
            validate_resume_compatibility(old, new)
        new['training']['resume_allow_teacher_guidance_change'] = True
        self.assertEqual(validate_resume_compatibility(old, new, allow_teacher_guidance_change=True),
                         {'field': 'training.teacher_guidance_scale', 'from': 1.0, 'to': 2.5})
        self.assertEqual(old, old_before)
        self.assertEqual(checkpoint_compatibility(new), new_before)
        self.assertIsNone(validate_resume_compatibility(new, new_before))
        with self.assertRaisesRegex(ValueError, 'boolean'):
            validate_resume_compatibility(old, new, allow_teacher_guidance_change=1)

    def test_permission_cannot_change_any_other_training_or_run_identity(self):
        changes = [(('world_size',), 40), (('teacher_sha256',), 'other'),
                   (('data', 'csv_sha256'), 'other'), (('data', 'bank'), 'other'),
                   (('latent_hw',), [60, 32]), (('epoch_steps',), 9000),
                   (('training', 'optimizer', 'lr'), 1e-5),
                   (('training', 'global_batch_size'), 80),
                   (('training', 'critic_updates'), 1), (('training', 'ema_decay'), 0.9),
                   (('training', 'lr_warmup_steps'), 0),
                   (('training', 'critic_lr_warmup_steps'), 0),
                   (('training', 'critic_warmup_steps'), 0),
                   (('training', 'student_steps'), 9000), (('training', 'seed'), 43),
                   (('training', 'training_method'), 'other')]
        for keys, value in changes:
            with self.subTest(keys=keys):
                changed = signature(2.5)
                target = changed
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = value
                with self.assertRaisesRegex(ValueError, 'differs'):
                    validate_resume_compatibility(signature(), changed, allow_teacher_guidance_change=True)

    def test_only_full_gt_one_to_two_point_five_is_authorized(self):
        for before, after in ((2.5, 1.0), (1.0, 3.0), (1.5, 2.5), (True, 2.5)):
            with self.subTest(before=before, after=after), self.assertRaisesRegex(ValueError, 'differs'):
                validate_resume_compatibility(signature(before), signature(after),
                                              allow_teacher_guidance_change=True)
        for schema in ('joint-nine-ref-v2-paired-batches', 'other'):
            old, new = signature(), signature(2.5)
            old['schema'] = new['schema'] = schema
            with self.assertRaisesRegex(ValueError, 'differs'):
                validate_resume_compatibility(old, new, allow_teacher_guidance_change=True)

    def test_receipt_is_idempotent_and_preserves_saved_state_and_epoch_plan(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.plan(root)
            source = self.checkpoint(root, 1800, signature())
            self.latest(root, source)
            before = {str(p): p.read_bytes() for p in source.iterdir()}
            plan_bytes = (root / 'epoch-plan.json').read_bytes()
            receipt = self.enroll(source)
            ledger_bytes = (root / TRANSITIONS_FILE).read_bytes()
            self.assertEqual(self.enroll(source), receipt)
            self.assertEqual((root / TRANSITIONS_FILE).read_bytes(), ledger_bytes)
            self.assertEqual({str(p): p.read_bytes() for p in source.iterdir()}, before)
            self.assertEqual((root / 'epoch-plan.json').read_bytes(), plan_bytes)
            self.assertEqual(receipt['source_metadata_sha256'],
                             hashlib.sha256(before[str(source / 'metadata.json')]).hexdigest())
            self.assertEqual(receipt['source_manifest_sha256'],
                             hashlib.sha256(before[str(source / 'checkpoint.complete.json')]).hexdigest())
            self.assertEqual(accepted_run_signatures(root, signature()), [signature(), signature(2.5)])

    def test_enrollment_requires_latest_complete_source_and_expected_generation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.plan(root)
            source = self.checkpoint(root, 1800, signature())
            newer = self.checkpoint(root, 1801, signature())
            self.latest(root, newer)
            with self.assertRaisesRegex(ValueError, 'latest full-state'):
                self.enroll(source)
            with self.assertRaisesRegex(ValueError, 'generation changed'):
                self.enroll(newer, expected='wrong-generation')
            (newer / 'rank-00000.pt').unlink()
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                self.enroll(newer, expected='wrong-generation')
            self.assertFalse((root / TRANSITIONS_FILE).exists())

    def test_retention_spans_enrolled_identities_without_changing_plan_or_policy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.plan(root)
            old = self.checkpoint(root, 1600, signature())
            source = self.checkpoint(root, 1800, signature())
            self.latest(root, source)
            enforce_checkpoint_retention(source, 2)
            preserved = {name: (root / name).read_bytes()
                         for name in ('epoch-plan.json', 'checkpoint-retention.json')}
            self.enroll(source)
            foreign_signature = signature(2.5)
            foreign_signature['data']['csv_sha256'] = 'foreign'
            foreign = self.checkpoint(root, 1700, foreign_signature)
            new = self.checkpoint(root, 2400, signature(2.5))
            self.latest(root, new)
            result = enforce_checkpoint_retention(new, 8)
            self.assertEqual(result['keep_last'], 2)
            self.assertEqual(set(result['kept']), {str(source), str(new)})
            self.assertFalse(old.exists())
            self.assertFalse((source / 'rank-00000.pt').exists())
            self.assertTrue((new / 'rank-00000.pt').is_file())
            self.assertTrue((foreign / 'rank-00000.pt').is_file())
            self.assertIn({'path': str(foreign), 'reason': 'foreign_identity'}, result['skipped'])
            self.assertEqual({name: (root / name).read_bytes() for name in preserved}, preserved)
            # The enrollment remains valid after normal pruning changes the
            # source metadata and eventually deletes the source directory.
            newest = self.checkpoint(root, 3200, signature(2.5))
            self.latest(root, newest)
            enforce_checkpoint_retention(newest, 2)
            self.assertFalse(source.exists())
            self.assertEqual(accepted_run_signatures(root, signature()), [signature(), signature(2.5)])

    def test_no_receipt_does_not_admit_changed_guidance_to_run(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.plan(root)
            old = self.checkpoint(root, 1800, signature())
            new = self.checkpoint(root, 2400, signature(2.5))
            self.latest(root, new)
            with self.assertRaisesRegex(ValueError, 'does not belong'):
                enforce_checkpoint_retention(new, 1)
            self.assertTrue((old / 'rank-00000.pt').is_file())

    def test_tampered_receipt_and_rehashed_foreign_transition_fail_closed(self):
        for rehash in (False, True):
            with self.subTest(rehash=rehash), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                self.plan(root)
                source = self.checkpoint(root, 1800, signature())
                self.latest(root, source)
                self.enroll(source)
                path = root / TRANSITIONS_FILE
                ledger = json.loads(path.read_text())
                entry = ledger['transitions'][0]
                entry['target_signature']['data']['csv_sha256'] = 'foreign'
                if rehash:
                    payload = {k: v for k, v in entry.items() if k != 'transition_id'}
                    entry['transition_id'] = hashlib.sha256(json.dumps(payload, sort_keys=True,
                        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
                path.write_text(json.dumps(ledger))
                with self.assertRaisesRegex(ValueError, 'differs'):
                    enforce_checkpoint_retention(source, 1)
                self.assertTrue((source / 'rank-00000.pt').is_file())
                self.assertFalse((root / 'checkpoint-retention.json').exists())

    def test_copied_run_receipt_and_changed_original_plan_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as other:
            root, foreign_root = Path(temp), Path(other)
            self.plan(root)
            self.plan(foreign_root)
            source = self.checkpoint(root, 1800, signature())
            self.latest(root, source)
            self.enroll(source)
            (foreign_root / TRANSITIONS_FILE).write_bytes((root / TRANSITIONS_FILE).read_bytes())
            with self.assertRaisesRegex(ValueError, 'run identity differs'):
                accepted_run_signatures(foreign_root, signature())
            (root / 'epoch-plan.json').write_text(json.dumps({'signature': signature(), 'changed': True}))
            with self.assertRaisesRegex(ValueError, 'run identity differs'):
                enforce_checkpoint_retention(source, 1)
            self.assertTrue((source / 'rank-00000.pt').is_file())


if __name__ == '__main__':
    unittest.main()
