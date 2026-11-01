"""Bounded CPU checks for the read-only post-resume audit tool."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from distillation.joint_metrics import TRAINING_METRIC_ALIASES, TRAINING_METRIC_MAP
from scripts.audit_joint_online import (NotReady, compare_scalars, validate_training_rows,
    verified_startup_gate, verify_full_reports)


class AuditJointOnlineTests(unittest.TestCase):
    def row(self, number):
        value = {source: 1. for source in TRAINING_METRIC_MAP.values()}
        value.update(rounds=number, benchmark=False, phase='joint')
        value.update({'phase/joint': 1, 'phase/critic_warmup': 0})
        value.update({alias: value[source] for alias, source in TRAINING_METRIC_ALIASES.items()})
        return value

    def test_requires_five_contiguous_full_contract_rows_and_checks_semantics(self):
        rows = {number: self.row(number) for number in range(1201, 1206)}
        result = validate_training_rows(rows, 1201, 5)
        self.assertEqual(len(result), 5)
        self.assertTrue(all(len(row) == 98 for row in result.values()))
        with self.assertRaises(NotReady):
            validate_training_rows({1201: rows[1201]}, 1201, 5)
        invalid = copy.deepcopy(rows)
        invalid[1201]['student/lr_next'] = 0.
        with self.assertRaisesRegex(ValueError, 'learning rates'):
            validate_training_rows(invalid, 1201, 5)
        invalid = copy.deepcopy(rows)
        invalid[1201]['diagnostic/resolver/teacher_critic_velocity_cosine'] = 2.
        with self.assertRaisesRegex(ValueError, 'cosine'):
            validate_training_rows(invalid, 1201, 5)
        invalid = copy.deepcopy(rows)
        invalid[1201].pop('dmd_resolver/normalizer')
        with self.assertRaisesRegex(ValueError, 'full98-field'):
            validate_training_rows(invalid, 1201, 5)

    def test_scalar_comparison_rejects_missing_nonfinite_and_changed_values(self):
        compare_scalars({'x': 1.}, {'x': 1.0000001}, 'fixture')
        for actual in ({}, {'x': float('nan')}, {'x': 2.}):
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                compare_scalars({'x': 1.}, actual, 'fixture')

    def recovery(self):
        counters = {'rounds': 1314, 'student_updates': 1164, 'critic_updates': 5256}
        initial = {'event': 'wandb_online_verified', 'online_metrics_verified': True,
                   'verified_round': 1314, 'verified_student_updates': 1164, 'publish_token': 'initial',
                   'verified_sensitivity_metrics': 0, 'sensitivity_report_uploads_verified': False,
                   'sensitivity_report_files': []}
        final = {**initial, 'publish_token': 'recovered', 'verified_sensitivity_metrics': 22,
                 'sensitivity_report_uploads_verified': True,
                 'sensitivity_report_files': [{'name': 'sensitivity-raw.json', 'sha256': 'digest'}]}
        final.pop('event')
        event = {'event': 'checkpoint_evaluation_recovered', 'path': '/run/checkpoint',
                 'counters': counters, 'checkpoint_id': 'identity', 'tracking': final}
        return counters, initial, final, event

    def test_recovered_evaluation_is_the_final_gate_before_training(self):
        counters, initial, final, event = self.recovery()
        parsed = [(10, initial), (20, event), (30, {'rounds': 1315})]
        result = verified_startup_gate(parsed, Path('/run/checkpoint'), counters, 'identity', final)
        self.assertEqual(result, (10, 20, final, True))
        with self.assertRaises(NotReady):
            verified_startup_gate(parsed[:1], Path('/run/checkpoint'), counters, 'identity', initial)
        with self.assertRaisesRegex(ValueError, 'lacks verified sensitivity'):
            verified_startup_gate([parsed[0], parsed[2]], Path('/run/checkpoint'), counters, 'identity', initial)
        with self.assertRaisesRegex(ValueError, 'preceded the final'):
            verified_startup_gate([parsed[0], (15, {'rounds': 1315}), parsed[1]],
                                  Path('/run/checkpoint'), counters, 'identity', final)
        # A normal resume with an already evaluated checkpoint uses its initial gate.
        normal = {'event': 'wandb_online_verified', **final}
        self.assertEqual(verified_startup_gate([(10, normal), (30, {'rounds': 1315})],
            Path('/run/checkpoint'), counters, 'identity', final), (10, 10, final | {'event': 'wandb_online_verified'}, False))

    def test_recovered_gate_rejects_foreign_identity_stale_receipts_and_unverified_files(self):
        counters, initial, final, event = self.recovery()
        mutations = [('path', '/foreign/checkpoint'), ('checkpoint_id', 'foreign'),
                     ('counters', {**counters, 'rounds': 1313}),
                     ('tracking', {**final, 'verified_round': 1313}),
                     ('tracking', {**final, 'sensitivity_report_uploads_verified': False}),
                     ('tracking', {**final, 'publish_token': 'stale'})]
        for key, value in mutations:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                verified_startup_gate([(10, initial), (20, {**event, key: value})],
                                      Path('/run/checkpoint'), counters, 'identity', final)

    def test_report_download_sha_is_compared_to_actual_checkpoint_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint = root / 'checkpoint'
            checkpoint.mkdir()
            payload = b'{"format":"fixture","value":12}'
            (checkpoint / 'sensitivity-raw.json').write_bytes(payload)
            (checkpoint / 'metadata.json').write_text(json.dumps({'counters': {'rounds': 1200}}))
            (checkpoint / 'evaluation.complete.json').write_text(json.dumps({'checkpoint_id': 'identity',
                'reports': [{'weights': 'raw', 'file': 'sensitivity-raw.json'}]}))
            staged = root / 'sensitivity_reports/identity'
            staged.mkdir(parents=True)
            definition = {'checkpoint_id': 'identity', 'checkpoint_round': 1200,
                          'reports': [{'file': 'sensitivity-raw.json',
                                       'sha256': hashlib.sha256(payload).hexdigest()}]}
            definition_path = staged / 'sensitivity-overview-definition.json'
            definition_path.write_text(json.dumps(definition))
            records, remote_files = [], {}
            for name, contents in [('sensitivity-raw.json', payload),
                                   ('sensitivity-overview-definition.json', definition_path.read_bytes())]:
                full_name = f'sensitivity_reports/identity/{name}'
                records.append({'name': full_name, 'bytes': len(contents),
                                'sha256': hashlib.sha256(contents).hexdigest(), 'checkpoint_id': 'identity'})
                remote_files[full_name] = FakeFile(full_name, contents)
            remote = FakeRemote(remote_files)
            verified = verify_full_reports(remote, None, records, checkpoint, root)
            self.assertEqual(len(verified), 2)
            self.assertTrue(all(row['remote_sha256_verified'] for row in verified))
            raw = remote_files['sensitivity_reports/identity/sensitivity-raw.json']
            raw.contents = payload.replace(b'12', b'13')  # Same size, changed remote bytes.
            with self.assertRaisesRegex(ValueError, 'Downloaded remote report differs'):
                verify_full_reports(remote, None, records, checkpoint, root)


class FakeFile:
    def __init__(self, name, contents):
        self.name, self.contents, self.size = name, contents, len(contents)

    def download(self, root, **kwargs):
        path = Path(root) / self.name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.contents)
        return path.open('rb')


class FakeRemote:
    def __init__(self, files):
        self.entries = files

    def files(self, names):
        return [self.entries[name] for name in names if name in self.entries]


if __name__ == '__main__':
    unittest.main()
