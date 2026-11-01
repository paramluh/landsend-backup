import ast
import copy
import json
import hashlib
from pathlib import Path
import tempfile
import random
from types import SimpleNamespace
import unittest

from distillation.joint_tracking import checkpoint_metrics, sensitivity_metrics, start_online_tracking
from distillation.joint_sensitivity_tracking import (stage_checkpoint_reports,
    upload_checkpoint_reports, verify_report_files)


class FakeRun:
    def __init__(self, sdk, record):
        self.sdk, self.record = sdk, record
        self.id, self.entity, self.project = record['id'], 'canopy-labs', 'param-distillation'
        self.url = f'https://wandb.ai/{self.entity}/{self.project}/runs/{self.id}'
        self.settings = SimpleNamespace(mode=sdk.mode)
        self.offline = sdk.mode != 'online'
        self.summary = record['summary']
        self.calls, self.pending = [], {}
        self.step = max(record['history'], default=-1) + 1

    def define_metric(self, *args, **kwargs):
        pass

    def log(self, payload, *, step, commit):
        if step < self.step:
            raise AssertionError('W&B would discard a decreasing step')
        if step > self.step and self.pending:
            self.record['history'][self.step] = self.pending
            self.pending = {}
        self.step = step
        self.pending.update(payload)
        self.calls.append((copy.deepcopy(payload), step, commit))
        if commit:
            self.record['history'][step] = self.pending
            self.summary.update(self.pending)
            self.pending = {}
            self.step += 1

    def save(self, path, *, base_path, policy, glob):
        assert policy == "now" and glob is False
        name = str(Path(path).relative_to(base_path))
        self.record.setdefault("files", {})[name] = Path(path).read_bytes()
        return [path]

    def finish(self, **kwargs):
        if self.pending:
            self.record['history'][self.step] = self.pending
            self.summary.update(self.pending)
            self.pending = {}
        self.sdk.finishes.append(kwargs)


class FakeAPI:
    def __init__(self, sdk):
        self.sdk = sdk

    def flush(self):
        pass

    def run(self, path):
        record = self.sdk.records[path.rsplit('/', 1)[-1]]
        return FakeRemote(self.sdk, record)


class FakeFile:
    def __init__(self, name, data):
        self.name, self.data, self.size = name, data, len(data)

    def download(self, root, replace, api=None):
        path = Path(root) / self.name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.data)
        return path.open()


class FakeRemote:
    def __init__(self, sdk, record):
        self.sdk, self.record = sdk, record
        self.summary = record['summary'] if sdk.visible else {}
        self.lastHistoryStep = max(record['history'], default=-1)

    def files(self, *, names):
        return [FakeFile(name, data) for name, data in self.record.get('files', {}).items()
                if name in names and self.sdk.files_visible]

    def upload_file(self, path, root):
        self.record.setdefault('files', {})[str(Path(path).relative_to(root))] = Path(path).read_bytes()


class FakeWandb:
    def __init__(self):
        self.records, self.inits, self.finishes = {}, [], []
        self.visible, self.files_visible, self.mode = True, True, 'online'

    def Api(self, **kwargs):
        return FakeAPI(self)

    def init(self, **kwargs):
        self.inits.append(kwargs)
        identity = kwargs['id']
        if kwargs['resume'] == 'must' and identity not in self.records:
            raise ValueError('Cannot resume missing W&B run')
        record = self.records.setdefault(identity, {'id': identity, 'summary': {}, 'history': {}})
        return FakeRun(self, record)


class JointTrackingTests(unittest.TestCase):
    counters = {'rounds': 600, 'student_updates': 450, 'critic_updates': 2400}
    config = {'wandb': {'enabled': True, 'mode': 'online', 'entity': 'canopy-labs',
                         'project': 'param-distillation', 'run_name': 'actual-production'}}

    def fixture(self, root):
        checkpoint = root / 'step-000600-actual'
        checkpoint.mkdir()
        case = {'status': 'ok', 'maximum_amplification': .8, 'last_frame_amplification': .3,
                'frames': [{'target_index': 1, 'amplification': .8, 'nonfinite_rows': 0},
                           {'target_index': 2, 'amplification': .3, 'nonfinite_rows': 0}]}
        report = {'format': 'joint-latent-sensitivity-v1', 'status': 'ok',
                  'measured_target_frames': 2, 'requested_input_rms': .01, 'seeds': [1729],
                  'history_indices_oldest_first': [0, 1, 2], 'reference_indices': [0, 4, 8],
                  'rollouts': [{'seed': 1729, 'cases': {name: copy.deepcopy(case) for name in
                    ('history_0', 'history_1', 'history_2', 'sink', 'reference_0', 'reference_4', 'reference_8')}}]}
        (checkpoint / 'metadata.json').write_text(json.dumps({'checkpoint_id': 'actual', 'counters': self.counters}))
        (checkpoint / 'evaluation.complete.json').write_text(json.dumps({'checkpoint_id': 'actual',
            'reports': [{'weights': 'raw', 'file': 'sensitivity-raw.json'}]}))
        (checkpoint / 'sensitivity-raw.json').write_text(json.dumps(report))
        (root / 'metrics.jsonl').write_text(json.dumps({**self.counters, 'critic/loss': .2, 'student/loss': .1}) + '\n'
            + json.dumps({'rounds': 699, 'student_updates': 549, 'critic/loss': 999}) + '\n')
        return checkpoint, report

    def start(self, root, checkpoint, sdk):
        return start_online_tracking(self.config, root, self.counters, epoch_steps=168284,
            world_size=40, resume_checkpoint=checkpoint, wandb_module=sdk, verify_timeout=0)

    def test_compact_sites_and_nonfinite_status_remain_visible(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint, report = self.fixture(Path(folder))
            values = checkpoint_metrics(checkpoint)
            self.assertEqual(len(values), 22)
            self.assertEqual(values['sensitivity_overview/checkpoint_round'], 600)
            self.assertEqual(values['sensitivity_history/raw_history_0_max_amplification'], .8)
            self.assertEqual(values['sensitivity_history/raw_history_0_last_frame_amplification'], .3)
            report['status'] = 'nonfinite_output'
            report['rollouts'][0]['cases']['history_0']['maximum_amplification'] = None
            report['rollouts'][0]['cases']['reference_8'] = {'status': 'skipped_no_active_rows', 'frames': []}
            values = sensitivity_metrics(report, 'raw')
            self.assertEqual(values['sensitivity_overview/raw_status_ok'], 0)
            self.assertEqual(values['sensitivity_overview/raw_cases_failed'], 1)
            self.assertEqual(values['sensitivity_overview/raw_cases_skipped'], 1)
            self.assertNotIn('sensitivity_history/raw_history_0_max_amplification', values)
            self.assertNotIn('sensitivity_reference/raw_reference_8_max_amplification', values)

    def test_multiple_seeds_weights_and_sites_preserve_scientific_definition(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint, report = self.fixture(Path(folder))
            second = copy.deepcopy(report['rollouts'][0]); second['seed'] = 1730
            site = second['cases']['history_0']
            site.update(maximum_amplification=1.2, last_frame_amplification=.7)
            site['frames'][0]['amplification'] = 1.2
            site['frames'][1]['amplification'] = .7
            report['seeds'].append(1730); report['rollouts'].append(second)
            for label in ('raw', 'ema'):
                (checkpoint / f'sensitivity-{label}.json').write_text(json.dumps(report))
            (checkpoint / 'evaluation.complete.json').write_text(json.dumps({'checkpoint_id': 'actual',
                'reports': [{'weights': label, 'file': f'sensitivity-{label}.json'} for label in ('raw', 'ema')]}))
            values = checkpoint_metrics(checkpoint)
            self.assertEqual(len(values), 42)
            self.assertEqual(values['sensitivity_history/raw_history_0_max_amplification'], 1.2)
            self.assertEqual(values['sensitivity_history/raw_history_0_last_frame_amplification'], .5)
            self.assertEqual(values['sensitivity_history/ema_history_0_max_amplification'], 1.2)
            self.assertEqual(values['sensitivity_reference/raw_reference_0_max_amplification'], .8)
            second['cases']['history_0']['last_frame_amplification'] = None
            values = sensitivity_metrics(report, 'raw')
            self.assertNotIn('sensitivity_history/raw_history_0_max_amplification', values)
            self.assertEqual(values['sensitivity_overview/raw_cases_failed'], 1)

    def test_unknown_case_seed_frame_and_aggregate_mismatches_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            _, original = self.fixture(Path(folder))
            for mutation in ('unknown', 'seed', 'frame', 'aggregate'):
                report = copy.deepcopy(original)
                if mutation == 'unknown':
                    report['rollouts'][0]['cases']['other_input'] = {}
                elif mutation == 'seed':
                    report['rollouts'].append(copy.deepcopy(report['rollouts'][0]))
                elif mutation == 'frame':
                    report['rollouts'][0]['cases']['sink']['frames'][0]['target_index'] = 9
                else:
                    report['rollouts'][0]['cases']['sink']['maximum_amplification'] = 999
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    sensitivity_metrics(report, 'raw')

    def test_online_preflight_then_same_id_resume_does_not_consume_601(self):
        with tempfile.TemporaryDirectory() as folder:
            root, sdk = Path(folder), FakeWandb()
            checkpoint, _ = self.fixture(root)
            first = self.start(root, checkpoint, sdk)
            identity = first.receipt['id']
            token = first.receipt['publish_token']
            self.assertEqual(first.run.step, 601)
            self.assertEqual(sdk.records[identity]['history'][600]['train/critic_loss'], .2)
            self.assertNotIn(699, sdk.records[identity]['history'])
            first.finish()
            second = self.start(root, checkpoint, sdk)
            self.assertEqual(second.receipt['id'], identity)
            self.assertNotEqual(second.receipt['publish_token'], token)
            self.assertEqual(second.run.calls, [])
            self.assertEqual(second.run.step, 601)
            second.training({**self.counters, 'rounds': 601, 'student_updates': 451, 'critic/loss': .19})
            self.assertEqual(second.run.calls[-1][1:], (601, False))
            self.assertTrue(second.receipt['online_metrics_verified'])
            self.assertEqual(sdk.inits[-1]['resume'], 'must')
            self.assertEqual(sdk.inits[-1]['mode'], 'online')
            self.assertFalse(sdk.inits[-1]['save_code'])

    def test_remote_history_ahead_starts_linked_run(self):
        with tempfile.TemporaryDirectory() as folder:
            root, sdk = Path(folder), FakeWandb()
            checkpoint, _ = self.fixture(root)
            first = self.start(root, checkpoint, sdk)
            old = first.receipt['id']
            sdk.records[old]['history'][699] = {'rounds': 699}
            second = self.start(root, checkpoint, sdk)
            self.assertNotEqual(second.receipt['id'], old)
            self.assertEqual(second.receipt['parent']['id'], old)
            self.assertEqual(sdk.inits[-1]['resume'], 'never')
            self.assertEqual(sorted(sdk.records[second.receipt['id']]['history']), [600])

    def test_same_round_training_and_evaluation_commit_together(self):
        with tempfile.TemporaryDirectory() as folder:
            root, sdk = Path(folder), FakeWandb()
            checkpoint, _ = self.fixture(root)
            tracker = self.start(root, checkpoint, sdk)
            counters = {**self.counters, 'rounds': 601, 'student_updates': 451}
            tracker.training({**counters, 'student/loss': .08})
            tracker.evaluation(checkpoint, counters)
            row = sdk.records[tracker.receipt['id']]['history'][601]
            self.assertEqual(row['train/student_loss'], .08)
            self.assertEqual(row['sensitivity_overview/checkpoint_round'], 600)
            self.assertEqual(tracker.run.step, 602)

    def test_offline_or_unverified_remote_state_fails_closed(self):
        for offline in (True, False):
            with self.subTest(offline=offline), tempfile.TemporaryDirectory() as folder:
                root, sdk = Path(folder), FakeWandb()
                checkpoint, _ = self.fixture(root)
                sdk.mode = 'offline' if offline else 'online'
                sdk.visible = offline
                with self.assertRaisesRegex(RuntimeError, 'online run|remote API'):
                    self.start(root, checkpoint, sdk)
                self.assertEqual(sdk.finishes[-1], {'exit_code': 1})

    def test_full_raw_files_remotely_verified_and_no_weights_uploaded(self):
        with tempfile.TemporaryDirectory() as folder:
            root, sdk = Path(folder), FakeWandb()
            checkpoint, _ = self.fixture(root)
            (checkpoint / 'student.safetensors').write_bytes(b'NEVER UPLOAD')
            tracker = self.start(root, checkpoint, sdk)
            files = tracker.receipt['sensitivity_report_files']
            self.assertEqual(len(files), 2)
            self.assertTrue(tracker.receipt['sensitivity_report_uploads_verified'])
            self.assertTrue(all(row['remote_sha256_verified'] for row in files))
            raw = next(row for row in files if row['weights'] == 'raw')
            self.assertEqual(raw['sha256'], hashlib.sha256((checkpoint / 'sensitivity-raw.json').read_bytes()).hexdigest())
            self.assertEqual(sdk.records[tracker.run.id]['files'][raw['name']], (checkpoint / 'sensitivity-raw.json').read_bytes())
            self.assertFalse(any('safetensors' in name for name in sdk.records[tracker.run.id]['files']))
            sdk.records[tracker.run.id]['files'][raw['name']] = b'x' * raw['bytes']
            with self.assertRaisesRegex(RuntimeError, 'SHA256'):
                verify_report_files(FakeRemote(sdk, sdk.records[tracker.run.id]), files)

    def test_missing_remote_files_fail_and_staging_is_immutable(self):
        with tempfile.TemporaryDirectory() as folder:
            root, sdk = Path(folder), FakeWandb()
            checkpoint, report = self.fixture(root)
            sdk.files_visible = False
            with self.assertRaisesRegex(RuntimeError, 'remote API'):
                self.start(root, checkpoint, sdk)
            report['requested_input_rms'] = .02
            (checkpoint / 'sensitivity-raw.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'Immutable'):
                stage_checkpoint_reports(checkpoint, root)

    def test_failed_site_omits_new_value_clears_stale_summary_and_keeps_history(self):
        with tempfile.TemporaryDirectory() as folder:
            root, sdk = Path(folder), FakeWandb()
            checkpoint, report = self.fixture(root)
            tracker = self.start(root, checkpoint, sdk)
            key = 'sensitivity_history/raw_history_0_max_amplification'
            report['rollouts'][0]['cases']['history_0']['maximum_amplification'] = None
            metrics = sensitivity_metrics(report, 'raw')
            tracker.publish(metrics, {**self.counters, 'rounds': 601})
            self.assertNotIn(key, tracker.run.summary)
            self.assertNotIn(key, sdk.records[tracker.run.id]['history'][601])
            self.assertEqual(sdk.records[tracker.run.id]['history'][600][key], .8)
            self.assertIn(key, tracker.receipt['cleared_stale_sensitivity_summary_keys'])

    def test_public_api_historical_upload_and_validated_prune_lineage(self):
        with tempfile.TemporaryDirectory() as folder:
            root, sdk = Path(folder), FakeWandb()
            checkpoint, _ = self.fixture(root)
            weights = {'student.safetensors': {'size': 3}}
            (checkpoint / 'student.safetensors').write_bytes(b'abc')
            meta = json.loads((checkpoint / 'metadata.json').read_text())
            meta.update(checkpoint_id='pruned', optimizer_state_retained=False,
                        optimizer_pruned_from_checkpoint_id='actual', optimizer_pruned_weights=weights)
            (checkpoint / 'metadata.json').write_text(json.dumps(meta))
            (checkpoint / 'checkpoint.complete.json').write_text(json.dumps({'checkpoint_id': 'pruned',
                'files': [{'path': 'student.safetensors', 'size': 3}]}))
            receipt = checkpoint.with_name(checkpoint.name + '.optimizer-prune.json')
            receipt.write_text(json.dumps({'path': str(checkpoint), 'previous_checkpoint_id': 'actual',
                'checkpoint_id': 'pruned', 'optimizer_state_retained': False, 'weights_unchanged': True, 'weights': weights}))
            record = {'summary': {}, 'history': {}, 'files': {}}
            verified = upload_checkpoint_reports(FakeRemote(sdk, record), checkpoint, root, verify_timeout=0)
            self.assertEqual(len(verified), 2)
            self.assertTrue(all(row['checkpoint_id'] == 'actual' for row in verified))
            self.assertEqual(sdk.inits, [])
            receipt.unlink()
            with self.assertRaises(ValueError):
                checkpoint_metrics(checkpoint)

    def test_config_must_explicitly_enable_online(self):
        for change in ({'enabled': False}, {'mode': 'offline'}):
            with tempfile.TemporaryDirectory() as folder:
                config = copy.deepcopy(self.config)
                config['wandb'].update(change)
                with self.assertRaisesRegex(ValueError, 'requires'):
                    start_online_tracking(config, folder, self.counters, epoch_steps=1, world_size=40,
                                          wandb_module=FakeWandb())

    def test_evaluation_receipt_must_identify_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint, _ = self.fixture(Path(folder))
            receipt = checkpoint / 'evaluation.complete.json'
            value = json.loads(receipt.read_text()); value['checkpoint_id'] = 'other'
            receipt.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'does not identify'):
                checkpoint_metrics(checkpoint)

    def test_root_tracking_guard_restores_rng_even_when_sdk_raises(self):
        source = Path(__file__).resolve().parents[1] / 'learn_dmd_joint.py'
        function = next(node for node in ast.parse(source.read_text()).body
                        if isinstance(node, ast.FunctionDef) and node.name == 'root_tracking_action')
        namespace = {'root_action': lambda action: action(), 'rng_state': random.getstate,
                     'restore_rng': random.setstate}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
        state = random.getstate()
        try:
            expected = random.random()
            random.setstate(state)
            def failed_sdk_call():
                random.random()  # Same global RNG used by W&B API retry jitter.
                raise RuntimeError('retry exhausted')
            with self.assertRaisesRegex(RuntimeError, 'retry exhausted'):
                namespace['root_tracking_action'](failed_sdk_call)
            self.assertEqual(random.random(), expected)
        finally:
            random.setstate(state)


if __name__ == '__main__':
    unittest.main()
