import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from distillation.training_metrics import METRIC_SCHEMA_VERSION, TRAINING_METRIC_KEYS
from distillation.training_tracking import DRIFT_SCALARS, EVALUATION_SCHEMA_VERSION, FIGURES, start_training_tracking


class FakeRun:
    def __init__(self, sdk, record, options):
        self.sdk, self.record = sdk, record
        self.id, self.entity, self.project = record['id'], options['entity'], options['project']
        self.url = f'https://wandb.invalid/{self.entity}/{self.project}/runs/{self.id}'
        self.settings, self.offline = SimpleNamespace(mode=sdk.mode), sdk.mode != 'online'
        self.summary = record['summary']
        self.definitions = {}

    @property
    def step(self):
        return len(self.record['history'])

    def define_metric(self, name, **kwargs):
        self.definitions[name] = kwargs

    def log(self, payload, *, commit):
        assert commit is True
        self.record['history'].append({'sdk_step': self.step, 'payload': copy.deepcopy(payload)})
        self.summary.update(payload)

    def save(self, path, *, base_path, policy, glob):
        assert policy == 'now' and glob is False
        relative = str(Path(path).relative_to(base_path))
        self.record['files'][relative] = Path(path).read_bytes()
        return [path]

    def finish(self, **kwargs):
        self.sdk.finishes.append(kwargs)


class FakeWandb:
    def __init__(self):
        self.records, self.inits, self.finishes, self.api_paths = {}, [], [], []
        self.mode = 'online'

    def init(self, **kwargs):
        self.inits.append(copy.deepcopy(kwargs))
        identity = kwargs['id']
        if kwargs['resume'] == 'must':
            assert identity in self.records
        else:
            assert kwargs['resume'] == 'never' and identity not in self.records
        record = self.records.setdefault(identity, {'id': identity, 'summary': {}, 'history': [], 'files': {}})
        return FakeRun(self, record, kwargs)

    def Api(self, **kwargs):
        assert kwargs == {'timeout': 15}
        return self

    def run(self, path):
        self.api_paths.append(path)
        return SimpleNamespace(summary=self.records[path.rsplit('/', 1)[-1]]['summary'])

    def Image(self, path, *, caption):
        assert Path(path).exists()
        return {'media': 'image', 'path': path, 'caption': caption}

    def Video(self, path, *, caption, format):
        assert Path(path).exists() and format == 'mp4'
        return {'media': 'video', 'path': path, 'caption': caption, 'format': format}


class TrainingTrackingTests(unittest.TestCase):
    config = {'wandb': {'enabled': True, 'mode': 'online', 'entity': 'test-entity',
                        'project': 'test-project', 'run_name': 'fixed-metrics'}}

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root, self.sdk = Path(temporary.name), FakeWandb()

    def start(self, step=0, checkpoint=None, config=None):
        return start_training_tracking(config or self.config, self.root, {'rounds': step},
                                       epoch_steps=120, world_size=40, resume_checkpoint=checkpoint,
                                       wandb_module=self.sdk)

    def fixture(self, step=8, labels=('raw',), identity=None):
        identity = identity or f'checkpoint-{step}'
        directory = self.root / identity
        directory.mkdir()
        reports = []
        for label in labels:
            variant = directory / label
            variant.mkdir()
            artifacts = {}
            for name in (*FIGURES, 'generated_video', 'gt_video', 'measurements'):
                suffix = '.png' if name in FIGURES else '.mp4' if name.endswith('video') else '.json'
                path = variant / (name + suffix)
                path.write_bytes(f'{identity} {label} {name}'.encode())
                artifacts[name] = {'path': path.name, 'bytes': path.stat().st_size,
                                   'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            manifest = {'weights': label, 'checkpoint_id': identity, 'checkpoint_step': step,
                        'status': 'complete', 'scalars': {name: index * .1 for index, name in enumerate(DRIFT_SCALARS)},
                        'artifacts': artifacts}
            path = variant / 'manifest.json'
            path.write_text(json.dumps(manifest))
            reports.append({'weights': label, 'file': str(path.relative_to(directory))})
        completion = {'checkpoint_id': identity, 'reports': reports}
        (directory / 'evaluation.complete.json').write_text(json.dumps(completion))
        return directory

    def edit_manifest(self, directory, mutate, label='raw'):
        path = directory / label / 'manifest.json'
        manifest = json.loads(path.read_text())
        mutate(manifest)
        path.write_text(json.dumps(manifest))

    def test_in_memory_evaluation_uses_real_identity_and_existing_round_axis(self):
        tracker = self.start(step=550)
        directory = self.fixture(step=550)
        identity = {'checkpoint_id': None, 'evaluation_id': '12345678123456781234567812345678',
                    'evaluation_kind': 'in_memory', 'student_updates': 400, 'rounds': 550}
        completion_path = directory / 'evaluation.complete.json'
        completion = json.loads(completion_path.read_text())
        completion.update(identity)
        completion_path.write_text(json.dumps(completion))
        self.edit_manifest(directory, lambda manifest: manifest.update(identity))
        tracker.evaluation(directory, {'rounds': 550, 'student_updates': 400})
        payload = tracker.run.record['history'][-1]['payload']
        self.assertEqual(payload['eval/checkpoint_step'], 550)
        self.assertIn('student update 400', payload['eval/sensitivity/raw/stabilizer_context_gain']['caption'])
        self.assertIn('in-memory', payload['eval/sensitivity/raw/stabilizer_context_gain']['caption'])
        self.assertIsNone(tracker.receipt['last_eval_checkpoint'])
        self.assertEqual(tracker.receipt['last_eval_id'], identity['evaluation_id'])
        with self.assertRaisesRegex(ValueError, 'counters differ'):
            tracker.evaluation(directory, {'rounds': 550, 'student_updates': 401})
        self.edit_manifest(directory, lambda manifest: manifest.update(
            evaluation_id='87654321876543218765432187654321'))
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            tracker.evaluation(directory, {'rounds': 550, 'student_updates': 400})

    def test_exact_schema_and_system_streams_disabled(self):
        tracker = self.start()
        defined = tracker.run.definitions
        self.assertEqual(set(defined), TRAINING_METRIC_KEYS | {'train/step', 'eval/checkpoint_step', 'eval/*'})
        self.assertEqual(defined['train/step'], {'hidden': True})
        self.assertEqual(defined['eval/checkpoint_step'], {'hidden': True})
        for key in TRAINING_METRIC_KEYS:
            self.assertEqual(defined[key], {'step_metric': 'train/step', 'step_sync': False})
        self.assertEqual(defined['eval/*'], {'step_metric': 'eval/checkpoint_step', 'step_sync': False})
        options = self.sdk.inits[0]
        self.assertTrue(options['settings']['x_disable_stats'])
        self.assertTrue(options['settings']['x_disable_meta'])
        self.assertFalse(options['save_code'])
        self.assertEqual(options['config']['metric_schema'], METRIC_SCHEMA_VERSION)
        self.assertEqual(options['config']['evaluation_schema'], EVALUATION_SCHEMA_VERSION)
        self.assertEqual(tracker.run.record['history'], [])

    def test_only_explicit_training_scalars_are_logged(self):
        tracker = self.start()
        metrics = {key: index * .1 for index, key in enumerate(sorted(TRAINING_METRIC_KEYS))}
        tracker.training({'rounds': 1, 'training_metrics': metrics, 'phase': 'joint', 'progress': 99,
                          'seconds': {'outer': 10}, 'compile': {'unique_graphs': 32}})
        payload = tracker.run.record['history'][0]['payload']
        self.assertEqual(payload, {'train/step': 1, **metrics})
        self.assertEqual(len(payload), 55)
        self.assertFalse(any(key.startswith(('phase/', 'progress/', 'compile/', 'timing/')) for key in payload))
        self.assertEqual(json.loads((self.root / 'wandb.json').read_text())['last_train_step'], 1)

    def test_unexpected_training_keys_are_rejected(self):
        tracker = self.start()
        for key in ('phase', 'phase/joint', 'progress/epoch', 'gradients/student_norm'):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Unexpected training keys'):
                tracker.training({'rounds': 1, 'training_metrics': {key: 1.}})
        self.assertEqual(tracker.run.record['history'], [])

    def test_nonfinite_and_nonnumeric_training_values_are_rejected(self):
        tracker = self.start()
        for value in (float('nan'), float('inf'), -float('inf'), '0.1', None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'finite scalars'):
                tracker.training({'rounds': 1, 'training_metrics': {'train/dmd_loss': value}})
        self.assertEqual(tracker.run.record['history'], [])

    def test_duplicate_or_older_training_round_is_rejected(self):
        tracker = self.start()
        tracker.training({'rounds': 5, 'training_metrics': {'train/critic_fm_loss': .2}})
        for step in (5, 4):
            with self.subTest(step=step), self.assertRaisesRegex(ValueError, 'once per completed round'):
                tracker.training({'rounds': step, 'training_metrics': {'train/critic_fm_loss': .3}})
        self.assertEqual(len(tracker.run.record['history']), 1)

    def test_warmup_does_not_fabricate_student_or_empty_bucket_observations(self):
        tracker = self.start()
        warmup = {'train/critic_fm_loss': .2, 'train/critic_lr': .0001,
                  'critic_fm_stabilizer/timewise_endpoint_loss/t_0.20_0.40': .3}
        tracker.training({'rounds': 1, 'training_metrics': warmup})
        payload = tracker.run.record['history'][0]['payload']
        self.assertEqual(payload, {'train/step': 1, **warmup})
        self.assertNotIn('train/dmd_loss', payload)
        self.assertFalse(any(key.startswith('velocity_probes/') for key in payload))

    def test_late_evaluation_uses_checkpoint_axis_and_monotonic_sdk_history(self):
        tracker = self.start()
        tracker.training({'rounds': 100, 'training_metrics': {'train/dmd_loss': .1}})
        tracker.evaluation(self.fixture(80), {'rounds': 80})
        tracker.training({'rounds': 101, 'training_metrics': {'train/dmd_loss': .2}})
        history = tracker.run.record['history']
        self.assertEqual([row['sdk_step'] for row in history], [0, 1, 2])
        self.assertEqual(history[1]['payload']['eval/checkpoint_step'], 80)
        self.assertNotIn('train/step', history[1]['payload'])
        self.assertEqual(tracker.receipt['last_train_step'], 101)
        self.assertEqual(tracker.receipt['last_eval_step'], 80)

    def test_fixed_eval_keys_and_captions_across_checkpoints_and_weight_variants(self):
        tracker = self.start()
        for step in (8, 9):
            tracker.evaluation(self.fixture(step, ('raw', 'ema')), {'rounds': step})
        history = tracker.run.record['history']
        self.assertEqual(set(history[0]['payload']), set(history[1]['payload']))
        expected = {'eval/checkpoint_step'}
        for label in ('raw', 'ema'):
            expected |= {f'eval/drift/{label}/{name}' for name in DRIFT_SCALARS}
            expected |= {f'eval/{group}/{label}/{name}' for name, group in FIGURES.items()}
            expected |= {f'eval/rollout/{label}/{name}' for name in ('generated_video', 'gt_video')}
        self.assertEqual(set(history[0]['payload']), expected)
        for key, value in history[1]['payload'].items():
            if isinstance(value, dict):
                label = key.split('/')[2]
                self.assertIn(f'{label} · checkpoint 9 · checkpoint-9', value['caption'])
                self.assertEqual(value['media'], 'video' if key.endswith('_video') else 'image')

    def test_all_measurements_and_media_upload_as_exact_verified_bytes(self):
        tracker = self.start()
        directory = self.fixture()
        tracker.evaluation(directory, {'rounds': 8})
        files = tracker.run.record['files']
        self.assertEqual(len(files), 8)
        for relative, data in files.items():
            self.assertEqual(data, (self.root / relative).read_bytes())
        self.assertIn('checkpoint-8/raw/measurements.json', files)
        self.assertIn('checkpoint-8/raw/manifest.json', files)

    def test_artifact_hash_and_size_fail_before_any_history_is_published(self):
        tracker = self.start()
        for label, replacement in (('hash', b'X'), ('size', b'additional bytes')):
            directory = self.fixture(identity=label)
            path = directory / 'raw' / 'generated_video.mp4'
            if label == 'hash':
                replacement *= path.stat().st_size
            path.write_bytes(replacement)
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, 'integrity check'):
                tracker.evaluation(directory, {'rounds': 8})
        self.assertEqual(tracker.run.record['history'], [])
        self.assertEqual(tracker.run.record['files'], {})

    def test_invalid_eval_identity_status_and_step_are_rejected(self):
        tracker = self.start()
        for field, value in (('checkpoint_id', 'different'), ('checkpoint_step', 9), ('status', 'partial'), ('weights', 'unknown')):
            directory = self.fixture(identity=field)
            self.edit_manifest(directory, lambda manifest: manifest.update({field: value}))
            with self.subTest(field=field), self.assertRaises(ValueError):
                tracker.evaluation(directory, {'rounds': 8})
        self.assertEqual(tracker.run.record['history'], [])

    def test_missing_required_media_and_unknown_scalar_fail(self):
        tracker = self.start()
        for name in ('generated_video', 'stabilizer_context_gain', 'random_metric'):
            directory = self.fixture(identity=name)
            if name == 'random_metric':
                self.edit_manifest(directory, lambda manifest: manifest['scalars'].update(random_metric=1.))
            else:
                self.edit_manifest(directory, lambda manifest: manifest['artifacts'].pop(name))
            with self.subTest(name=name), self.assertRaises(ValueError):
                tracker.evaluation(directory, {'rounds': 8})
        self.assertEqual(tracker.run.record['history'], [])

    def test_missing_drift_scalar_stays_absent_and_nan_is_rejected(self):
        tracker = self.start()
        directory = self.fixture()
        self.edit_manifest(directory, lambda manifest: manifest['scalars'].pop('background_color_tail'))
        tracker.evaluation(directory, {'rounds': 8})
        self.assertNotIn('eval/drift/raw/background_color_tail', tracker.run.record['history'][0]['payload'])
        bad = self.fixture(identity='nan')
        self.edit_manifest(bad, lambda manifest: manifest['scalars'].update(background_color_tail=float('nan')))
        with self.assertRaisesRegex(ValueError, 'omitted, not NaN'):
            tracker.evaluation(bad, {'rounds': 8})
        self.assertEqual(len(tracker.run.record['history']), 1)

    def test_compatible_resume_preserves_run_and_last_training_step(self):
        first = self.start()
        first.training({'rounds': 10, 'training_metrics': {'train/dmd_loss': .1}})
        resumed = self.start(10)
        self.assertEqual(resumed.run.id, first.run.id)
        self.assertEqual(self.sdk.inits[-1]['resume'], 'must')
        self.assertEqual(resumed.receipt['last_train_step'], 10)
        with self.assertRaises(ValueError):
            resumed.training({'rounds': 10, 'training_metrics': {}})
        resumed.training({'rounds': 11, 'training_metrics': {'train/dmd_loss': .2}})
        self.assertEqual([row['sdk_step'] for row in resumed.run.record['history']], [0, 1])

    def test_remote_ahead_training_or_evaluation_forks(self):
        for axis in ('train/step', 'eval/checkpoint_step'):
            with self.subTest(axis=axis):
                first = self.start()
                first.run.summary[axis] = 11
                resumed = self.start(10)
                self.assertNotEqual(resumed.run.id, first.run.id)
                self.assertEqual(self.sdk.inits[-1]['resume'], 'never')
                self.assertEqual(resumed.receipt['parent']['id'], first.run.id)
                self.assertEqual(resumed.receipt['parent']['checkpoint_step'], 10)
                self.assertEqual(resumed.receipt['last_train_step'], -1)

    def test_changed_training_or_evaluation_schema_forks(self):
        for schema in ('metric_schema', 'evaluation_schema'):
            with self.subTest(schema=schema):
                first = self.start()
                receipt_path = self.root / 'wandb.json'
                receipt = json.loads(receipt_path.read_text())
                receipt[schema] = 'previous-protocol'
                receipt_path.write_text(json.dumps(receipt))
                resumed = self.start(10)
                self.assertNotEqual(resumed.run.id, first.run.id)
                self.assertEqual(resumed.receipt['parent']['id'], first.run.id)

    def test_identity_mismatch_and_offline_mode_fail(self):
        first = self.start()
        changed = copy.deepcopy(self.config)
        changed['wandb']['project'] = 'different-project'
        with self.assertRaisesRegex(ValueError, 'identity differs'):
            self.start(config=changed)
        self.assertEqual(len(self.sdk.inits), 1)
        self.sdk.mode = 'offline'
        with self.assertRaisesRegex(RuntimeError, 'online run'):
            self.start()
        self.assertEqual(self.sdk.finishes[-1], {'exit_code': 1})
        self.assertEqual(first.run.id, self.sdk.inits[-1]['id'])

    def test_resume_checkpoint_evaluation_keeps_training_axis_separate(self):
        checkpoint = self.fixture(8)
        tracker = self.start(8, checkpoint)
        self.assertEqual(tracker.run.record['history'][0]['payload']['eval/checkpoint_step'], 8)
        tracker.training({'rounds': 9, 'training_metrics': {'train/critic_fm_loss': .1}})
        self.assertEqual(tracker.run.record['history'][1]['payload']['train/step'], 9)
        self.assertEqual(tracker.receipt['resume_checkpoint'], str(checkpoint))

    def test_empty_duplicate_or_mislabelled_reports_are_rejected(self):
        tracker = self.start()
        for case in ('empty', 'duplicate', 'mislabelled'):
            directory = self.fixture(identity=case)
            path = directory / 'evaluation.complete.json'
            completion = json.loads(path.read_text())
            if case == 'empty':
                completion['reports'] = []
            elif case == 'duplicate':
                completion['reports'] *= 2
            else:
                completion['reports'][0]['weights'] = 'ema'
            path.write_text(json.dumps(completion))
            with self.subTest(case=case), self.assertRaises(ValueError):
                tracker.evaluation(directory, {'rounds': 8})
        self.assertEqual(tracker.run.record['history'], [])


if __name__ == '__main__':
    unittest.main()
