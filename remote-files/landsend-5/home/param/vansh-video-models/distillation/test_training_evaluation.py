import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from distillation.training_evaluation import evaluate_checkpoint
from distillation.training_tracking import EVALUATION_SCHEMA_VERSION


def artifact(path):
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


class TrainingEvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.run_dir = Path(temporary.name) / 'run-fixed'
        self.directory = self.run_dir / 'step-000012'
        self.directory.mkdir(parents=True)
        (self.directory / 'metadata.json').write_text(json.dumps({'checkpoint_id': 'id-12'}))
        self.net = object()
        self.rows = {'latents': torch.zeros(1, 2, 11, 1, 1), 'pose': torch.ones(1, 2, 11, 1, 1),
                     'ofi': torch.arange(11)[None], 'audio': torch.ones(1, 22, 3),
                     'refs': torch.ones(1, 2, 9, 1, 1), 'ref_keep': torch.ones(1, 9, dtype=torch.bool),
                     'drop': torch.zeros(1, dtype=torch.bool)}
        self.raw = {'chunk_index': torch.tensor([17]), 'person_index': torch.tensor([4]),
                    'source_frame_indices': torch.arange(11)[None], 'epoch_window_id': torch.tensor([71]),
                    'source_paths': {'video': '/source/person4.mp4', 'audio': '/source/person4.pt'}}
        self.audio = np.zeros(32, dtype=np.float32)
        self.config = {'clipnames_root': '/datasets', 'train_csv': 'train.csv', 'refs': {'bank': '/banks/strict'},
                       'band_steps': [2, 2, 2, 2], 'resolver_steps': 1, 'sensitivity_seeds': [1729],
                       'sensitivity_rms': .01, 'sensitivity_validate_half_rms': False,
                       'rollout_eval': {'seed': 1729, 'latent_frames': 189, 'fps': 25, 'height': 256,
                                        'width': 480, 'vae_path': '/weights/wan22.pth', 'vae_sha256': 'fixed-vae'}}
        self.rollout = self._patch('evaluate_rollout', self._rollout)
        self.sensitivity = self._patch('run_endpoint_sensitivity', lambda *args: {'status': 'ok', 'panels': []})
        self.render = self._patch('render_endpoint_sensitivity', self._render)
        self.gallery = self._patch('append_local_gallery', lambda result, path: {'directory': str(path)})

    def _patch(self, name, function):
        patcher = patch('distillation.training_evaluation.' + name, side_effect=function)
        mocked = patcher.start()
        self.addCleanup(patcher.stop)
        return mocked

    def _rollout(self, net, rows, config, output, *, audio_np):
        self.assertIs(net, self.net)
        self.assertIs(rows, self.rows)
        self.assertIs(audio_np, self.audio)
        options = config['rollout_eval']
        paths = {}
        for name in ('generated_video', 'gt_video', 'original_video', 'original_gt_video',
                     'gt_l2_curve', 'background_curves', 'curves', 'latents'):
            path = output / (name + '.bin')
            path.write_bytes(f'{options["checkpoint_id"]} {options["weights"]} {name}'.encode())
            paths[name] = artifact(path)
        result = {'status': 'complete', 'run_id': options['run_id'], 'checkpoint_id': options['checkpoint_id'],
                  'weights': options['weights'], 'source': options['source'], 'seed': options['seed'],
                  'scalars': {'background_color_mean': 1.2, 'background_color_tail': None,
                              'background_texture_mean': -.5, 'background_texture_tail': None},
                  'artifacts': paths, 'manifest_path': str(output / 'rollout.json')}
        if options.get('evaluation_kind') == 'in_memory':
            result.update({key: options[key] for key in
                           ('evaluation_id', 'evaluation_kind', 'student_updates', 'rounds')})
        (output / 'rollout.json').write_text(json.dumps(result))
        return result

    def _render(self, sensitivity, output):
        self.assertEqual(sensitivity['status'], 'ok')
        figures = {}
        for kind in ('stabilizer', 'resolver'):
            path = output / (kind + '-context.png')
            path.write_bytes(kind.encode())
            figures[kind + '_context_gain'] = path
        return figures

    def evaluate(self, *, config=None, step=12, weights='raw', directory=None, identity=None):
        return evaluate_checkpoint(self.net, self.rows, self.raw, self.audio, config or self.config,
                                   directory or self.directory, step, weights, self.run_dir,
                                   'https://wandb.invalid/run', identity=identity)

    def test_live_model_evaluation_has_its_own_identity_without_checkpoint_metadata(self):
        identity = {'checkpoint_id': None, 'evaluation_id': '12345678123456781234567812345678',
                    'evaluation_kind': 'in_memory', 'student_updates': 400, 'rounds': 550,
                    'source_commit': 'tested-code'}
        directory = self.run_dir / 'evaluation-runs' / 'eval-student-400-fixture'
        report = self.evaluate(directory=directory, step=550, identity=identity)
        path = (directory / report['file']).resolve()
        self.assertEqual(path, self.result_path(identity['evaluation_id']))
        result = json.loads(path.read_text())
        self.assertIsNone(result['checkpoint_id'])
        self.assertEqual(result['evaluation_identity'], identity)
        self.assertEqual(result['evaluation_id'], identity['evaluation_id'])
        self.assertEqual(result['checkpoint_step'], 550)
        self.assertEqual(set(p.name for p in directory.iterdir()), {'evaluation.identity.json'})
        self.assertEqual(self.evaluate(directory=directory, step=550, identity=identity), report)
        self.assertEqual(self.rollout.call_count, 1)
        changed = {**identity, 'source_commit': 'other'}
        with self.assertRaisesRegex(ValueError, 'identity differs'):
            self.evaluate(directory=directory, step=550, identity=changed)
        with self.assertRaisesRegex(ValueError, 'training round'):
            self.evaluate(directory=directory, step=551, identity=identity)
        self.edit(path, lambda value: value.update(evaluation_id='87654321876543218765432187654321'))
        with self.assertRaisesRegex(ValueError, 'checkpoint/protocol'):
            self.evaluate(directory=directory, step=550, identity=identity)

    def test_live_model_identity_rejects_checkpoint_aliases_and_unsafe_identifiers(self):
        base = {'checkpoint_id': None, 'evaluation_id': '12345678123456781234567812345678',
                'evaluation_kind': 'in_memory', 'student_updates': 400, 'rounds': 550}
        for key, value in (('checkpoint_id', 'fake-checkpoint'), ('evaluation_id', '../escape'),
                           ('student_updates', True), ('rounds', 399)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.evaluate(step=550, identity={**base, key: value})

    def result_path(self, identity='id-12', weights='raw'):
        return self.run_dir / 'evaluations' / identity / weights / 'evaluation.json'

    def edit(self, path, mutate):
        value = json.loads(path.read_text())
        mutate(value)
        path.write_text(json.dumps(value))

    def test_composes_fixed_source_plots_rollout_and_checkpoint_metadata(self):
        config_before = copy.deepcopy(self.config)
        report = self.evaluate()
        result_path = (self.directory / report['file']).resolve()
        self.assertEqual(result_path, self.result_path())
        self.assertEqual(report['weights'], 'raw')
        result = json.loads(result_path.read_text())
        self.assertEqual((result['checkpoint_id'], result['checkpoint_step'], result['weights']), ('id-12', 12, 'raw'))
        self.assertEqual(result['schema'], EVALUATION_SCHEMA_VERSION)
        self.assertEqual(result['wandb_url'], 'https://wandb.invalid/run')
        self.assertEqual(result['run_id'], self.run_dir.name)
        self.assertNotIn('manifest_path', result)
        self.assertEqual(result['scalars'], {'background_color_mean': 1.2, 'background_texture_mean': -.5})
        expected_source = {'chunk_index': 17, 'person_index': 4, 'source_frame_indices': list(range(11)),
                           'source_clip_index': 71, 'paths': self.raw['source_paths'],
                           'csv': '/datasets/train.csv', 'bank': '/banks/strict'}
        self.assertEqual(result['source'], expected_source)
        options = self.rollout.call_args.args[2]['rollout_eval']
        self.assertEqual(options, {**self.config['rollout_eval'], 'run_id': self.run_dir.name,
                                  'checkpoint_id': 'id-12', 'weights': 'raw', 'source': expected_source})
        expected_artifacts = {'generated_video', 'gt_video', 'original_video', 'original_gt_video',
                              'gt_l2_curve', 'background_curves', 'curves', 'latents',
                              'stabilizer_context_gain', 'resolver_context_gain',
                              'sensitivity_measurements', 'evaluation_protocol'}
        self.assertEqual(set(result['artifacts']), expected_artifacts)
        for record in result['artifacts'].values():
            self.assertEqual(record, artifact(Path(record['path'])))
        self.assertEqual(self.config, config_before)
        self.assertEqual(self.gallery.call_args.args[1], self.run_dir / 'gallery')
        self.assertEqual(self.gallery.call_args.args[0]['scalars'], result['scalars'])
        self.assertEqual(set(path.name for path in self.directory.iterdir()), {'metadata.json'})

    def test_completed_evaluation_is_idempotent_and_keeps_checkpoint_flat(self):
        expected = self.evaluate()
        before = self.result_path().read_bytes()
        actual = self.evaluate()
        self.assertEqual(actual, expected)
        self.assertEqual(self.result_path().read_bytes(), before)
        for mocked in (self.rollout, self.sensitivity, self.render, self.gallery):
            self.assertEqual(mocked.call_count, 1)
        self.assertFalse(any(path.is_dir() for path in self.directory.iterdir()))

    def test_retry_after_sensitivity_failure_reuses_completed_rollout(self):
        self.sensitivity.side_effect = [RuntimeError('sensitivity interrupted'), {'status': 'ok', 'panels': []}]
        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            self.evaluate()
        self.assertFalse(self.result_path().exists())
        self.assertTrue((self.result_path().parent / 'rollout.json').exists())
        self.evaluate()
        self.assertEqual(self.rollout.call_count, 1)
        self.assertEqual(self.sensitivity.call_count, 2)
        self.assertEqual(self.render.call_count, 1)
        self.assertEqual(self.gallery.call_count, 1)

    def test_retry_after_plot_failure_reuses_rollout_and_sensitivity(self):
        self.render.side_effect = RuntimeError('plot interrupted')
        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            self.evaluate()
        self.assertTrue((self.result_path().parent / 'sensitivity.json').exists())
        self.assertFalse(self.result_path().exists())
        self.render.side_effect = self._render
        self.evaluate()
        self.assertEqual(self.rollout.call_count, 1)
        self.assertEqual(self.sensitivity.call_count, 1)
        self.assertEqual(self.render.call_count, 2)

    def test_failed_gallery_does_not_create_a_completed_evaluation(self):
        self.gallery.side_effect = RuntimeError('gallery interrupted')
        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            self.evaluate()
        self.assertFalse(self.result_path().exists())
        self.gallery.side_effect = lambda result, path: {'directory': str(path)}
        self.evaluate()
        self.assertEqual(self.rollout.call_count, 1)
        self.assertEqual(self.sensitivity.call_count, 1)
        self.assertEqual(self.gallery.call_count, 2)

    def test_completed_checkpoint_identity_weight_schema_and_step_mismatch_rejected(self):
        self.evaluate()
        path = self.result_path()
        original = path.read_bytes()
        for key, value in (('checkpoint_id', 'other'), ('checkpoint_step', 13),
                           ('weights', 'ema'), ('schema', 'old-protocol')):
            with self.subTest(key=key):
                path.write_bytes(original)
                self.edit(path, lambda result: result.update({key: value}))
                with self.assertRaisesRegex(ValueError, 'checkpoint/protocol'):
                    self.evaluate()
        self.assertEqual(self.rollout.call_count, 1)

    def test_cached_partial_rollout_identity_mismatch_rejected(self):
        self.sensitivity.side_effect = RuntimeError('sensitivity interrupted')
        with self.assertRaises(RuntimeError):
            self.evaluate()
        path = self.result_path().parent / 'rollout.json'
        original = path.read_bytes()
        for key, value in (('checkpoint_id', 'other'), ('weights', 'ema'), ('source', {'chunk_index': 999})):
            with self.subTest(key=key):
                path.write_bytes(original)
                self.edit(path, lambda result: result.update({key: value}))
                with self.assertRaisesRegex(ValueError, 'rollout does not match'):
                    self.evaluate()
        self.assertEqual(self.rollout.call_count, 1)

    def test_same_run_protocol_freezes_source_options_and_tensor_inputs(self):
        self.evaluate()
        protocol = json.loads((self.run_dir / 'evaluation-protocol.json').read_text())
        self.assertEqual(protocol['schema'], EVALUATION_SCHEMA_VERSION)
        self.assertEqual(set(protocol['inputs']), {'latents', 'pose', 'ofi', 'audio', 'refs', 'ref_keep', 'drop'})
        for key, value in (('sensitivity_rms', .02), ('sensitivity_seeds', [5]), ('resolver_steps', 2),
                           ('sensitivity_validate_half_rms', True), ('band_steps', [1, 1, 1, 1])):
            changed = copy.deepcopy(self.config)
            changed[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'protocol changed'):
                self.evaluate(config=changed)
        changed = copy.deepcopy(self.config)
        changed['rollout_eval']['seed'] += 1
        with self.assertRaisesRegex(ValueError, 'protocol changed'):
            self.evaluate(config=changed)
        self.raw['epoch_window_id'] += 1
        with self.assertRaisesRegex(ValueError, 'protocol changed'):
            self.evaluate()
        self.raw['epoch_window_id'] -= 1
        self.rows['refs'][0, 0, 0, 0, 0] += 1
        with self.assertRaisesRegex(ValueError, 'protocol changed'):
            self.evaluate()
        self.assertEqual(self.rollout.call_count, 1)

    def test_completed_and_partial_rollout_artifacts_are_verified_before_reuse(self):
        self.evaluate()
        path = self.result_path().parent / 'generated_video.bin'
        original = path.read_bytes()
        path.write_bytes(b'X' * len(original))
        with self.assertRaisesRegex(ValueError, 'integrity verification'):
            self.evaluate()
        self.result_path().unlink()
        with self.assertRaisesRegex(ValueError, 'integrity verification'):
            self.evaluate()
        self.assertEqual(self.rollout.call_count, 1)
        self.assertEqual(self.sensitivity.call_count, 1)

    def test_protocol_supports_bfloat16_and_detects_layout_changes(self):
        self.rows['audio'] = self.rows['audio'].bfloat16()
        self.evaluate()
        self.rows['audio'] = self.rows['audio'].reshape(1, 11, 6)
        with self.assertRaisesRegex(ValueError, 'protocol changed'):
            self.evaluate()

    def test_raw_ema_and_new_checkpoints_share_one_protocol_without_overwriting_entries(self):
        raw = self.evaluate()
        ema = self.evaluate(weights='ema')
        self.assertNotEqual(raw['file'], ema['file'])
        next_directory = self.run_dir / 'step-000024'
        next_directory.mkdir()
        (next_directory / 'metadata.json').write_text(json.dumps({'checkpoint_id': 'id-24'}))
        later = self.evaluate(directory=next_directory, step=24)
        self.assertEqual(json.loads(self.result_path('id-24').read_text())['checkpoint_step'], 24)
        self.assertTrue((self.directory / raw['file']).exists())
        self.assertTrue((self.directory / ema['file']).exists())
        self.assertTrue((next_directory / later['file']).exists())
        self.assertEqual(self.gallery.call_count, 3)
        self.assertFalse(any(path.is_dir() for path in next_directory.iterdir()))


if __name__ == '__main__':
    unittest.main()
