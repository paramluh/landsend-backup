"""CPU checks for rollout causality, precompression metrics, and publication staging."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from distillation.rollout_evaluation import (
    DRIFT_KEYS, _decode_video, _eager_blocks, _evaluation_geometry, append_local_gallery, generate_student_rollout,
    resolve_generated_sequence, summarize_drift,
)


class FakeNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.shards = None
        self.transformer = type('Blocks', (), {'blocks': [None]})()

    @contextmanager
    def references(self, latents, keep):
        yield self

    def audio_tokens(self, audio):
        return audio

    def frame_audio(self, tokens, ofi, drop):
        return torch.zeros(ofi.shape[0], ofi.shape[1], 1, 1)

    def forward_pass(self, frames, plan, ext, audio, mask, heads, keep=(), mix_mask=None):
        self.calls.append([(frame[0].clone(), frame[3], frame[4]) for frame in frames])
        # An explicit dependence on resolver history makes GT leakage visible.
        history = frames[1][0] + frames[2][0] if heads else frames[0][0]*0
        velocity = torch.stack([frames[i][0] + history for i in range(heads)], dim=2) if heads else None
        kv = torch.stack([frames[i][0].mean((1, 2, 3))[:, None, None, None] for i in keep], dim=1) if keep else None
        return velocity, [kv], [kv]


def rows(total=9):
    return {'latents': torch.full((1, 1, total, 1, 1), 100.),
            'pose': torch.arange(total).float().view(1, 1, total, 1, 1),
            'ofi': torch.arange(total)[None], 'audio': torch.zeros(1, 20, 1),
            'refs': torch.zeros(1, 1, 9, 1, 1), 'ref_keep': torch.ones(1, 9, dtype=torch.bool),
            'drop': torch.zeros(1, dtype=torch.bool)}


class RolloutMetricsTests(unittest.TestCase):
    def test_evaluation_preserves_portrait_source_geometry(self):
        source = {'latents': torch.empty(1, 48, 1, 30, 16)}
        self.assertEqual(_evaluation_geometry(source, {'height': 480, 'width': 256}), (480, 256))
        with self.assertRaises(ValueError):
            _evaluation_geometry(source, {'height': 256, 'width': 480})
        with self.assertRaises(ValueError):
            _evaluation_geometry({'latents': torch.empty(1, 48, 1, 16, 30)}, {})

    def test_eager_block_and_rng_restoration_on_generation_failure(self):
        net, source = FakeNet(), rows()
        previous, eager = object(), object()
        net.block_fn, net._block = previous, eager
        before = torch.random.get_rng_state().clone()
        def fail(*args, **kwargs):
            self.assertIs(net.block_fn, eager)
            self.assertFalse(net.training)
            raise RuntimeError('injected rollout failure')
        with patch('distillation.joint_dmd.generate_stabilizer', fail), self.assertRaises(RuntimeError):
            generate_student_rollout(net, source, {'latent_frames': 9, 'seed': 7})
        self.assertIs(net.block_fn, previous)
        self.assertTrue(net.training)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_duration_aware_tail_startup_and_missing_regions(self):
        seconds = np.arange(31)
        measured = {'m1': seconds.astype(float), 'm2': -seconds.astype(float), 'n_bg': np.ones(31)}
        measured['n_bg'][30] = 0
        measured['m1'][29] = float('nan')
        scalars, valid = summarize_drift(seconds, measured, duration=30.12, continuation_start=1.)
        self.assertEqual(set(scalars), set(DRIFT_KEYS))
        self.assertAlmostEqual(scalars['background_color_tail'], 27.)  # 26,27,28; final missing seconds excluded
        self.assertAlmostEqual(scalars['background_texture_tail'], -27.5)
        self.assertFalse(valid['m1']['valid'][0])
        self.assertEqual(valid['m1']['tail_valid_count'], 3)
        self.assertEqual(valid['m2']['tail_valid_count'], 4)

    def test_empty_background_stays_missing(self):
        scalars, valid = summarize_drift([0, 1], {'m1': [0, np.nan], 'm2': [0, np.inf], 'n_bg': [1, 0]},
                                        duration=2., continuation_start=1.)
        self.assertTrue(all(value is None for value in scalars.values()))
        self.assertEqual(valid['m1']['valid_count'], 0)
        json.dumps(scalars, allow_nan=False)

    def test_resolver_uses_generated_clean_history_without_renoising(self):
        net, source = FakeNet(), rows(5)
        stabilized = torch.tensor([100., 1., 2., 3., 4.]).view(1, 1, 5, 1, 1)
        value = resolve_generated_sequence(net, source, stabilized)
        self.assertEqual(float(value[0, 0, 0]), 100.)
        self.assertAlmostEqual(float(value[0, 0, 1]), .8, places=6)
        self.assertAlmostEqual(float(value[0, 0, 2]), 1.44, places=6)
        self.assertAlmostEqual(float(value[0, 0, 3]), 1.952, places=6)
        self.assertEqual([call[0][0].item() for call in net.calls], [1., 2., 3., 4.])
        self.assertFalse(net.calls[0][1][2])
        self.assertFalse(net.calls[0][2][2])
        self.assertAlmostEqual(net.calls[2][1][0].item(), .8, places=6)
        self.assertAlmostEqual(net.calls[2][2][0].item(), 1.44, places=6)
        changed = dict(source, latents=source['latents'].clone())
        changed['latents'][:, :, 1:] = -999.
        self.assertTrue(torch.equal(value, resolve_generated_sequence(FakeNet(), changed, stabilized)))

    def test_generation_assembles_prefix_and_preserves_rng_and_mode(self):
        net, source = FakeNet(), rows()
        before = torch.random.get_rng_state().clone()
        def generator(model, supplied, steps, **kwargs):
            self.assertTrue(kwargs['return_prefix'])
            self.assertEqual(kwargs['prefix_frames'], 6)
            self.assertEqual(kwargs['num_targets'], 2)
            self.assertEqual(kwargs['history_dropout'], ())
            self.assertFalse(model.training)
            noise = kwargs['noise']
            return noise[:, :, 6:], {'prefix': noise[:, :, :6]}
        with patch('distillation.joint_dmd.generate_stabilizer', generator):
            resolved, stabilized = generate_student_rollout(net, source, {'latent_frames': 9, 'seed': 7})
        self.assertEqual(resolved.shape, source['latents'].shape)
        self.assertEqual(stabilized.shape, source['latents'].shape)
        self.assertTrue(net.training)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_actual_generator_prefix_hook_preserves_default_outputs(self):
        from distillation.joint_dmd import generate_stabilizer
        source, noise = rows(), torch.randn(1, 1, 8, 1, 1)
        with torch.no_grad():
            expected, original_context = generate_stabilizer(FakeNet(), source, noise=noise,
                                                              num_targets=2, history_dropout=())
            actual, context = generate_stabilizer(FakeNet(), source, noise=noise,
                                                  num_targets=2, history_dropout=(), return_prefix=True)
        self.assertTrue(torch.equal(expected, actual))
        self.assertNotIn('prefix', original_context)
        self.assertEqual(context['prefix'].shape[2], 6)
        self.assertTrue(torch.equal(context['prefix'][:, :, -3:], context['history']))

    def test_decode_metrics_precede_lossy_encoding_and_original_is_lossless(self):
        import av
        latents = torch.zeros(1, 1, 3, 1, 1)
        def chunks(*_):
            yield torch.full((3, 1, 8, 8), .2)
            yield torch.full((3, 4, 8, 8), .2)
            yield torch.full((3, 4, 8, 8), .2)
        with tempfile.TemporaryDirectory() as directory, \
                patch('inference.bounded_decode.decoded_chunks', chunks):
            output = Path(directory)/'generated.mp4'
            result = _decode_video(None, latents, output, fps=2, audio_np=np.zeros(108_000),
                                   expected_hw=(8, 8), compare_rgb=np.full((9, 8, 8, 3), .5))
            self.assertEqual(result['frames'], 9)
            self.assertEqual(len(result['sample_frames']), 5)
            np.testing.assert_allclose(result['l2'], np.sqrt(8*8*3)*.1, atol=1e-6)
            with av.open(result['original_path']) as container:
                self.assertEqual(container.streams.audio[0].codec_context.name, 'pcm_s16le')
                decoded = np.stack([frame.to_ndarray(format='rgb24') for frame in container.decode(video=0)])
            self.assertEqual(decoded.shape, (9, 8, 8, 3))
            np.testing.assert_array_equal(decoded, np.full(decoded.shape, 153, dtype=np.uint8))

    def test_gallery_survives_source_retention_and_rejects_bad_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'checkpoint'
            source.mkdir()
            artifacts = {}
            for key in ('generated_video', 'gt_video', 'gt_l2_curve', 'background_curves', 'sensitivity_stabilizer'):
                path = source/f'{key}.png'
                path.write_bytes(key.encode())
                artifacts[key] = {'path': str(path), 'sha256': hashlib.sha256(key.encode()).hexdigest(), 'bytes': len(key)}
            manifest = {'status': 'complete', 'run_id': 'run-a', 'checkpoint_id': 'step-10',
                        'weights': 'raw', 'artifacts': artifacts}
            gallery = Path(directory)/'gallery'
            result = append_local_gallery(manifest, gallery)
            for path in source.iterdir():
                path.unlink()
            index = json.loads(Path(result['manifest_path']).read_text())
            for path in index['entries'][0]['artifacts'].values():
                self.assertTrue((gallery/path).exists())
            self.assertEqual(append_local_gallery(manifest, gallery), result)
            self.assertEqual(len(json.loads(Path(result['manifest_path']).read_text())['entries']), 1)
            # Reconstruct an orphan committed media directory after a crash.
            index['entries'] = []
            Path(result['manifest_path']).write_text(json.dumps(index))
            self.assertEqual(append_local_gallery(manifest, gallery), result)
            self.assertEqual(len(json.loads(Path(result['manifest_path']).read_text())['entries']), 1)
            manifest['artifacts']['generated_video']['sha256'] = '0'*64
            with self.assertRaises(ValueError):
                append_local_gallery(manifest, gallery)

    def test_gallery_labels_live_model_evaluation_without_inventing_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'plot.png'
            source.write_bytes(b'plot')
            identity = '12345678123456781234567812345678'
            manifest = {'status': 'complete', 'run_id': 'run-a', 'checkpoint_id': None,
                        'evaluation_id': identity, 'evaluation_kind': 'in_memory',
                        'student_updates': 400, 'rounds': 550, 'weights': 'raw',
                        'artifacts': {'gt_l2_curve': {'path': str(source), 'bytes': 4,
                                      'sha256': hashlib.sha256(b'plot').hexdigest()}}}
            result = append_local_gallery(manifest, root / 'gallery')
            entry, = json.loads(Path(result['manifest_path']).read_text())['entries']
            self.assertEqual(entry['id'], identity + '-raw')
            self.assertIsNone(entry['checkpoint_id'])
            self.assertEqual(entry['evaluation_id'], identity)
            self.assertEqual(entry['student_updates'], 400)
            self.assertIn('in-memory', Path(result['index_path']).read_text())
            self.assertEqual(append_local_gallery(manifest, root / 'gallery'), result)
            changed = {**manifest, 'student_updates': 401}
            with self.assertRaisesRegex(ValueError, 'different evaluation identity'):
                append_local_gallery(changed, root / 'gallery')


if __name__ == '__main__':
    unittest.main()
