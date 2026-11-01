import json
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from finetune.pose_opd_credit.samples_decode import (
    CPUCodecs, lane_transcript, pose_plane, publish_update, read_sample,
)


class SamplesDecodeTests(unittest.TestCase):
    def sample(self, directory, index=0):
        item = {'id': f'u1-r0-s{index}', 'file': f'sample-{index}.npz', 'frames': 4,
                'fps': 12.5, 'prefill_frames': 2, 'row_id': f'actual-row-{index}'}
        arrays = {'input_audio': np.zeros(4 * 1920, np.float32),
                  'text_ids': np.array([1, 151643, 2, 3], np.int64),
                  'audio_codes': np.tile(np.arange(4, dtype=np.int64), (32, 1)),
                  'pose_codes': np.ones((32, 4), np.int64)}
        arrays['input_audio'][:2] = [1.25, -1.25]
        arrays.update({f'sampled_{stream}_mask': np.array([False, False, True, True])
                       for stream in ('text', 'audio', 'pose')})
        np.savez(directory / item['file'], **arrays)
        return item, arrays

    def test_physical_timeline_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            item, arrays = self.sample(directory)
            loaded = read_sample(directory, item)
            np.testing.assert_array_equal(loaded['audio_codes'], arrays['audio_codes'])
            arrays['audio_codes'][2, 1] = 4096
            np.savez(directory / item['file'], **arrays)
            with self.assertRaisesRegex(ValueError, 'local IDs'):
                read_sample(directory, item)
            item['file'] = '../other.npz'
            with self.assertRaisesRegex(ValueError, 'Unsafe'):
                read_sample(directory, item)

    def test_committed_file_hash_and_size_are_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            item, _ = self.sample(directory)
            path = directory / item['file']
            item.update(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            read_sample(directory, item)
            item['sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                read_sample(directory, item)
            item['bytes'] += 1
            with self.assertRaisesRegex(ValueError, 'size'):
                read_sample(directory, item)

    def test_confidence_is_not_depth_and_parents_integrate(self):
        features = np.zeros((2, 55, 3), np.float32)
        features[:, 0] = [1, 2, .7]
        features[:, 1] = [3, 4, .2]
        parents = np.full(55, -1)
        parents[1] = 0
        result = pose_plane(features, parents)
        np.testing.assert_allclose(result['frames'][0][:2], [[2, -1, 0], [6, -4, 0]])
        self.assertEqual(result['dimensionality'], 2)
        self.assertEqual(result['units'], 'shoulder_widths')
        self.assertAlmostEqual(result['confidence'][0][0], .7)

    def test_codec_receives_entire_prefilled_conversation_once(self):
        codec = CPUCodecs.__new__(CPUCodecs)
        codec.torch = torch
        calls = []
        def decode_audio(codes):
            calls.append(codes.clone())
            return SimpleNamespace(audio_values=torch.zeros(1, 1, codes.shape[-1] * 1920))
        codec.mimi = SimpleNamespace(decode=decode_audio)
        codec.pose = SimpleNamespace(decode=lambda codes: torch.zeros(1, 2 * codes[0].shape[-1], 165))
        codec.parents = np.full(55, -1)
        codec.text = SimpleNamespace(decode=lambda ids: ''.join(map(str, ids)), convert_ids_to_tokens=lambda token: str(token))
        arrays = {'audio_codes': np.tile(np.arange(4), (32, 1)),
                  'pose_codes': np.zeros((32, 4), np.int64), 'text_ids': np.array([1, 151643, 2, 3])}
        audio, pose, text = codec.decode(arrays)
        self.assertEqual(len(calls), 1)
        np.testing.assert_array_equal(calls[0][0], arrays['audio_codes'])
        self.assertEqual(len(audio), 7680)
        self.assertEqual(len(pose['frames']), 8)
        self.assertEqual(text, '1 23')

    def test_four_real_samples_publish_atomically_and_idempotently(self):
        class FakeCodecs:
            metadata = {'device': 'cpu'}
            calls = 0
            def decode(self, arrays):
                self.calls += 1
                return arrays['input_audio'], pose_plane(np.zeros((8, 55, 3)), np.full(55, -1)), 'actual transcript'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / 'raw', root / 'web'
            source.mkdir()
            items = [self.sample(source, i)[0] for i in range(4)]
            manifest = source / 'manifest.json'
            manifest.write_text(json.dumps({'schema_version': 1, 'synthetic': False, 'update': 1, 'samples': items}))
            codec = FakeCodecs()
            entry = publish_update(manifest, output, codec)
            self.assertEqual(codec.calls, 4)
            public = json.loads((output / 'manifest.json').read_text())
            self.assertFalse(public['synthetic'])
            self.assertEqual(len(public['rollouts'][0]['samples']), 4)
            for sample in entry['samples']:
                for key in ('input_audio_url', 'output_audio_url', 'pose_url'):
                    self.assertTrue((output / sample[key]).is_file())
            import soundfile as sf
            audio, rate = sf.read(output / entry['samples'][0]['input_audio_url'], dtype='float32')
            self.assertEqual(rate, 24000)
            np.testing.assert_array_equal(audio[:2], [1.25, -1.25])
            publish_update(manifest, output, codec)
            self.assertEqual(codec.calls, 4)
            doc = json.loads(manifest.read_text())
            doc['samples'][0]['row_id'] = 'changed'
            manifest.write_text(json.dumps(doc))
            replaced = publish_update(manifest, output, codec)
            self.assertEqual(codec.calls, 8)
            self.assertNotEqual(replaced['samples'][0]['pose_url'], entry['samples'][0]['pose_url'])
            self.assertTrue((output / entry['samples'][0]['pose_url']).exists())
            public = json.loads((output / 'manifest.json').read_text())
            self.assertEqual(len(public['rollouts']), 1)
            self.assertEqual(public['rollouts'][0]['samples'][0]['row_id'], 'changed')

    def test_failure_cannot_publish_partial_update(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            items = [self.sample(root, i)[0] for i in range(4)]
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'schema_version': 1, 'synthetic': False, 'update': 1, 'samples': items}))
            def fail(arrays):
                raise RuntimeError('codec failed')
            with self.assertRaisesRegex(RuntimeError, 'codec failed'):
                publish_update(manifest, root / 'out', SimpleNamespace(decode=fail, metadata={}))
            self.assertFalse((root / 'out/manifest.json').exists())
            self.assertEqual(list((root / 'out/assets').iterdir()), [])

    def test_missing_text_vocabulary_is_an_error_not_a_blank_transcript(self):
        tokenizer = SimpleNamespace(convert_ids_to_tokens=lambda token: None)
        with self.assertRaisesRegex(ValueError, 'missing executable token'):
            lane_transcript([20], tokenizer)


if __name__ == '__main__':
    unittest.main()
