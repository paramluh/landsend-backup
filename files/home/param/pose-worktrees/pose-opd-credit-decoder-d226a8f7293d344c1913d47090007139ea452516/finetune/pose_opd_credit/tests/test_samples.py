import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from finetune.pose_opd_credit.samples import sample_slots, save_samples
from finetune.pose_opd_credit.tests.test_rollout import FakeCachedEngine, seed_row
from finetune.pose_opd_credit.rollout import rollout_row


class SampleExportTests(unittest.TestCase):
    def test_selection_does_not_sample_or_export_dummies(self):
        state = np.random.get_state()
        self.assertEqual(sample_slots([[1, 2, 0], None, [1, 3, 1], [1, 4, 0]], 4, 2), [0, 2])
        self.assertEqual(np.random.get_state()[0], state[0])
        np.testing.assert_array_equal(np.random.get_state()[1], state[1])
        with self.assertRaises(ValueError):
            sample_slots([[1, 2, 0], None], 2, 2)

    def test_exact_physical_alignment_and_atomic_resume_archive(self):
        row = seed_row()
        row.update(item=[1, 2, 0], row_key='example', aref=json.dumps({'side': 'user'}),
                   conditioning={'mode': 'audio'}, dummy=False)
        rollout = rollout_row(FakeCachedEngine(), row, seed=12)
        wave = np.linspace(-.5, .5, 30 * 1920, dtype=np.float32)
        original_audio = rollout.audio.copy()
        with tempfile.TemporaryDirectory() as directory:
            kwargs = dict(update=1, payload=[(row, None)], rollouts=[rollout], waveforms={0: wave},
                          slots=[0], fps=12.5, source_identity='test')
            receipt = save_samples(directory, **kwargs)
            path = Path(receipt['manifest'])
            manifest = json.loads(path.read_text())
            self.assertEqual(manifest['samples'][0]['frames'], 30)
            with np.load(path.parent / manifest['samples'][0]['file'], allow_pickle=False) as data:
                np.testing.assert_array_equal(data['input_audio'], wave)
                self.assertEqual(data['audio_codes'].shape, (32, 30))
                np.testing.assert_array_equal(data['audio_codes'][:, :25], np.full((32, 25), 4))
                np.testing.assert_array_equal(data['audio_codes'][:, 25:], np.full((32, 5), 11))
                np.testing.assert_array_equal(data['pose_codes'][:, :25], np.full((32, 25), 5))
                np.testing.assert_array_equal(data['pose_codes'][:, 25:], np.full((32, 5), 12))
                self.assertEqual(data['sampled_audio_mask'].tolist(), [False] * 25 + [True] * 5)
                self.assertEqual(data['sampled_text_mask'].tolist(), [False] * 25 + [True] * 5)
            np.testing.assert_array_equal(rollout.audio, original_audio)
            save_samples(directory, **kwargs)
            histories = list((Path(directory) / 'raw-rollouts').glob('history-update-*'))
            self.assertEqual(len(histories), 1)
            self.assertTrue((histories[0] / 'manifest.json').is_file())

    def test_invalid_codes_never_commit_manifest(self):
        row = seed_row()
        row.update(item=[1, 2, 0], row_key='example', aref='{}', conditioning={})
        rollout = rollout_row(FakeCachedEngine(), row, seed=1)
        rollout.pose[-1] = -1
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                save_samples(directory, update=1, payload=[(row, None)], rollouts=[rollout],
                    waveforms={0: np.zeros(30 * 1920, np.float32)}, slots=[0], fps=12.5, source_identity='test')
            self.assertFalse(list((Path(directory) / 'raw-rollouts').glob('update-*/manifest.json')))


if __name__ == '__main__':
    unittest.main()
