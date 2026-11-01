"""Listening exports preserve the received waveform and label speaker channels."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.io.wavfile import read

from finetune.pose_opd_credit.config import TrainingConfig
from finetune.pose_opd_credit.conversation_metrics import empty_conversation_metrics
from finetune.pose_opd_credit.trial_audio import save_trial_audio


class TrialAudioTests(unittest.TestCase):
    def test_export_preserves_float_pcm_prefill_and_all_pose_codes(self):
        frames = 6
        row = {'conversation_frames': frames, 'prefill_frames': 2, 'row_key': 'real/source/c00',
               'item': (1, 2, 0), 'conditioning': {'clip_frames': 125},
               'partner': {'conditioning': {'clip_frames': 0}}}
        student = np.linspace(-1.03, .81, frames * 1920, dtype=np.float32)
        partner = np.linspace(.23, -.61, frames * 1920, dtype=np.float32)
        pose = np.arange((frames + 2) * 32).reshape(frames + 2, 32)
        rollout = SimpleNamespace(text=np.arange(frames + 2), audio=pose + 1, pose=pose,
            sampled_masks={key: np.arange(frames + 2) >= 2 for key in ('text', 'audio', 'pose')})
        result = {'rollouts': [rollout], 'student_waveforms': [student], 'waveforms': [partner],
            'partner_rollouts': [{'lane_b': rollout.text, 'dep_b': rollout.audio}],
            'delivery_delay_samples': 72, 'conversation_metrics': [empty_conversation_metrics()],
            'partner_sampling': 'natural'}
        with tempfile.TemporaryDirectory() as folder:
            manifest = save_trial_audio(result, [row], [32], TrainingConfig(), folder, row_offset=4)
            entry = manifest['samples'][0]
            sample_rate, actual = read(Path(folder) / entry['audio'])
            self.assertEqual(sample_rate, 24000)
            np.testing.assert_array_equal(actual[:, 0], student)
            np.testing.assert_array_equal(actual[:, 1], partner)
            self.assertEqual(entry['id'], 'conversation-05')
            self.assertEqual(entry['prefill_seconds'], .16)
            self.assertTrue(entry['pose_generated'])
            with np.load(Path(folder) / entry['tokens'], allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['student_pose'], pose)
            self.assertEqual(json.loads((Path(folder) / 'manifest.json').read_text()), manifest)


if __name__ == '__main__':
    unittest.main()
