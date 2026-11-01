import copy
from types import SimpleNamespace
import unittest

import numpy as np

from finetune.pose_opd_credit.teacher_partner_benchmark import _audit_result


class CausalFingerprintTests(unittest.TestCase):
    def fixture(self):
        masks = {key: np.array([False, True, True, True]) for key in ('text', 'audio', 'pose')}
        probabilities = {key: np.array([np.nan, -.2, -.3, -.4], np.float32) for key in masks}
        rollout = SimpleNamespace(conversation_frames=4, prefill_frames=1,
            text=np.arange(4), audio=np.zeros((4, 32), np.int64),
            pose=np.ones((4, 32), np.int64), sampled_masks=masks,
            behavior_logprobs=probabilities)
        result = {'rollouts': [rollout], 'waveforms': [np.zeros(16, np.float32)],
            'latents': [np.zeros((4, 8), np.float32)],
            'partner_rollouts': [{'lane_b': np.arange(4), 'dep_b': np.zeros((4, 32), np.int64)}]}
        return result, [{'conversation_frames': 4, 'prefill_frames': 1}]

    def test_input_changes_are_detected_when_student_tokens_match(self):
        source, rows = self.fixture()
        original = _audit_result(source, rows)
        for field in ('waveforms', 'latents', 'partner_rollouts'):
            changed = copy.deepcopy(source)
            target = changed[field][0]
            if isinstance(target, dict):
                target = target['lane_b']
            target.flat[0] = 1
            audit = _audit_result(changed, rows)
            self.assertEqual(original['rollout_sha256'], audit['rollout_sha256'])
            self.assertNotEqual(original['causal_sha256'][field], audit['causal_sha256'][field])

    def test_behavior_probabilities_are_part_of_the_contract(self):
        source, rows = self.fixture()
        changed = copy.deepcopy(source)
        changed['rollouts'][0].behavior_logprobs['audio'][1] = -.25
        self.assertNotEqual(_audit_result(source, rows)['causal_sha256']['student_masks_and_behavior'],
            _audit_result(changed, rows)['causal_sha256']['student_masks_and_behavior'])

    def test_identical_outputs_with_unscored_nans_have_identical_fingerprints(self):
        source, rows = self.fixture()
        self.assertEqual(_audit_result(source, rows), _audit_result(copy.deepcopy(source), rows))


if __name__ == '__main__':
    unittest.main()
