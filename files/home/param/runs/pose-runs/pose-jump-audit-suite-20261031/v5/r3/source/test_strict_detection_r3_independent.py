"""Independent controls for r3's destination-confidence and one-cut contract."""
import json
from pathlib import Path
import unittest

import numpy as np

from strict_detection_r3 import evaluate, prepare, scan
from test_detection import base_clip


def found(p, frame=250):
    return any(abs(e['frame']-frame) <= 1 for e in scan(p, prefill_seconds=0)['events'])


class IndependentR3Controls(unittest.TestCase):
    def test_low_source_is_allowed_with_confident_geometric_destination(self):
        for before in (0., .05, .19, .39):
            for after in (.4, .6, .8):
                with self.subTest(before=before, after=after):
                    p = base_clip();p[:250, :7, 2] = before;p[250:, :7, 2] = after
                    p[250:, :7, :2] += [8/256, -5/480]
                    self.assertTrue(found(p))

    def test_confidence_recovery_without_coordinate_cut_is_not_an_event(self):
        p = base_clip();p[:250, :7, 2] = .01
        self.assertFalse(found(p))
        p[250:, 13:, 1] -= .4;p[:250, 13:, 2] = .01
        self.assertFalse(found(p))

    def test_low_destination_is_excluded_even_for_large_coordinate_cut(self):
        for after in (.05, .2, .35, .399999):
            with self.subTest(after=after):
                p = base_clip();p[250:, :7, 2] = after;p[250:, :7, 1] += 20/480
                self.assertFalse(found(p))

    def test_confidence_gate_checks_each_shoulder_and_three_head_anchors(self):
        p = base_clip();p[250:, :7, 1] += 12/480;p[250:, 6, 2] = .399999
        self.assertFalse(found(p))
        p = base_clip();p[250:, :7, 1] += 12/480;p[250:, 2:5, 2] = .35
        self.assertFalse(found(p))
        p[250:, 2, 2] = .4
        self.assertTrue(found(p))

    def test_destination_subset_gating_and_fits_ignore_nonvoting_head_anchors(self):
        p = base_clip();p[:, 3:5, 2] = .05
        p[250:, :3, 1] += 5/480;p[250:, 5:7, 1] += 2/480
        p[:, 3:5, :2] += np.random.default_rng(173).normal(0, .025, (len(p), 2, 2))
        self.assertTrue(evaluate(prepare(p), 250)['passes'])
        self.assertTrue(found(p))
        p[250:, :3, 1] -= 5/480;p[250:, 5:7, 1] -= 2/480
        self.assertFalse(found(p))

    def test_low_confidence_noisy_history_does_not_bypass_geometry(self):
        p = base_clip();p[:250, :7, 2] = .05
        p[:, :7, :2] += np.random.default_rng(211).normal(0, [2/256, 2/480], (len(p), 1, 2))
        p[250:, :7, 1] += 16/480
        self.assertFalse(found(p))

    def test_smooth_motion_before_and_after_cut_does_not_require_a_plateau(self):
        for frame in (180, 250, 340):
            p = base_clip();t = np.arange(len(p))-frame
            p[:, :7, 0] += (.12*t)[:, None]/256
            p[frame:, :7, :2] += [8/256, -5/480]
            p[:, :7, 1] += (-.08*np.maximum(t, 0))[:, None]/480
            self.assertTrue(found(p, frame))

    def test_smooth_fast_ramp_and_continuous_velocity_corner_reject(self):
        t = np.arange(500)-250
        for width in (1.5, 2., 4.):
            p = base_clip();p[:, :7, 1] += (.05*(1+np.tanh(t/width))/2)[:, None]
            self.assertFalse(found(p))
        p = base_clip();p[:, :7, 1] += (.5*np.maximum(t+1, 0))[:, None]/480
        self.assertFalse(found(p))

    def test_impulse_and_equal_two_frame_transition_reject(self):
        p = base_clip();p[250, :7, 1] += 20/480
        self.assertFalse(found(p))
        p = base_clip();p[250:, :7, 1] += 8/480;p[251:, :7, 1] += 8/480
        self.assertFalse(found(p))

    def test_real_controls_preserve_human_label_and_filter_distinction(self):
        base = Path('/home/param/runs/pose-runs')
        cases = [
            (base/'pose-sites-20261030/video-inspection/v1-19-21/pose_overlay.json', 512, True),
            (base/'pose-jump-audit-suite-20261031/independent-qa/rollout-000243/pose_overlay.json', 292, False),
            (base/'pose-jump-audit-20261031-nocfg/positive-qa/runs/rollout-000145/pose_overlay.json', 474, False),
        ]
        for path, frame, expected in cases:
            with self.subTest(path=path):
                if not path.exists():self.skipTest(f'Local control absent: {path}')
                p = json.loads(path.read_text())['frames']
                row = evaluate(prepare(p), frame)
                self.assertEqual(row['passes'], expected)
                self.assertEqual(found(p, frame), expected)
                if frame == 292:
                    # Human-positive243 remains a visual positive; this newer
                    # precision filter deliberately excludes its orange endpoints.
                    self.assertIn('destination_not_medium_high_confidence_for_head_and_both_shoulders', row['rejection_reasons'])


if __name__ == '__main__':
    unittest.main()
