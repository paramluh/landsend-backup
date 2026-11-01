"""Independent behavioral checks for the frozen v5 detector, without retuning."""
import json
from pathlib import Path
import unittest

import numpy as np

from strict_detection import evaluate, prepare, scan
from test_detection import base_clip


def events(p):
    return scan(p, prefill_seconds=0)['events']


class IndependentStrictControls(unittest.TestCase):
    def test_smooth_fast_monotonic_transitions(self):
        t = np.arange(500)-250
        for width in (1.5, 2., 3., 6.):
            for vector in ((12, 0), (0, -18), (8, 12)):
                with self.subTest(width=width, vector=vector):
                    p = base_clip()
                    transition = (1+np.tanh(t/width))/2
                    p[:, :7, :2] += transition[:, None, None]*np.asarray(vector)/[256, 480]
                    self.assertEqual(events(p), [])

    def test_sustained_jitter_with_and_without_extra_step(self):
        for seed, sigma in ((31, 1.), (53, 2.), (79, 4.)):
            for step in (0, 8):
                with self.subTest(seed=seed, sigma=sigma, step=step):
                    p = base_clip()
                    noise = np.random.default_rng(seed).normal(0, sigma, (len(p), 1, 2))
                    p[:, :7, :2] += noise/[256, 480]
                    p[250:, :7, 1] += step/480
                    self.assertEqual(events(p), [])

    def test_single_frame_excursion_returns(self):
        for amplitude in (4, 12, 24):
            with self.subTest(amplitude=amplitude):
                p = base_clip();p[250, :7, 1] += amplitude/480
                self.assertEqual(events(p), [])

    def test_continuous_position_with_velocity_corner(self):
        t = np.arange(500)-249
        for before, after in ((0., .2), (.05, .8), (-.1, .5)):
            with self.subTest(before=before, after=after):
                p = base_clip()
                displacement = before*np.minimum(t, 0)+after*np.maximum(t, 0)
                p[:, :7, 1] += displacement[:, None]/480
                self.assertEqual(events(p), [])

    def test_true_step_with_new_post_transition_velocity(self):
        for frame in (180, 250, 340):
            for vector in ((6, 0), (0, -6), (4, 4)):
                with self.subTest(frame=frame, vector=vector):
                    p = base_clip();t = np.arange(len(p))-frame
                    p[frame:, :7, :2] += np.asarray(vector)/[256, 480]
                    p[:, :7, :2] += np.maximum(t, 0)[:, None, None]*np.asarray([.10, -.15])/[256, 480]
                    self.assertIn(frame, [x['frame'] for x in events(p)])

    def test_three_visible_head_anchors_are_not_diluted(self):
        p = base_clip();p[:, 3:5, 2] = 0
        p[250:, :3, 1] += 2.8/480;p[250:, 5:7, 1] += 2/480
        self.assertTrue(evaluate(prepare(p), 250)['passes'])
        self.assertIn(250, [x['frame'] for x in events(p)])
        # The same invisible anchors must not contaminate local fitting.
        p[:, 3:5, :2] += np.random.default_rng(91).normal(0, .03, (len(p), 2, 2))
        self.assertIn(250, [x['frame'] for x in events(p)])

    def test_equal_two_frame_transition_is_not_single_frame_dominated(self):
        p = base_clip();p[250:, :7, 1] += 6/480;p[251:, :7, 1] += 6/480
        self.assertEqual(events(p), [])

    def test_dominant_first_frame_with_short_settling_is_retained(self):
        p = base_clip();p[250:, :7, 1] += 8/480;p[251:, :7, 1] += 2/480
        self.assertIn(250, [x['frame'] for x in events(p)])

    def test_user_confirmed_controls(self):
        base = Path('/home/param/runs/pose-runs')
        controls = [
            (base/'pose-sites-20261030/video-inspection/v1-19-21/pose_overlay.json', 512, True),
            (base/'pose-jump-audit-suite-20261031/independent-qa/rollout-000243/pose_overlay.json', 292, True),
            (base/'pose-jump-audit-20261031-nocfg/positive-qa/runs/rollout-000145/pose_overlay.json', 474, False),
        ]
        for path, frame, expected in controls:
            with self.subTest(path=path, frame=frame):
                if not path.exists():
                    self.skipTest(f'Local control artifact absent: {path}')
                p = json.loads(path.read_text())['frames']
                self.assertEqual(evaluate(prepare(p), frame)['passes'], expected)
                matches = [r for r in events(p) if abs(r['frame']-frame) <= 2]
                self.assertEqual(bool(matches), expected)


if __name__ == '__main__':
    unittest.main()
