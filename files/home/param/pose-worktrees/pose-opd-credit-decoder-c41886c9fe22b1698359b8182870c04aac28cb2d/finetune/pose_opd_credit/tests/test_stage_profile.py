"""Bounded actor profiling preserves replay calls and removes all wrappers."""
import contextlib
import unittest
from unittest import mock

import torch

from finetune.pose_opd_credit.rollout import _StageProfile


class StageProfileTests(unittest.TestCase):
    def test_events_measure_stages_then_unwrap_without_more_frame_overhead(self):
        clock, records = [0.], [0]

        class Event:
            def __init__(self, enable_timing):
                self.timing = enable_timing
            def record(self):
                self.value = clock[0]
                records[0] += 1
            def elapsed_time(self, other):
                return other.value - self.value

        class Graph:
            def __init__(self, duration):
                self.duration, self.calls = duration, 0
            def replay(self):
                clock[0] += self.duration
                self.calls += 1

        names = ['text', 'speech_full', 'pose_full', 'seg0', 'seg1']
        graphs = {name: Graph(i + 1.) for i, name in enumerate(names)}
        originals = dict(graphs)
        devices = {name: 'cuda:' + str(i % 2) for i, name in enumerate(names)}
        with mock.patch.object(torch.cuda, 'Event', Event), \
             mock.patch.object(torch.cuda, 'device', side_effect=lambda _: contextlib.nullcontext()), \
             mock.patch.object(torch.cuda, 'synchronize') as synchronize:
            profile = _StageProfile(graphs, devices, 2)
            for _ in range(2):
                for name in names:
                    graphs[name].replay()
            self.assertEqual(graphs, originals)
            event_count = records[0]
            for name in names:
                graphs[name].replay()
            self.assertEqual(records[0], event_count)
            synchronize.assert_not_called()
            result = profile.collect()
            self.assertTrue(result['completed'])
            self.assertEqual(synchronize.call_count, 2)
            self.assertIs(result, profile.collect())
            self.assertEqual(synchronize.call_count, 2)
        for i, name in enumerate(names):
            self.assertEqual(originals[name].calls, 3)
            self.assertEqual(result['stages'][name]['samples'], 2)
            self.assertEqual(result['stages'][name]['gpu_ms_mean'], i + 1.)
            self.assertEqual(result['stages'][name]['gpu_ms_total'], 2 * (i + 1.))
        self.assertFalse(profile.events)

    def test_requires_captured_graphs_and_positive_bounded_sample(self):
        for frames in (0, -1, 1.5, True):
            with self.assertRaisesRegex(ValueError, 'positive frame'):
                _StageProfile({}, {}, frames)
        with self.assertRaisesRegex(ValueError, 'Capture'):
            _StageProfile({}, {'text': 'cuda:0'}, 32)


if __name__ == '__main__':
    unittest.main()
