"""CPU checks for replica assignment, genuine host concurrency and PCM ordering."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from finetune.pose_opd_credit.partner_audio import AudioTimeline
from finetune.pose_opd_credit.partner_fanout import (
    _dispatch, _merge_audio, _Partner, _Profile, partner_partitions,
)


class PartnerFanoutTests(unittest.TestCase):
    def test_worker_preserves_inference_mode_for_buffers_created_by_caller(self):
        # The benchmark initializes buffers in inference_mode on its caller
        # thread. Worker-local no_grad alone cannot mutate those tensors.
        with torch.inference_mode():
            timeline = AudioTimeline([np.zeros(2 * 1920, np.float32)], 10, 'cpu',
                                     inverse_gains=[1.])
        worker = object.__new__(_Partner)
        worker.context = nullcontext
        worker.streams = {'cpu': SimpleNamespace(wait_event=lambda event: None)}
        worker.engine = SimpleNamespace(d0='cpu')
        worker.front = None
        worker.profile = _Profile(0)
        worker.timeline = timeline
        worker.ready = lambda: 'ready'
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(worker.refresh, 2, torch.full((1, 1920), .25),
                                 None, None, 'block').result()
        self.assertEqual(result, (None, 'ready'))
        np.testing.assert_array_equal(timeline.waveform(0, 3)[-1920:], np.full(1920, .25, np.float32))

    def test_partitions_cover_every_row_once_for_all_layouts(self):
        for counts in ([48], [24, 24], [16, 16, 16], [15, 16, 17]):
            parts = partner_partitions(48, counts)
            selected = [list(range(48))[part] for part in parts]
            self.assertEqual([len(part) for part in selected], counts)
            self.assertEqual([row for part in selected for row in part], list(range(48)))
        for count, partners in ((0, []), (48, []), (48, [16, 16]), (48, [48, 0]),
                                 (48, [16, -16, 48]), (48, [24., 24])):
            with self.subTest(count=count, partners=partners), self.assertRaises(ValueError):
                partner_partitions(count, partners)

    def test_all_partner_refreshes_start_before_waiting_and_results_keep_order(self):
        barrier = threading.Barrier(3)
        def refresh(index):
            barrier.wait(timeout=3)
            return index, threading.get_ident()
        with ThreadPoolExecutor(max_workers=3) as pool:
            results = _dispatch(pool, refresh, range(3))
        self.assertEqual([index for index, _ in results], [0, 1, 2])
        self.assertEqual(len({thread for _, thread in results}), 3)
        self.assertEqual(_dispatch(None, lambda value: value * 2, range(3)), [0, 2, 4])

    def test_worker_failure_is_propagated(self):
        def refresh(index):
            if index == 1:
                raise RuntimeError('partner failed')
            return index
        with ThreadPoolExecutor(max_workers=3) as pool:
            with self.assertRaisesRegex(RuntimeError, 'partner failed'):
                _dispatch(pool, refresh, range(3))

    def test_profile_disabled_invokes_directly_without_event_or_host_timers(self):
        with patch('finetune.pose_opd_credit.partner_fanout._AudioProfile') as instrument:
            profile = _Profile(0)
            marker = object()
            self.assertIs(profile.run('test', 'unused', lambda: marker), marker)
            self.assertEqual(profile.result(), {})
            instrument.assert_not_called()

    def test_fanout_audio_equals_independent_rows_with_mixed_prefills_and_gains(self):
        parts = partner_partitions(6, [2, 2, 2])
        prefills = [2, 3, 2, 3, 2, 3]
        source = [np.full(count * 1920, -.1 * (index + 1), np.float32)
                  for index, count in enumerate(prefills)]
        gains = [2., .5, 1., 3., .2, 1.5]
        combined = AudioTimeline(source, 1500, 'cpu', inverse_gains=gains)
        separate = [AudioTimeline([wave], 1500, 'cpu', inverse_gains=[gain])
                    for wave, gain in zip(source, gains, strict=True)]
        target = torch.empty(6, 1920)
        for frame in range(2, 5):
            wave = torch.arange(1, 7, dtype=torch.float32)[:, None].expand(-1, 1920) * frame / 10
            outputs = [(wave[part].clone(), None) for part in parts]
            merged = _merge_audio(target, parts, outputs)
            torch.testing.assert_close(merged, wave, rtol=0, atol=0)
            combined.append(frame, merged)
            for index, timeline in enumerate(separate):
                timeline.append(frame, wave[index:index + 1])
                np.testing.assert_array_equal(combined.waveform(index, frame + 1),
                                              timeline.waveform(0, frame + 1))
        self.assertEqual(combined.frames, 1500)
        for index, count in enumerate(prefills):
            np.testing.assert_array_equal(combined.waveform(index, count), source[index])


if __name__ == '__main__':
    unittest.main()
