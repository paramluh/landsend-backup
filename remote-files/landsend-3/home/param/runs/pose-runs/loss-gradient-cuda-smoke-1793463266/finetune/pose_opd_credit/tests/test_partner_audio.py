"""CPU invariants for append-only generated PCM and live frontend lookahead."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from finetune.pose_opd_credit.partner_audio import AudioTimeline, StreamingFront, _block_frontend_embedding
from finetune.pose_opd_credit.streaming import vox_stream as VS


class AudioTimelineTests(unittest.TestCase):
    def test_delay_shifts_complete_prefill_and_generated_waveform_without_boundary_gap(self):
        delay, frame_samples = 72, VS.SPF24
        prefixes = [np.linspace(1.1, 1.3, 2 * frame_samples, dtype=np.float32),
                    np.linspace(-.8, -.6, 3 * frame_samples, dtype=np.float32)]
        gains = [2., .5]
        timeline = AudioTimeline(prefixes, 6, 'cpu', inverse_gains=gains,
                                 delivery_delay_samples=delay)
        expected_parts = [[prefix] for prefix in prefixes]
        prefills = [len(prefix) // frame_samples for prefix in prefixes]
        for frame in range(2, 6):
            ramp = torch.linspace(.1 * frame, .1 * frame + .4, frame_samples)
            chunk = torch.stack((ramp, -ramp))
            timeline.append(frame, chunk)
            for row in range(2):
                if frame >= prefills[row]:
                    expected_parts[row].append((chunk[row] * gains[row]).clamp(-.995, .995).numpy())
                # Compare to an independently assembled source waveform plus
                # one initial delay, not per-chunk padding or frame shifting.
                source = np.concatenate(expected_parts[row])
                delivered = np.pad(source, (delay, 0))
                read_frames = max(prefills[row], frame + 1)
                np.testing.assert_array_equal(timeline.waveform(row, read_frames),
                                              delivered[:read_frames * frame_samples])
                self.assertEqual(timeline.available[row], len(source) + delay)
        for row in range(2):
            heard = timeline.waveform(row, 6)
            boundary = delay + len(prefixes[row])
            np.testing.assert_array_equal(heard[delay:boundary], prefixes[row])
            source = np.concatenate(expected_parts[row])
            np.testing.assert_array_equal(heard[boundary - delay:boundary + delay],
                                          source[len(prefixes[row]) - delay:len(prefixes[row]) + delay])

    def test_exact_delay_boundary_keeps_previous_native_frontend_window_unchanged(self):
        prefix_frames = 2
        prefix = np.full(prefix_frames * VS.SPF24, .25, np.float32)
        for delay in (0, 71, 72):
            with self.subTest(delay=delay):
                timeline = AudioTimeline([prefix], 5, 'cpu', inverse_gains=[1.],
                                         delivery_delay_samples=delay)
                previous_frame = prefix_frames - 1
                previous_indices = previous_frame * VS.SPF24 + timeline.indices
                current_indices = prefix_frames * VS.SPF24 + timeline.indices
                previous_before = timeline.buffer.index_select(1, previous_indices).clone()
                current_before = timeline.buffer.index_select(1, current_indices).clone()
                if delay == VS.LOOK24:
                    timeline.require(previous_frame)
                else:
                    with self.assertRaisesRegex(ValueError, 'ungenerated audio'):
                        timeline.require(previous_frame)
                timeline.append(prefix_frames, torch.full((1, VS.SPF24), .5))
                previous_after = timeline.buffer.index_select(1, previous_indices)
                current_after = timeline.buffer.index_select(1, current_indices)
                changed_samples = torch.count_nonzero(previous_after != previous_before).item()
                self.assertEqual(changed_samples, max(0, VS.LOOK24 - delay))
                self.assertFalse(torch.equal(current_before, current_after))
                timeline.require(previous_frame)
                if delay == VS.LOOK24:
                    timeline.require(prefix_frames)
                else:
                    with self.assertRaisesRegex(ValueError, 'ungenerated audio'):
                        timeline.require(prefix_frames)

    def test_delayed_timeline_rejects_gaps_and_rewrites(self):
        timeline = AudioTimeline([np.zeros(2 * VS.SPF24, np.float32)], 5, 'cpu',
                                 inverse_gains=[1.], delivery_delay_samples=72)
        self.assertEqual(timeline.available.tolist(), [2 * VS.SPF24 + 72])
        chunk = torch.zeros(1, VS.SPF24)
        with self.assertRaisesRegex(ValueError, 'Non-contiguous'):
            timeline.append(3, chunk)
        timeline.append(2, chunk)
        self.assertEqual(timeline.available.tolist(), [3 * VS.SPF24 + 72])
        with self.assertRaisesRegex(ValueError, 'Non-contiguous'):
            timeline.append(2, chunk)
        with self.assertRaisesRegex(ValueError, 'ungenerated audio'):
            timeline.require(3)

    def test_delivery_delay_must_be_an_integer_shorter_than_one_frame(self):
        prefix = [np.zeros(VS.SPF24, np.float32)]
        for delay in (-1, VS.SPF24, 72., True, None):
            with self.subTest(delay=delay), self.assertRaisesRegex(ValueError, 'delivery_delay_samples'):
                AudioTimeline(prefix, 5, 'cpu', inverse_gains=[1.], delivery_delay_samples=delay)

    def test_inverse_gain_live_bound_and_raw_prefill_preservation(self):
        prefixes = [np.full(2 * VS.SPF24, 1.2, np.float32),
                    np.full(3 * VS.SPF24, -.7, np.float32)]
        timeline = AudioTimeline(prefixes, 6, 'cpu', inverse_gains=[2., .5])
        frame = torch.full((2, VS.SPF24), .8)
        timeline.append(2, frame)
        self.assertEqual(timeline.available.tolist(), [3 * VS.SPF24, 3 * VS.SPF24])
        first = timeline.waveform(0, 3)
        np.testing.assert_array_equal(first[:2 * VS.SPF24], prefixes[0])
        np.testing.assert_array_equal(first[2 * VS.SPF24:], np.full(VS.SPF24, .995, np.float32))
        np.testing.assert_array_equal(timeline.waveform(1, 3), prefixes[1])
        timeline.append(3, -frame)
        np.testing.assert_array_equal(timeline.waveform(0, 4)[3 * VS.SPF24:],
                                      np.full(VS.SPF24, -.995, np.float32))
        np.testing.assert_array_equal(timeline.waveform(1, 4)[3 * VS.SPF24:],
                                      np.full(VS.SPF24, -.4, np.float32))

    def test_rejects_generation_gaps_rewrites_and_missing_three_ms(self):
        timeline = AudioTimeline([np.zeros(2 * VS.SPF24, np.float32)], 5, 'cpu', inverse_gains=[1.])
        timeline.require(0)
        with self.assertRaisesRegex(ValueError, 'ungenerated audio'):
            timeline.require(1)
        chunk = torch.zeros(1, VS.SPF24)
        with self.assertRaisesRegex(ValueError, 'Non-contiguous'):
            timeline.append(3, chunk)
        timeline.append(2, chunk)
        timeline.require(1)
        with self.assertRaisesRegex(ValueError, 'Non-contiguous'):
            timeline.append(2, chunk)
        with self.assertRaisesRegex(ValueError, 'exactly one frame'):
            timeline.append(3, chunk[:, :-1])

    def test_rejects_partial_source_frames_and_bad_gain_dimensions(self):
        with self.assertRaisesRegex(ValueError, 'complete 80 ms'):
            AudioTimeline([np.zeros(VS.SPF24 - 1, np.float32)], 5, 'cpu', inverse_gains=[1.])
        for gains in ([float('nan')], [1., 2.], [0.], [-1.]):
            with self.assertRaisesRegex(ValueError, 'finite .*recorded inverse gain'):
                AudioTimeline([np.zeros(VS.SPF24, np.float32)], 5, 'cpu', inverse_gains=gains)


class BlockConvolutionTests(unittest.TestCase):
    def test_short_benchmark_keeps_full_logical_cache_with_paged_storage(self):
        requested = []
        def fake_front(encoder, batch, frames, device, fa, fused_ops, extractor, **kwargs):
            requested.append((frames, kwargs))
            return SimpleNamespace(Hd=1280, maxT=frames * 4 + 8, cache_capacity_positions=1024,
                mel_prev=torch.zeros(batch, 128, 2, dtype=torch.bfloat16),
                c1_prev=torch.zeros(batch, 1280, 1, dtype=torch.bfloat16),
                K=[torch.zeros(8, 256, 1, 2, dtype=torch.bfloat16)],
                V=[torch.zeros(8, 256, 1, 2, dtype=torch.bfloat16)], reset=lambda: None)
        engine = SimpleNamespace(d0=torch.device('cpu'), NI=2, n_fr=1500)
        front = SimpleNamespace(encoder=None, fe=None)
        with patch('finetune.pose_opd_credit.partner_audio.VoxStreamFast', side_effect=fake_front):
            stream = StreamingFront(front, engine, None, fa=None, fused_ops=None, frames=75, graphs=False)
        self.assertEqual(requested, [(1500, {'cache_mode': 'paged', 'rewind_frames': 8, 'max_block_frames': 8})])
        report = stream.cache_report()
        self.assertEqual(report['voxtral_cache_capacity_frames'], 1500)
        self.assertEqual(report['voxtral_cache_encoder_positions'], 1024)
        self.assertEqual(report['recorded_capacity_frames'], 83)

    @torch.no_grad()
    def test_every_internal_snapshot_matches_sequential_convolution_boundaries(self):
        torch.manual_seed(419)
        count, batch, bins, hidden = 6, 2, 3, 5
        embedder = SimpleNamespace(conv1=torch.nn.Conv1d(bins, hidden, 3),
                                  conv2=torch.nn.Conv1d(hidden, hidden, 3, stride=2))
        def state():
            return SimpleNamespace(enc=SimpleNamespace(embedder=embedder),
                mel_prev=initial_mel.clone(), c1_prev=initial_conv.clone())
        initial_mel = torch.randn(batch, bins, 2)
        initial_conv = torch.randn(batch, hidden, 1)
        # Native embed() rounds mel to BF16 before concatenating its cache.
        mel = torch.randn(batch, bins, count * VS.MPF).to(torch.bfloat16)
        sequential, blocked = state(), state()
        expected, mel_states, conv_states = [], [], []
        for index in range(count):
            mel_states.append(sequential.mel_prev.clone())
            conv_states.append(sequential.c1_prev.clone())
            expected.append(VS.VoxStream.embed(sequential, mel[..., index * VS.MPF:(index + 1) * VS.MPF]))
        actual, block_mel, block_conv = _block_frontend_embedding(blocked, mel, count)
        torch.testing.assert_close(block_mel, torch.stack(mel_states), rtol=0, atol=0)
        torch.testing.assert_close(block_conv, torch.stack(conv_states), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(actual, torch.cat(expected, 1), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(blocked.mel_prev, sequential.mel_prev, rtol=0, atol=0)
        torch.testing.assert_close(blocked.c1_prev, sequential.c1_prev, rtol=1e-5, atol=1e-6)
        for boundary in range(count):
            restored = state()
            restored.mel_prev.copy_(block_mel[boundary])
            restored.c1_prev.copy_(block_conv[boundary])
            suffix = [VS.VoxStream.embed(restored, mel[..., i * VS.MPF:(i + 1) * VS.MPF])
                      for i in range(boundary, count)]
            torch.testing.assert_close(torch.cat(suffix, 1), torch.cat(expected[boundary:], 1), rtol=1e-5, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
