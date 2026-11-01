"""Physical waveform dependencies and irreversible partner-output commitment."""
import unittest
import json
from pathlib import Path

import numpy as np

from finetune.pose_opd_credit.teacher_partner_schedule import (
    AudioTiming, CommittedAudioTimeline, bootstrap_plan, plan_for_student, reference_rollout,
)
from finetune.pose_opd_credit.config import TrainingConfig


class TeacherPartnerScheduleTests(unittest.TestCase):
    def test_actual_three_ms_requires_six_replayed_positions_but_one_new_draw(self):
        # S[50] has just arrived. S[51] needs T[52] and the first72samples of T[53].
        plan = plan_for_student(51, 38)
        self.assertEqual(plan.newly_real_student_frame, 50)
        self.assertEqual(plan.teacher_new_audio_frame, 53)
        self.assertEqual(plan.teacher_sample_model_frame, 55)
        self.assertEqual(list(plan.teacher_backbone.frames()), list(range(49, 55)))
        self.assertEqual(list(plan.teacher_forced_tower.frames()), list(range(50, 55)))
        self.assertEqual(len(plan.teacher_tower), 6)
        self.assertEqual(plan.student_real_stop, 51 * 1920)
        self.assertEqual(plan.student_heard_stop, 53 * 1920 + 72)
        self.assertEqual(plan.teacher_input_stop, 55 * 1920 + 72)

    def test_removing_lookahead_still_needs_four_backbone_positions(self):
        plan = plan_for_student(51, 38, AudioTiming(lookahead_samples=0))
        self.assertEqual(plan.teacher_new_audio_frame, 52)
        self.assertEqual(list(plan.teacher_backbone.frames()), list(range(50, 54)))
        self.assertEqual(len(plan.teacher_forced_tower), 3)
        self.assertEqual(plan.student_heard_stop, 53 * 1920)

    def test_delivery_delay_removes_both_extra_positions_without_removing_lookahead(self):
        timing = AudioTiming(delivery_delay_samples=72)
        plan = plan_for_student(51, 38, timing)
        self.assertEqual(timing.lookahead_samples, 72)
        self.assertEqual(plan.teacher_new_audio_frame, 52)
        self.assertEqual(plan.teacher_sample_model_frame, 54)
        self.assertEqual(list(plan.teacher_backbone.frames()), list(range(50, 54)))
        self.assertEqual(list(plan.teacher_forced_tower.frames()), list(range(51, 54)))
        self.assertEqual(plan.student_real_stop, 51 * 1920)
        self.assertEqual(plan.teacher_received_real_stop, 51 * 1920 + 72)
        self.assertEqual(plan.student_heard_stop, 53 * 1920 + 72)
        self.assertEqual(plan.student_source_stop, 53 * 1920)
        self.assertEqual(plan.teacher_input_stop, 54 * 1920 + 72)
        self.assertEqual(plan.teacher_source_stop, 54 * 1920)

    def test_delivery_delay_boundary_is_exactly_72_samples(self):
        for delay, count in ((0, 6), (1, 6), (71, 6), (72, 4), (73, 4), (144, 4), (1919, 4)):
            with self.subTest(delay=delay):
                timing = AudioTiming(delivery_delay_samples=delay)
                plan = plan_for_student(51, 38, timing)
                self.assertEqual(len(plan.teacher_backbone), count)
                # The native frontend endpoint never changes or loses lookahead.
                self.assertEqual(plan.student_heard_stop, 53 * 1920 + 72)
                self.assertEqual(plan.student_source_stop + delay, plan.student_heard_stop)

    def test_delivery_delay_validation_and_selected_recipe(self):
        for value in (-1, 1920, 1.5, True, None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, 'delivery_delay_samples'):
                    AudioTiming(delivery_delay_samples=value)
                with self.assertRaisesRegex(ValueError, 'partner_audio_delay_samples'):
                    TrainingConfig(partner_audio_delay_samples=value).validate()
        self.assertEqual(TrainingConfig().partner_audio_delay_samples, 0)
        path = Path(__file__).resolve().parents[1] / 'recipe.teacher_partner.json'
        config = TrainingConfig.from_dict(json.loads(path.read_text()))
        self.assertEqual(config.partner_audio_delay_samples, 72)
        self.assertEqual(config.to_dict()['partner_audio_delay_samples'], 72)

    def test_refresh_is_the_minimal_interval_covering_changed_dependencies(self):
        for look, delay in ((0, 0), (0, 72), (1, 0), (72, 0), (72, 71),
                            (72, 72), (72, 144), (1919, 0), (1919, 1919)):
            timing = AudioTiming(lookahead_samples=look, delivery_delay_samples=delay)
            for i in range(39, 65):
                plan = plan_for_student(i, 38, timing)
                changed_sample = (i - 1) * timing.frame_samples + delay
                first = plan.teacher_backbone.start
                # The preceding frame's frontend cannot see the changed sample.
                self.assertLessEqual(first * 1920 + look, changed_sample)
                self.assertGreater((first + 1) * 1920 + look, changed_sample)
                # The fresh teacher target uses the very last rebuilt backbone state.
                self.assertEqual(plan.teacher_backbone.stop - 1,
                                 plan.teacher_new_audio_frame + timing.code_delay - 1)
                # Any fewer teacher audio frames would leave student input unavailable.
                self.assertLess(plan.teacher_new_audio_frame * 1920, plan.student_source_stop)
                self.assertGreaterEqual((plan.teacher_new_audio_frame + 1) * 1920,
                                        plan.student_source_stop)

    def test_all_prefill_lengths_bootstrap_teacher_before_student_frontend(self):
        for n in range(38, 63):
            plan = bootstrap_plan(n)
            self.assertEqual(plan.teacher_prefill_stop_model_frame, n - 1)
            self.assertEqual(list(plan.teacher_new_audio.frames()), [n, n + 1, n + 2])
            self.assertEqual(list(plan.student_forced_media_model_frames.frames()), [n, n + 1])
            self.assertEqual(plan.student_heard_stop, (n + 2) * 1920 + 72)
            with self.assertRaises(ValueError):
                plan_for_student(n, n)

    def test_delivery_delay_applies_to_every_prefill_length_from_sample_zero(self):
        timing = AudioTiming(delivery_delay_samples=72)
        for n in range(38, 63):
            plan = bootstrap_plan(n, timing)
            self.assertEqual(plan.teacher_prefill_stop_model_frame, n)
            self.assertEqual(list(plan.teacher_new_audio.frames()), [n, n + 1])
            self.assertEqual(list(plan.student_forced_media_model_frames.frames()), [n, n + 1])
            self.assertEqual(plan.teacher_backbone_stop_model_frame, n + 3)
            self.assertEqual(plan.student_heard_stop, (n + 2) * 1920 + 72)
            self.assertEqual(plan.student_source_stop, (n + 2) * 1920)
            self.assertEqual(plan.teacher_received_real_stop, n * 1920 + 72)

    def test_partial_frame_exposure_prevents_whole_frame_rewrite(self):
        audio = CommittedAudioTimeline(np.zeros(2 * 1920, np.float32))
        chunk = np.arange(1920, dtype=np.float32)
        audio.append(2, chunk)
        chunk[:] = -99  # caller mutation cannot change committed samples
        heard = audio.hear_through(2 * 1920 + 72)
        np.testing.assert_array_equal(heard[-72:], np.arange(72, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "append-only"):
            audio.append(2, np.zeros(1920))
        with self.assertRaisesRegex(ValueError, "not yet committed"):
            audio.hear_through(3 * 1920 + 1)
        heard[:] = 99
        self.assertEqual(audio.read(2 * 1920, 2 * 1920 + 1)[0], 0)

    def test_delivered_prefill_is_shifted_once_and_handoff_has_no_inserted_gap(self):
        prefill = np.arange(1, 17, dtype=np.float32)
        audio = CommittedAudioTimeline(prefill, frame_samples=8, delivery_delay_samples=1)
        self.assertEqual(audio.available_stop, 16)
        self.assertEqual(audio.delivered_available_stop, 17)
        np.testing.assert_array_equal(audio.hear_through(17), np.r_[0., prefill])
        chunk = np.arange(17, 25, dtype=np.float32)
        audio.append(2, chunk)
        # The first generated sample follows the last prefill sample directly.
        np.testing.assert_array_equal(audio.hear_through(25), np.arange(25, dtype=np.float32))
        np.testing.assert_array_equal(audio.delivered_read(14, 21), np.arange(14, 21, dtype=np.float32))
        np.testing.assert_array_equal(audio.read(0, 24), np.arange(1, 25, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, 'append-only'):
            audio.append(2, chunk)
        with self.assertRaisesRegex(ValueError, 'not yet committed'):
            audio.hear_through(26)
        speculative = audio.silence_future(30)
        np.testing.assert_array_equal(speculative[:25], np.arange(25, dtype=np.float32))
        np.testing.assert_array_equal(speculative[25:], 0.)

    def test_reference_delays_both_prefills_and_generated_audio_without_rewriting_history(self):
        timing = AudioTiming(frame_samples=8, lookahead_samples=1, delivery_delay_samples=1)
        n, total = 4, 11
        student_chunks = list(np.arange(1, n * 8 + 1, dtype=np.float32).reshape(n, 8))
        teacher_chunks = list(np.arange(101, 101 + n * 8, dtype=np.float32).reshape(n, 8))
        heard_prefix = np.empty(0, np.float32)
        teacher_calls, student_calls = [], []

        def generate_teacher(frame, incoming, own):
            np.testing.assert_array_equal(own, np.concatenate(teacher_chunks))
            truth = np.r_[np.zeros(1, np.float32), np.concatenate(student_chunks)]
            np.testing.assert_array_equal(incoming[:len(truth)], truth)
            np.testing.assert_array_equal(incoming[len(truth):], 0.)
            self.assertEqual(len(incoming), (frame + 2) * 8 + 1)
            teacher_calls.append((frame, len(student_chunks)))
            chunk = np.arange(frame * 8, (frame + 1) * 8, dtype=np.float32) + 101
            teacher_chunks.append(chunk)
            return chunk

        def generate_student(frame, incoming, own):
            nonlocal heard_prefix
            self.assertEqual(len(incoming), (frame + 2) * 8 + 1)
            truth = np.r_[np.zeros(1, np.float32), np.concatenate(teacher_chunks)]
            np.testing.assert_array_equal(incoming, truth[:len(incoming)])
            np.testing.assert_array_equal(incoming[:len(heard_prefix)], heard_prefix)
            np.testing.assert_array_equal(own, np.concatenate(student_chunks))
            heard_prefix = incoming.copy()
            student_calls.append(frame)
            chunk = np.arange(frame * 8, (frame + 1) * 8, dtype=np.float32) + 1
            student_chunks.append(chunk)
            return chunk

        result = reference_rollout(
            np.concatenate(student_chunks), np.concatenate(teacher_chunks), total,
            generate_student, generate_teacher, timing,
        )
        self.assertEqual(teacher_calls[:2], [(n, n), (n + 1, n)])
        self.assertEqual(teacher_calls[2:], [(i + 1, i) for i in range(n + 1, total)])
        self.assertEqual(student_calls, list(range(n, total)))
        np.testing.assert_array_equal(result.student_audio, np.concatenate(student_chunks))
        np.testing.assert_array_equal(result.teacher_audio, np.concatenate(teacher_chunks))
        self.assertEqual(len(result.teacher_heard_stops), total - n + 1)
        self.assertEqual(len(result.student_heard_stops), total - n)

    def test_reference_never_uses_unknown_student_audio_or_changes_teacher_history(self):
        # Tiny sample units retain the same CD/lookahead dependency graph.
        timing = AudioTiming(frame_samples=8, lookahead_samples=1)
        n, total = 4, 11
        student_chunks = [np.full(8, -3., np.float32) for _ in range(n)]
        teacher_chunks = [np.full(8, -7., np.float32) for _ in range(n)]
        heard_prefix = np.empty(0, np.float32)
        teacher_calls, student_calls = [], []

        def generate_teacher(frame, incoming, own):
            np.testing.assert_array_equal(own, np.concatenate(teacher_chunks))
            truth = np.concatenate(student_chunks)
            np.testing.assert_array_equal(incoming[:len(truth)], truth)
            np.testing.assert_array_equal(incoming[len(truth):], 0.)
            self.assertEqual(len(incoming), (frame + 2) * 8 + 1)
            teacher_calls.append((frame, len(student_chunks)))
            # Updating student input influences NEW teacher output only.
            chunk = np.full(8, 100 * frame + student_chunks[-1][0], np.float32)
            teacher_chunks.append(chunk)
            return chunk

        def generate_student(frame, incoming, own):
            nonlocal heard_prefix
            self.assertEqual(len(incoming), (frame + 2) * 8 + 1)
            np.testing.assert_array_equal(incoming[:len(heard_prefix)], heard_prefix)
            np.testing.assert_array_equal(own, np.concatenate(student_chunks))
            heard_prefix = incoming.copy()
            student_calls.append(frame)
            chunk = np.full(8, frame + .25, np.float32)
            student_chunks.append(chunk)
            return chunk

        result = reference_rollout(
            np.concatenate(student_chunks), np.concatenate(teacher_chunks), total,
            generate_student, generate_teacher, timing,
        )
        self.assertEqual(teacher_calls[:3], [(n, n), (n + 1, n), (n + 2, n)])
        self.assertEqual(teacher_calls[3:], [(i + 2, i) for i in range(n + 1, total)])
        self.assertEqual(student_calls, list(range(n, total)))
        np.testing.assert_array_equal(result.student_audio, np.concatenate(student_chunks))
        np.testing.assert_array_equal(result.teacher_audio, np.concatenate(teacher_chunks))
        self.assertEqual(len(result.student_heard_stops), total - n)
        self.assertEqual(len(result.teacher_heard_stops), total - n + 2)


if __name__ == "__main__":
    unittest.main()
