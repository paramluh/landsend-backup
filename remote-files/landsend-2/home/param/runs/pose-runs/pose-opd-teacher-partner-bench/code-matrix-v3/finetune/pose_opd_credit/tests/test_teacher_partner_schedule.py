"""Physical waveform dependencies and irreversible partner-output commitment."""
import unittest

import numpy as np

from finetune.pose_opd_credit.teacher_partner_schedule import (
    AudioTiming, CommittedAudioTimeline, bootstrap_plan, plan_for_student, reference_rollout,
)


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

    def test_refresh_is_the_minimal_interval_covering_changed_dependencies(self):
        for look in (0, 1, 72, 1919):
            timing = AudioTiming(lookahead_samples=look)
            for i in range(39, 65):
                plan = plan_for_student(i, 38, timing)
                changed_sample = (i - 1) * timing.frame_samples
                first = plan.teacher_backbone.start
                # The preceding frame's frontend cannot see the changed sample.
                self.assertLessEqual(first * 1920 + look, changed_sample)
                self.assertGreater((first + 1) * 1920 + look, changed_sample)
                # The fresh teacher target uses the very last rebuilt backbone state.
                self.assertEqual(plan.teacher_backbone.stop - 1,
                                 plan.teacher_new_audio_frame + timing.code_delay - 1)
                # Any fewer teacher audio frames would leave student input unavailable.
                self.assertLess(plan.teacher_new_audio_frame * 1920, plan.student_heard_stop)
                self.assertGreaterEqual((plan.teacher_new_audio_frame + 1) * 1920,
                                        plan.student_heard_stop)

    def test_all_prefill_lengths_bootstrap_teacher_before_student_frontend(self):
        for n in range(38, 63):
            plan = bootstrap_plan(n)
            self.assertEqual(plan.teacher_prefill_stop_model_frame, n - 1)
            self.assertEqual(list(plan.teacher_new_audio.frames()), [n, n + 1, n + 2])
            self.assertEqual(list(plan.student_forced_media_model_frames.frames()), [n, n + 1])
            self.assertEqual(plan.student_heard_stop, (n + 2) * 1920 + 72)
            with self.assertRaises(ValueError):
                plan_for_student(n, n)

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
