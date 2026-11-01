"""Conversation token statistics follow actual unique draws and global counts."""
import unittest

import numpy as np

from finetune.pose_opd_credit.conversation_metrics import (
    CATEGORIES, PUBLIC_CATEGORIES, NEW_WORD_ID, PAD_ID, empty_conversation_metrics,
    finalize_conversation_metrics, row_conversation_metrics,
)


class ConversationMetricsTests(unittest.TestCase):
    def test_excludes_system_prefill_and_tail_and_closes_all_categories(self):
        text = np.full(11, PAD_ID, np.int64)
        text[5:9] = [PAD_ID, NEW_WORD_ID, 42, 151645]
        probabilities = np.full((11, 4), np.nan)
        probabilities[5:9] = [.4, .1, .3, .2]
        raw = row_conversation_metrics(text, probabilities, prefix=2, prefill_frames=3,
                                       frames=7, role='student')
        got = finalize_conversation_metrics(raw)
        self.assertEqual(got['conversation/student/decisions'], 4)
        for category, expected in zip(PUBLIC_CATEGORIES, (.4, .1, .3), strict=True):
            self.assertEqual(got[f'conversation/student/sampled_count/{category}'], 1)
            self.assertEqual(got[f'conversation/student/sampled_fraction/{category}'], .25)
            self.assertAlmostEqual(got[f'conversation/student/probability/{category}'], expected)
        self.assertFalse(any(key.endswith('/other') for key in got))
        self.assertAlmostEqual(sum(got[f'conversation/student/probability/{category}']
                                   for category in PUBLIC_CATEGORIES), .8)
        self.assertEqual(sum(got[f'conversation/student/sampled_count/{category}']
                             for category in PUBLIC_CATEGORIES), 3)

    def test_probabilities_are_not_reconstructed_from_sample_counts(self):
        raw = row_conversation_metrics(np.full(3, PAD_ID, np.int64),
            np.tile([.2, .1, .6, .1], (3, 1)), prefix=0, prefill_frames=0, frames=3, role='partner')
        got = finalize_conversation_metrics(raw)
        self.assertEqual(got['conversation/partner/sampled_fraction/pad'], 1.)
        self.assertAlmostEqual(got['conversation/partner/probability/pad'], .2)
        self.assertAlmostEqual(got['conversation/partner/probability/word'], .6)

    def test_unequal_rank_counts_are_summed_before_fractions(self):
        first = row_conversation_metrics(np.array([42]), np.array([[.2, .1, .6, .1]]),
            prefix=0, prefill_frames=0, frames=1, role='student')
        second = row_conversation_metrics(np.full(3, PAD_ID), np.tile([.9, 0, .1, 0], (3, 1)),
            prefix=0, prefill_frames=0, frames=3, role='student')
        averaged = {key: (first[key] + second[key]) / 2 for key in first}
        got = finalize_conversation_metrics({**averaged, 'unrelated': 7}, 2)
        self.assertEqual(got['unrelated'], 7)
        self.assertEqual(got['conversation/student/decisions'], 4)
        self.assertEqual(got['conversation/student/sampled_count/pad'], 3)
        self.assertEqual(got['conversation/student/sampled_fraction/pad'], .75)
        self.assertAlmostEqual(got['conversation/student/probability/pad'], .725)
        self.assertFalse(any(key.startswith('_conversation/') for key in got))

    def test_unique_model_slots_and_sampled_mask_exclude_forced_actions(self):
        tokens = np.array([PAD_ID, NEW_WORD_ID, 42, 151645])
        probabilities = np.tile([.25] * 4, (4, 1))
        raw = row_conversation_metrics(tokens, probabilities, prefix=0, prefill_frames=0,
            frames=4, role='partner', sampled_mask=np.array([True, False, True, True]))
        got = finalize_conversation_metrics(raw)
        self.assertEqual(got['conversation/partner/decisions'], 3)
        self.assertEqual(got['conversation/partner/sampled_count/new_word'], 0)

    def test_dummy_rows_and_all_dummy_ranks_do_not_invent_probabilities(self):
        raw = empty_conversation_metrics()
        self.assertEqual(len(raw), 18)
        dummy = row_conversation_metrics(None, None, prefix=0, prefill_frames=0,
                                         frames=4, role='student', dummy=True)
        self.assertTrue(all(value == 0 for value in dummy.values()))
        got = finalize_conversation_metrics(raw, 56)
        self.assertEqual(got['conversation/student/decisions'], 0)
        self.assertNotIn('conversation/student/probability/pad', got)
        self.assertNotIn('conversation/partner/sampled_fraction/word', got)

    def test_invalid_sampling_receipts_are_rejected(self):
        args = dict(prefix=0, prefill_frames=0, frames=2, role='student')
        text = np.array([PAD_ID, 42])
        for probabilities in (np.zeros((2, 4)), np.full((2, 4), np.nan),
                              np.ones((2, 3)), np.array([[-.1, .1, .5, .5]] * 2)):
            with self.assertRaises(ValueError):
                row_conversation_metrics(text, probabilities, **args)
        with self.assertRaises(ValueError):
            row_conversation_metrics(np.array([-1, 151682]), np.full((2, 4), .25), **args)
        with self.assertRaises(ValueError):
            row_conversation_metrics(text, np.full((2, 4), .25), sampled_mask=np.array([1, 0]), **args)
        with self.assertRaises(ValueError):
            finalize_conversation_metrics({'_conversation/student/decisions': 4})


if __name__ == '__main__':
    unittest.main()
