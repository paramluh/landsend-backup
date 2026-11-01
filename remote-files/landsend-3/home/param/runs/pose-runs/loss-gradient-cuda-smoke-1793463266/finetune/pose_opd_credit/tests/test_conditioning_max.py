"""Reference-clip duration policy, independent of conversation prefill."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from finetune.pose_opd_credit.config import TrainingConfig
from finetune.pose_opd_credit.data import ProductionRows, conditioning_clips


class MaximumConditioningTests(unittest.TestCase):
    def fixtures(self, frames=300, spoken=300):
        source = {'row_key': 'source', 'n_frames': frames,
                  'words_user': spoken, 'words_assistant': spoken}
        rows = [source]
        siblings = SimpleNamespace(rows=rows, siblings=lambda *_: ([], [(1, 0, 'assistant')]))
        native = SimpleNamespace(
            h64=lambda _: 7, rec_chunks=lambda *_: [0, 1],
            SPR=SimpleNamespace(coverage_mask=lambda count, n: (np.arange(n) < count).astype(np.int64)),
            voice_clips=lambda *_: {0: 'native user', 1: 'native assistant'})
        cfg = {'seed': 20261011, 'clip_min': 38, 'clip_max': 125,
               'stride': 4, 'min_cov': .5, 'top_frac': .9}
        row = {'row_key': 'current', 'chunk_index': 0}
        return native, row, siblings, cfg

    def test_full_reference_uses_source_maximum_for_both_partners(self):
        args = self.fixtures()
        for seed in (0, 1, 71, 20261011):
            cfg = {**args[-1], 'seed': seed}
            clips = conditioning_clips(*args[:-1], cfg, 'source_max')
            self.assertEqual([clips[track]['L'] for track in (0, 1)], [125, 125])
            self.assertTrue(all(clip['blank_lane'] for clip in clips.values()))

    def test_longest_coverage_valid_clip_does_not_use_halving_fallback(self):
        # Forty speech frames permit at most eighty frames at 50% coverage.
        # The native 125 -> 62 halving fallback would leave usable audio out.
        clips = conditioning_clips(*self.fixtures(spoken=40), 'source_max')
        self.assertEqual([clips[track]['L'] for track in (0, 1)], [80, 80])
        self.assertEqual(clips[0]['cov'], .5)

    def test_short_source_is_not_padded_and_missing_clip_stays_missing(self):
        clips = conditioning_clips(*self.fixtures(frames=90, spoken=90), 'source_max')
        self.assertEqual(clips[0]['L'], 90)
        for frames, spoken in ((37, 37), (300, 0), (300, 18)):
            self.assertEqual(conditioning_clips(*self.fixtures(frames, spoken), 'source_max'),
                             {0: None, 1: None})

    def test_native_selection_is_unchanged_by_default(self):
        self.assertEqual(conditioning_clips(*self.fixtures()),
                         {0: 'native user', 1: 'native assistant'})

    def test_later_second_half_sibling_rule_and_order_are_retained(self):
        native, row, siblings, cfg = self.fixtures()
        native.rec_chunks = lambda *_: [0, 1, 2, 3]
        siblings.rows = [dict(siblings.rows[0], row_key=f'source-{i}') for i in range(4)]
        siblings.siblings = lambda *_: ([], [(0, 0, 'assistant'), (1, 1, 'assistant'),
                                            (2, 2, 'assistant'), (3, 3, 'assistant')])
        clips = conditioning_clips(native, row, siblings, cfg, 'source_max')
        self.assertEqual(clips[0]['src_key'], 'source-2')
        self.assertEqual(clips[0]['src_chunk'], 2)

    def test_later_full_length_clip_is_preferred_to_shorter_nearest_sibling(self):
        native, row, siblings, cfg = self.fixtures()
        native.rec_chunks = lambda *_: [0, 1, 2, 3]
        siblings.rows = [
            dict(siblings.rows[0], row_key='short', words_assistant=40),
            dict(siblings.rows[0], row_key='full')]
        siblings.siblings = lambda *_: ([], [(2, 0, 'assistant'), (3, 1, 'assistant')])
        clips = conditioning_clips(native, row, siblings, cfg, 'source_max')
        self.assertEqual(clips[0]['L'], 125)
        self.assertEqual(clips[0]['src_key'], 'full')


@unittest.skipUnless(Path('/home/param/datasets/aloe-duplex/pose/POSE_INDEX.json').is_file(),
                     'Production data not installed')
class MaximumConditioningProductionTests(unittest.TestCase):
    def test_real_raw_row_uses_ten_second_reference_without_text_or_pose(self):
        cfg = replace(TrainingConfig(data_variant='raw'), conditioning_clip_length='source_max')
        rows = ProductionRows(cfg)
        seed = rows.load((0, 0, 0), load_audio=False)
        conditioning = seed['conditioning']
        self.assertEqual(conditioning['clip_frames'], 125)
        self.assertEqual(conditioning['maximum_clip_frames'], 125)
        self.assertEqual(conditioning['length_policy'], 'source_max')
        self.assertFalse(conditioning['reference_text'])
        self.assertFalse(conditioning['reference_pose'])
        self.assertEqual(np.count_nonzero(seed['depth'][1, :int(seed['P'][0]), 0] >= 0), 125)


if __name__ == '__main__':
    unittest.main()
