"""Deterministic production prefixes preserve prefill and source coverage."""
import contextlib
import copy
from dataclasses import replace
import json
from types import SimpleNamespace
import unittest

import numpy as np

from finetune.pose_opd_credit.config import TrainingConfig
from finetune.pose_opd_credit.data import ProductionRows, UnavailableRow, prefix_seed
from finetune.pose_opd_credit.tests.test_rollout import seed_row
from finetune.pose_opd_credit.train import make_plan


class ConversationPrefixTests(unittest.TestCase):
    def test_first_two_minutes_preserve_forced_prefill_and_erase_future_targets(self):
        source = seed_row(frames=2000, prefix=134)
        source['wav'] = np.arange(2000 * 1920, dtype=np.float32)
        source['aref'] = json.dumps({'n': 2000, 'frame0': 123, 'flac': 'same-a.flac'})
        original = copy.deepcopy(source)
        value = prefix_seed(source, 1500)
        self.assertEqual(value['conversation_frames'], 1500)
        self.assertEqual(value['original_conversation_frames'], 2000)
        self.assertEqual(value['segs'], [1636])
        np.testing.assert_array_equal(value['wav'], source['wav'][:1500 * 1920])
        np.testing.assert_array_equal(value['ids'][:, :134 + 25], source['ids'][:, :134 + 25])
        np.testing.assert_array_equal(value['depth'][:, :134 + 27], source['depth'][:, :134 + 27])
        np.testing.assert_array_equal(value['pdepth'][:134 + 27], source['pdepth'][:134 + 27])
        for stream in ('text', 'audio', 'pose'):
            self.assertEqual(value['sampled_masks'][stream].sum(), 1475)
        self.assertFalse(value['sampled_masks']['text'][-2:].any())
        self.assertTrue(value['sampled_masks']['audio'][-2:].all())
        self.assertTrue(value['sampled_masks']['pose'][-2:].all())
        self.assertTrue((value['depth'][1, value['sampled_masks']['audio']] == -1).all())
        self.assertTrue((value['pdepth'][value['sampled_masks']['pose']] == -1).all())
        ref = json.loads(value['aref'])
        self.assertEqual((ref['n'], ref['original_n'], ref['frame0']), (1500, 2000, 123))
        for key in ('ids', 'depth', 'pdepth', 'wav'):
            np.testing.assert_array_equal(source[key], original[key])

    def test_shorter_rows_and_audio_free_loads_remain_complete(self):
        source = seed_row(frames=1000, prefix=5)
        value = prefix_seed(source, 1500)
        self.assertNotIn('wav', value)
        self.assertEqual(value['conversation_frames'], 1000)
        self.assertEqual(value['original_conversation_frames'], 1000)
        for key in ('ids', 'depth', 'pdepth', 'present', 'pid', 'vin'):
            np.testing.assert_array_equal(value[key], source[key])
        self.assertEqual(prefix_seed(prefix_seed(seed_row(frames=2000), 1500), 750)['original_conversation_frames'], 2000)
        for cap in (0, 25, True):
            with self.assertRaises(ValueError):
                prefix_seed(source, cap)

    def fake_rows(self, prefix, capacity=None, lengths=(7500, 1000, 25), exclude_short=False):
        rows = ProductionRows.__new__(ProductionRows)
        rows.config = TrainingConfig(conversation_prefix_frames=prefix, max_row_frames=capacity,
                                    exclude_invalid_pose_frame0=False, exclude_short_rows=exclude_short)
        rows.cfg, rows.metadata = {}, {}
        source = [dict(n_frames=n, chunk_index=0, row_key=f'row-{n}') for n in lengths]
        rows._open = lambda module, k: ({}, source, None)
        module = SimpleNamespace(D=SimpleNamespace(voice_clips=lambda *_: (None, None)),
            pose_entry=lambda _: {'ok': True}, splits=lambda: {'train': [(0, i) for i in range(len(lengths))]})
        rows.arena = SimpleNamespace(enter=lambda: contextlib.nullcontext(module))
        return rows

    def test_exclude_short_rows_filters_before_plan_at_exact_boundary(self):
        rows = self.fake_rows(1500, lengths=(1499, 1500, 1501), exclude_short=True)
        items = rows.items()
        self.assertEqual(set(items), {(0, 1, 0), (0, 1, 1), (0, 2, 0), (0, 2, 1)})
        self.assertEqual(len(rows.audit), 2)
        self.assertTrue(all('minimum required is 1500' in item['reason'] for item in rows.audit))
        self.assertEqual(set(rows.frame_lengths(items).values()), {1500})
        self.assertEqual(rows.metadata['conversation_selection']['selected_frames'], 6000)
        self.assertTrue(rows.metadata['conversation_selection']['exclude_short_rows'])
        plan = make_plan(items, world=2, epochs=1, seed=1, rows_per_rank=2)
        self.assertEqual(plan['rows_per_epoch'], 4)
        self.assertCountEqual(map(tuple, plan['updates'][0]), items)
        retained = self.fake_rows(1500, lengths=(1499, 1500, 1501)).items()
        self.assertEqual(len(retained), 6)
        # Duration selection remains exclusion even under strict clip validation.
        strict = self.fake_rows(1500, lengths=(1499,), exclude_short=True)
        strict.config = replace(strict.config, missing_clip_policy='error')
        with self.assertRaisesRegex(UnavailableRow, 'No rows meet'):
            strict.items()
        self.assertEqual(len(strict.audit), 2)
        for kwargs in ({'exclude_short_rows': True}, {'conversation_prefix_frames': 1500, 'exclude_short_rows': 1}):
            with self.assertRaises(ValueError):
                TrainingConfig(**kwargs).validate()

    def test_metadata_capacity_duration_and_epochs_use_capped_rows_once(self):
        rows = self.fake_rows(1500, capacity=1500)
        items = rows.items()
        self.assertEqual(len(items), 4)
        self.assertEqual(sorted(rows.frame_lengths(items).values()), [1000, 1000, 1500, 1500])
        self.assertEqual(rows.metadata['conversation_selection']['source_frames'], 17000)
        self.assertEqual(rows.metadata['conversation_selection']['selected_frames'], 5000)
        self.assertEqual(max(spec['total_frames'] for spec in rows.shape_specs(items)), 1507)
        self.assertEqual(max(spec['original_conversation_frames'] for spec in rows.shape_specs(items)), 7500)
        plan = make_plan(items, world=2, epochs=2, seed=1, rows_per_rank=2)
        for update in plan['updates']:
            self.assertCountEqual(map(tuple, update), items)
        # Without the configured prefix, the original capacity rejection still applies.
        original = self.fake_rows(None, capacity=1500)
        self.assertEqual(len(original.items()), 2)
        original = self.fake_rows(None)
        original_items = original.items()
        self.assertEqual(max(original.frame_lengths(original_items).values()), 7500)
        self.assertIsNone(TrainingConfig().conversation_prefix_frames)


if __name__ == '__main__':
    unittest.main()
