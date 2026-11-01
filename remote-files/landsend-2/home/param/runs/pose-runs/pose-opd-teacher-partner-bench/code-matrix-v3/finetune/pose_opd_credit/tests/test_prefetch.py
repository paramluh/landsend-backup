"""Spawned CPU prefetch isolation, bounded queue and production row parity."""
import json
import os
from pathlib import Path
import unittest

import numpy as np

from finetune.pose_opd_credit.config import TrainingConfig
from finetune.pose_opd_credit.prefetch import RowPrefetch


class FakeRows:
    def load(self, item, rank):
        import torch
        if item[0] < 0:
            raise ValueError('deliberate loader failure')
        return {'item': tuple(item), 'rank': rank, 'pid': os.getpid(),
                'visible': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'threads': torch.get_num_threads(), 'values': np.arange(4)}


def fake_factory(config):
    return FakeRows()


class FakePartnerRows:
    def load(self, item, rank, *, update, slot):
        return {'item': tuple(item), 'rank': rank, 'update': update, 'slot': slot}


def fake_partner_factory(config):
    return FakePartnerRows()


class PrefetchTests(unittest.TestCase):
    def test_partner_prefetch_preserves_update_and_row_seed_coordinates(self):
        cfg = TrainingConfig(data_variant='raw', rollout_mode='teacher_partner', effective_batch_per_rank=2)
        with RowPrefetch(cfg, rank=9, fallback_item=(7, 8, 0), loader_factory=fake_partner_factory) as loader:
            loader.submit(17, [(1, 2, 0), None])
            actual = loader.take(17, timeout=30)
        self.assertEqual(actual, [{'item': (1, 2, 0), 'rank': 9, 'update': 17, 'slot': 0},
                                  {'item': (7, 8, 0), 'rank': 9, 'update': 17, 'slot': 1}])

    def test_two_batches_same_worker_cpu_isolation_and_dummy_fallback(self):
        cfg = TrainingConfig(effective_batch_per_rank=2)
        parent_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        with RowPrefetch(cfg, rank=9, fallback_item=(7, 8, 0), loader_factory=fake_factory) as loader:
            loader.submit(4, [(1, 2, 0), None])
            with self.assertRaisesRegex(RuntimeError, 'pending'):
                loader.submit(5, [(1, 2, 0), None])
            with self.assertRaisesRegex(RuntimeError, 'cursor'):
                loader.take(5)
            first = loader.take(4, timeout=30)
            loader.submit(5, [(3, 4, 1), (5, 6, 0)])
            second = loader.take(5, timeout=30)
            self.assertEqual(first[1]['item'], (7, 8, 0))
            self.assertEqual(second[0]['item'], (3, 4, 1))
            self.assertEqual(first[0]['pid'], second[0]['pid'])
            self.assertNotEqual(first[0]['pid'], os.getpid())
            self.assertTrue(all(row['visible'] == '' and row['threads'] == 1 and row['rank'] == 9
                                for row in first + second))
        self.assertEqual(os.environ.get('CUDA_VISIBLE_DEVICES'), parent_visible)
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            loader.submit(6, [(1, 2, 0), None])

    def test_worker_exception_reaches_caller_and_capacity_is_bounded(self):
        with RowPrefetch(TrainingConfig(), rank=0, fallback_item=(0, 0, 0),
                         loader_factory=fake_factory) as loader:
            with self.assertRaisesRegex(ValueError, 'effective_batch'):
                loader.submit(0, [])
            loader.submit(0, [(-1, 0, 0)])
            with self.assertRaisesRegex(ValueError, 'deliberate loader failure'):
                loader.take(0, timeout=30)
            loader.submit(1, [(1, 0, 0)])
            self.assertEqual(loader.take(1, timeout=30)[0]['item'], (1, 0, 0))

    @unittest.skipUnless(Path('/home/param/datasets/aloe-duplex').is_dir(), 'production dataset unavailable')
    def test_real_eight_rows_equal_direct_loading(self):
        from finetune.pose_opd_credit.data import ProductionRows
        cfg = TrainingConfig.from_dict(json.loads((Path(__file__).parents[1] / 'recipe.raw.json').read_text()))
        items = [(3, 2344, 0), (2, 1184, 1), (0, 1975, 1), (1, 466, 1),
                 (2, 504, 1), (0, 1159, 0), (3, 2506, 0), (3, 2616, 0)]
        direct = ProductionRows(cfg)
        with RowPrefetch(cfg, rank=0, fallback_item=items[0]) as loader:
            loader.submit(0, items)
            expected = [direct.load(item, 0) for item in items]
            actual = loader.take(0, timeout=120)
        def equal(a, b):
            if isinstance(a, np.ndarray):
                np.testing.assert_array_equal(a, b)
            elif isinstance(a, dict):
                self.assertEqual(set(a), set(b))
                for key in a:
                    equal(a[key], b[key])
            elif isinstance(a, (list, tuple)):
                self.assertEqual(len(a), len(b))
                for x, y in zip(a, b):
                    equal(x, y)
            else:
                self.assertEqual(a, b)
        for x, y in zip(expected, actual):
            equal(x, y)


if __name__ == '__main__':
    unittest.main()
