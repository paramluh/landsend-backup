"""Physical page reuse must preserve every causally visible key after rewind."""
import unittest

import torch

from finetune.pose_opd_credit.streaming.paged_kv import PageRingLayout, allocate_cache


class PagedKVTests(unittest.TestCase):
    def test_absolute_logical_pages_reuse_only_out_of_window_storage(self):
        layout = PageRingLayout(6040, 750, 32, 32)
        self.assertEqual(layout.capacity_positions, 1024)
        table = layout.page_table(2, 'cpu')
        self.assertEqual(tuple(table.shape), (2, 24))
        torch.testing.assert_close(table[1] - table[0], torch.full((24,), 4, dtype=torch.int32))
        # Each entry records the absolute position last written into its slot.
        physical = [-1] * layout.capacity_positions
        high_water = 0
        for start in range(0, 6000, 4):
            for position in range(start, start + 4):
                physical[position % layout.capacity_positions] = position
            high_water = start + 4
            # Check worst allowed rewind at both sides of every page boundary.
            if start % 256 in (0, 4, 228, 232, 252):
                rewind = max(0, high_water - 32)
                for query in range(rewind, high_water):
                    first = max(0, query - layout.attention_window + 1)
                    self.assertEqual(physical[first % layout.capacity_positions], first)
                    self.assertEqual(physical[query % layout.capacity_positions], query)

    def test_allocator_never_materializes_logical_capacity_in_paged_mode(self):
        keys, values, table, layout = allocate_cache(batch=2, positions=6040, heads=3,
            head_dim=4, layers=2, device='cpu', mode='paged')
        self.assertEqual(keys[0].shape, (8, 256, 3, 4))
        self.assertEqual(len(keys), len(values))
        self.assertEqual(sum(x.numel() * x.element_size() for x in keys + values),
                         2 * 2 * 2 * 1024 * 3 * 4 * 2)
        self.assertEqual(table.shape[1], 24)
        self.assertEqual(layout.capacity_positions, 1024)

    def test_capacity_grows_when_window_rewind_or_block_needs_it(self):
        layout = PageRingLayout(10000, 750, 300, 32)
        self.assertGreaterEqual(layout.capacity_positions, 1082)
        with self.assertRaises(ValueError):
            PageRingLayout(6040, 750, 32, 32, page_size=128)
        with self.assertRaises(ValueError):
            allocate_cache(batch=1, positions=100, heads=1, head_dim=8, layers=1,
                           device='cpu', mode='unknown')


if __name__ == '__main__':
    unittest.main()
