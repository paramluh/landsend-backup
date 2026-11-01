import unittest

from finetune.pose_opd_credit.train import make_plan


class BalancedPlanTests(unittest.TestCase):
    def test_preserves_every_shuffled_global_batch_and_zero_weight_tail(self):
        items = [(0, index, index % 2) for index in range(173)]
        lengths = {item: 100 + (index * 73) % 7400 for index, item in enumerate(items)}
        options = dict(world=40, epochs=2, seed=11, rows_per_rank=4)
        plain = make_plan(items, **options)
        balanced = make_plan(items, **options, frame_lengths=lengths)
        self.assertEqual(balanced, make_plan(items, **options, frame_lengths=lengths))
        self.assertNotEqual(plain['digest'], balanced['digest'])
        for before, after in zip(plain['updates'], balanced['updates']):
            self.assertCountEqual(before, after)
            self.assertEqual(len(after), 160)
            for rank in range(40):
                self.assertEqual(len(after[rank * 4:(rank + 1) * 4]), 4)

    def test_balances_large_and_small_rows_with_exact_slot_counts(self):
        items = [(0, index, 0) for index in range(160)]
        lengths = {item: (7500 if index < 80 else 1000) for index, item in enumerate(items)}
        plan = make_plan(items, world=40, epochs=1, seed=1, rows_per_rank=4, frame_lengths=lengths)
        batch = plan['updates'][0]
        totals = [sum(lengths[tuple(row)] for row in batch[rank * 4:(rank + 1) * 4]) for rank in range(40)]
        self.assertEqual(totals, [17000] * 40)

    def test_rejects_missing_or_nonpositive_lengths(self):
        items = [(0, 1, 0)]
        for lengths in ({}, {items[0]: 0}, {items[0]: 1.5}, {items[0]: 5, (0, 2, 0): 8}):
            with self.assertRaises(ValueError):
                make_plan(items, world=1, epochs=1, seed=1, frame_lengths=lengths)


if __name__ == '__main__':
    unittest.main()
