"""Optimizer-batch identity and normalization checks for execution regrouping."""

from types import SimpleNamespace
import unittest

import numpy as np
import torch

from dataloading.ds_aloe import AloeRecord
from distillation.full_gt_data import (FullGTEpochDataset, concatenate_full_gt_batches,
                                       regroup_full_gt_group, split_full_gt_batch)


class MemoryDataset(FullGTEpochDataset):
    """Use the actual epoch planner without reading large video assets."""

    def _load(self, windows, row_ids, valid, *, targets=1):
        ids = torch.tensor(row_ids)
        return {
            'latents_low': ids.float()[:, None, None] / 50,
            'pose_latents_low': (ids + 1).float()[:, None, None],
            'source_frame_indices': torch.stack([ofi for _, ofi in windows]),
            'refs': ids[:, None] * 10 + torch.arange(9)[None],
            'ref_keep': torch.arange(9)[None] != ids[:, None] % 9,
            'codes': ids[:, None].expand(-1, 4),
            'bad_audio': ids.remainder(2).bool(),
            'window_source': torch.arange(len(ids)),
            'sample_weight': torch.tensor(valid),
            'epoch_window_id': torch.where(torch.tensor(valid).bool(), ids, -1),
        }


def dataset(rank):
    # Forty-six eligible rows: five full batches and a six-row padded tail.
    source = SimpleNamespace(bank='/unused/reference-bank', lengths=np.array([53]),
                             clips=[AloeRecord('video', 'pose', 'mimi', 53, 'actor')])
    return MemoryDataset(source, rank, 2, 2, global_batch_size=8, critic_updates=4)


class FullGTRegroupTests(unittest.TestCase):
    def test_four_critic_updates_keep_distinct_rows_and_paired_references(self):
        global_seen = set()
        for update in range(4):
            update_ids = set()
            for rank in range(2):
                original = dataset(rank).critic_group(2, update)
                regrouped = regroup_full_gt_group(original, {'stabilizer': 4, 'resolver': 4})
                self.assertEqual(len(regrouped), 2)
                stab, resolver = regrouped
                for batch in regrouped:
                    self.assertEqual(batch['critic_update_index'], update)
                    self.assertEqual(batch['phase'], 'critic')
                    self.assertEqual(batch['global_objective_rows'], {'stabilizer': 8, 'resolver': 8})
                    self.assertEqual(batch['global_microbatch_rows'], 8)
                    self.assertEqual(batch['global_valid_rows'], 8)
                for key in ('epoch_window_id', 'refs', 'ref_keep', 'sample_weight', 'source_frame_indices'):
                    torch.testing.assert_close(stab[key], resolver[key])
                    torch.testing.assert_close(stab[key], torch.cat([original[0][key], original[1][key]]))
                torch.testing.assert_close(stab['window_source'], torch.arange(4))
                update_ids.update(stab['epoch_window_id'].tolist())
            self.assertEqual(len(update_ids), 8)
            self.assertFalse(update_ids & global_seen)
            global_seen.update(update_ids)
        for rank in range(2):
            student_ids = set(torch.cat([batch['epoch_window_id'] for batch in dataset(rank).group(2)[:2]]).tolist())
            self.assertFalse(student_ids & global_seen)

    def test_padding_keeps_global_counts_even_on_a_padded_rank(self):
        for rank in range(2):
            original = dataset(rank).group(5)
            regrouped = regroup_full_gt_group(original, {'stabilizer': 2, 'resolver': 4})
            self.assertEqual([b['objective'] for b in regrouped], ['stabilizer', 'stabilizer', 'resolver'])
            self.assertEqual([b['microbatch_index'] for b in regrouped], [0, 1, 2])
            self.assertEqual([b['objective_microbatch_index'] for b in regrouped], [0, 1, 0])
            self.assertEqual([b['global_microbatch_rows'] for b in regrouped], [4, 2, 6])
            self.assertTrue(all(b['global_objective_rows'] == {'stabilizer': 6, 'resolver': 6}
                                for b in regrouped))
            singles = split_full_gt_batch(regrouped[-1], 1)
            self.assertEqual([b['global_microbatch_rows'] for b in singles], [2, 2, 1, 1])
            if rank == 1:
                self.assertEqual(regrouped[-1]['sample_weight'].tolist(), [1, 1, 0, 0])
                self.assertEqual(regrouped[-1]['epoch_window_id'][-2:].tolist(), [-1, -1])

    def test_weighted_loss_and_parameter_gradient_survive_regrouping(self):
        def accumulated_gradient(regroup):
            per_rank = []
            losses = []
            for rank in range(2):
                parameter = torch.tensor(0.75, requires_grad=True)
                batches = dataset(rank).group(5)
                if regroup:
                    batches = regroup_full_gt_group(batches, {'stabilizer': 2, 'resolver': 4})
                total = 0.
                for batch in batches:
                    prediction = batch['latents_low'].flatten(1).mean(1) * parameter
                    loss = ((prediction - 0.4).square() * batch['sample_weight']).sum()
                    loss = loss * 2 / batch['global_objective_rows'][batch['objective']] * .5
                    total += float(loss.detach())
                    loss.backward()
                per_rank.append(parameter.grad)
                losses.append(total)
            return torch.stack(per_rank).mean(), sum(losses) / 2

        old_grad, old_loss = accumulated_gradient(False)
        new_grad, new_loss = accumulated_gradient(True)
        torch.testing.assert_close(old_grad, new_grad)
        self.assertAlmostEqual(old_loss, new_loss, places=7)

    def test_mixed_updates_and_changed_objective_order_are_rejected(self):
        data = dataset(0)
        first, second = data.critic_group(0, 0), data.critic_group(0, 1)
        with self.assertRaisesRegex(ValueError, 'critic_update_index'):
            concatenate_full_gt_batches([first[0], second[0]])
        with self.assertRaisesRegex(ValueError, 'mixed critic_update_index'):
            regroup_full_gt_group(first[:2] + second[2:], {'stabilizer': 4, 'resolver': 4})
        with self.assertRaisesRegex(ValueError, 'objective order'):
            regroup_full_gt_group(first[::-1], {'stabilizer': 4, 'resolver': 4})
        with self.assertRaisesRegex(ValueError, 'divide evenly'):
            regroup_full_gt_group(first, {'stabilizer': 3, 'resolver': 4})

    def test_regroup_does_not_mutate_prefetched_shared_metadata_or_tensors(self):
        original = dataset(1).group(5)
        snapshots = [{key: value.clone() if isinstance(value, torch.Tensor) else
                      dict(value) if isinstance(value, dict) else value for key, value in batch.items()}
                     for batch in original]
        regrouped = regroup_full_gt_group(original, {'stabilizer': 4, 'resolver': 4})
        regrouped[0]['global_objective_rows']['stabilizer'] = -1
        for batch, expected in zip(original, snapshots):
            for key, value in expected.items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(batch[key], value)
                else:
                    self.assertEqual(batch[key], value)


if __name__ == '__main__':
    unittest.main()
