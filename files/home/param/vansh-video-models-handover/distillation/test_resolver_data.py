import unittest
from unittest.mock import patch

import numpy as np
import torch

from distillation.resolver_data import ResolverWindows, learning_rate_factor, rank_row_units
from dataloading.ds_dense import DenseWindows


class FakeAloeSource:
    """Frame values encode absolute clip/frame identity, including audio shifts."""
    def __init__(self):
        self.lengths = np.array([410, 419], dtype=np.int64)
        self.clips = list(range(len(self.lengths)))
        self.chunk_ids = np.array([100, 101], dtype=np.int32)
        self.person_ids = np.array([20, 21], dtype=np.int32)
        self.bad_audio = np.array([False, True])
        self.bank = None

    def load_window(self, clip, ofi):
        assert ofi[0] == 0 and bool((ofi[1:] >= 1).all())
        assert int(ofi.max()) < self.lengths[clip]
        start, end = int(ofi[1:].min()), int(ofi.max())+1
        base = 1000*clip
        sink = torch.full((1,1,1,1), float(base))
        body = (torch.arange(start,end,dtype=torch.float32)+base).view(1,-1,1,1)
        shift = max(0,end-1-374)
        shifted = ofi.clone();shifted[1:] -= shift
        codes = torch.full((2,750), clip, dtype=torch.long)
        return sink,body,sink+10000,body+10000,start,codes,shifted,shift


class ResolverDataTests(unittest.TestCase):
    def datasets(self):
        source = FakeAloeSource()
        with patch('dataloading.ds_aloe.is_main_process', return_value=False):
            return [ResolverWindows(source,rank,8,160,5,5) for rank in range(8)]

    @staticmethod
    def identities(batch):
        return [(int(chunk),int(target)) for chunk,target in
                zip(batch['chunk_index'],batch['latents_low'][:,0,7,0,0])]

    def test_all_eight_ranks_are_disjoint_and_have_four_target_windows(self):
        datasets = self.datasets()
        self.assertEqual({len(d) for d in datasets},{5})
        epoch_seen = set()
        shifted_window_seen = False
        for step in range(len(datasets[0])):
            step_seen = set()
            for rank,dataset in enumerate(datasets):
                batch = dataset[step]
                self.assertEqual(batch['latents_low'].shape,(5,1,11,1,1))
                self.assertEqual(batch['pose_latents_low'].shape,(5,1,11,1,1))
                self.assertEqual(batch['present_valid'].shape,(5,4))
                self.assertTrue(bool(batch['present_valid'].all()))
                self.assertEqual(batch['codes'].shape,(5,2,750))
                self.assertTrue(bool((batch['original_frame_indices'][:,0]==0).all()))
                self.assertTrue(bool((batch['original_frame_indices'][:,1:]>=1).all()))
                self.assertTrue(bool((batch['original_frame_indices'][:,1:]<=374).all()))
                for row in range(5):
                    clip=int(batch['chunk_index'][row])-100
                    frames=batch['latents_low'][row,0,:,0,0]-1000*clip
                    self.assertEqual(float(frames[0]),0.)
                    self.assertGreaterEqual(float(frames[1]),1.)
                    self.assertGreaterEqual(float(frames[7]),7.)
                    torch.testing.assert_close(frames[1:],torch.arange(frames[1],frames[1]+10))
                    torch.testing.assert_close(frames[7:],torch.arange(frames[7],frames[7]+4))
                    torch.testing.assert_close(batch['pose_latents_low'][row],batch['latents_low'][row]+10000)
                    self.assertEqual(bool(batch['bad_audio'][row]),clip==1)
                    shifted_window_seen |= int(frames[-1])>374
                current=set(self.identities(batch))
                self.assertEqual(len(current),5)
                self.assertFalse(step_seen & current)
                self.assertFalse(epoch_seen & current)
                step_seen.update(current);epoch_seen.update(current)
            self.assertEqual(len(step_seen),40)  # 8 ranks * (4 critic rows + 1 student row).
        self.assertEqual(len(epoch_seen),200)
        self.assertTrue(shifted_window_seen)

    def test_epoch_order_is_reproducible_under_random_access(self):
        first, second = self.datasets(), self.datasets()
        size=len(first[0])
        for index in [size+3,0,4,size,2,0,size+3]:
            for rank in range(8):
                self.assertEqual(self.identities(first[rank][index]),self.identities(second[rank][index]))
        self.assertNotEqual(self.identities(first[0][0]),self.identities(first[0][size]))

    def test_forty_ranks_give_exact_global_batch_160(self):
        units=[rank_row_units(160,5,8,r) for r in range(8)]
        self.assertEqual(units,[(1,r,8) for r in range(8)])
        self.assertEqual(5*sum(row[0]*4 for row in units),160)
        with self.assertRaises(ValueError):
            rank_row_units(159,5,8,0)

    def test_history_masks_match_finetuning_parent_stream_and_resolver_views(self):
        source = FakeAloeSource()
        with patch('dataloading.ds_aloe.is_main_process', return_value=False), \
             patch('dataloading.ds_dense.is_main_process', return_value=False):
            resolver = ResolverWindows(source, 0, 8, 160, 5, 5,
                                       history_dropout=(.1, .1, 0.))
            parent = DenseWindows(source, 3, 4, [.1, .1, 0.], 0, 1, 1, 1)
        kinds = set()
        for epoch in range(4):
            for clip in range(2):
                for ordinal in range(2, 95):
                    _, actual = resolver._window(epoch, int(resolver._prefix[clip])+ordinal)
                    _, expected = parent._window(epoch, int(parent._prefix[clip])+ordinal)
                    torch.testing.assert_close(actual[4:7] >= 0, expected[1:4] >= 0)
                    # The resolver inherits slices of the parent mask, rather
                    # than independently drawing dropout for each target.
                    inherited = torch.stack([actual[5+j:7+j] >= 0 for j in range(4)])
                    reference = torch.ones(4, 2, dtype=torch.bool)
                    reference[0] = expected[2:4] >= 0
                    torch.testing.assert_close(inherited, reference)
                    self.assertTrue(bool((actual[7:] >= 0).all()))
                    kinds.add(tuple((actual[4:7] >= 0).tolist()))
        self.assertEqual(kinds, {(True, True, True), (False, True, True), (False, False, True)})


class ResolverScheduleTests(unittest.TestCase):
    def test_actual_optimizer_used_lr_ends_at_one_tenth_for_both_roles(self):
        peak=2e-6
        for role,total,warmup in [('student',1850,50),('critic',8000,100)]:
            with self.subTest(role=role):
                parameter=torch.nn.Parameter(torch.tensor(1.))
                optimizer=torch.optim.SGD([parameter],lr=peak)
                scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,
                    lambda update:learning_rate_factor(update,total,warmup,.1))
                used=[]
                for _ in range(total):
                    used.append(optimizer.param_groups[0]['lr'])
                    parameter.grad=torch.ones_like(parameter)
                    optimizer.step();scheduler.step()
                self.assertAlmostEqual(used[0],peak/warmup,places=18)
                self.assertAlmostEqual(max(used),peak,places=18)
                self.assertAlmostEqual(used[warmup-1],peak,places=18)
                self.assertAlmostEqual(used[-1],peak*.1,places=18)
                self.assertTrue(all(a<=b+1e-20 for a,b in zip(used[:warmup-1],used[1:warmup])))
                self.assertTrue(all(a>=b-1e-20 for a,b in zip(used[warmup:-1],used[warmup+1:])))


if __name__=='__main__':
    unittest.main()
