"""Actual two-rank CPU FSDP proof of the chunked replay gradient lifecycle.

This uses the causally faithful small PPM from test_runtime, not the production
CUDA attention kernels. Decoder modules and every temporal module are genuinely
fully_shard-wrapped. Production's resident context and replay_update run without
mocks. The separate test_runtime decoder-hook test covers the native depth
driver's checkpoint/module-call mechanics.
"""
import copy
import datetime
import os
import tempfile
import time
import unittest
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule, fully_shard

from finetune.pose_opd_credit.config import TrainingConfig
from finetune.pose_opd_credit.runtime import replay_update
from finetune.pose_opd_credit.tests.test_runtime import ToyPPM, dense_expected, fixture


def _fsdp_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank,
                            world_size=2, timeout=datetime.timedelta(seconds=20))
    try:
        torch.manual_seed(498)
        models = [ToyPPM(), ToyPPM().requires_grad_(False), ToyPPM().requires_grad_(False)]
        references = copy.deepcopy(models)
        torch.manual_seed(810 + rank)
        # Rank0 has four generated media frames (two chunks); rank1 has seven
        # (three chunks). The short rank must execute a zero-weight final chunk.
        bundle, latents = fixture(frames=6 + 3 * rank)
        config = TrainingConfig(decoder_chunk_frames=3)
        mesh = init_device_mesh('cpu', (2,))
        for model in models:
            for child in model.children():
                fully_shard(child, mesh=mesh, reshard_after_forward=True)
            fully_shard(model, mesh=mesh, reshard_after_forward=True)
            for child in model.modules():
                if isinstance(child, FSDPModule):
                    # Gloo's CPU reduce_scatter does not implement the NCCL AVG
                    # behavior FSDP requests by default. SUM plus FSDP's explicit
                    # division tests identical averaging math. This override is
                    # confined to the test; production uses CUDA/NCCL normally.
                    child.set_force_sum_reduction_for_comms(True)

        saved_flags = []
        for model in models:
            for decoder in (model.dec_s, model.dec_p):
                state = decoder._get_fsdp_state()
                group = state._fsdp_param_group
                saved_flags.append((group, group.post_forward_mesh_info, group.reshard_after_backward))

        replay_update(*models, bundle, latents, config, None, torch.device('cpu'), world=2)
        # dense_expected uses world=1 with the actual all-reduced row count2,
        # so each rank contributes half of its row-mean objective. Summing these
        # dense gradients is the independent global reference for FSDP's mean.
        expected, _, _ = dense_expected(*references, bundle, latents, config)
        expected.backward()
        for (name, actual), (_, reference) in zip(models[0].named_parameters(), references[0].named_parameters()):
            if actual.grad is None or reference.grad is None:
                raise AssertionError(f'Missing student gradient: {name}')
            dist.all_reduce(reference.grad, op=dist.ReduceOp.SUM)
            global_gradient = actual.grad.full_tensor()
            torch.testing.assert_close(global_gradient, reference.grad, rtol=3e-5, atol=3e-6,
                                       msg=lambda message: f'{name}: {message}')
        for group, mesh_info, reshard in saved_flags:
            if group.post_forward_mesh_info is not mesh_info or group.reshard_after_backward != reshard:
                raise AssertionError('Resident decoder failed to restore FSDP flags')
        if any(p.grad is not None for model in models[1:] for p in model.parameters()):
            raise AssertionError('Teacher gradients must remain absent')
        for decoder in (models[0].dec_s, models[0].dec_p):
            # Exactly three first-pass chunks and three differentiable chunks
            # on both ranks, despite unequal numbers of useful frames.
            if len(decoder.calls) != 6 or sum(grad for _, grad in decoder.calls) != 3:
                raise AssertionError('Rank chunk calls are not aligned')
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'CPU Gloo is unavailable')
class DistributedReplayTests(unittest.TestCase):
    def test_nested_fsdp_resident_two_pass_replay_matches_global_dense_gradient(self):
        with tempfile.TemporaryDirectory(prefix='pose-opd-fsdp-') as directory, \
             mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}):
            context = mp.start_processes(_fsdp_worker, args=(directory + '/rendezvous',),
                                         nprocs=2, start_method='spawn', join=False)
            deadline = time.monotonic() + 30
            try:
                while not context.join(timeout=1):
                    if time.monotonic() >= deadline:
                        self.fail('CPU FSDP replay workers exceeded their timeout')
            finally:
                for process in context.processes:
                    if process.is_alive():
                        process.terminate()
                for process in context.processes:
                    process.join(timeout=5)


if __name__ == '__main__':
    unittest.main()
