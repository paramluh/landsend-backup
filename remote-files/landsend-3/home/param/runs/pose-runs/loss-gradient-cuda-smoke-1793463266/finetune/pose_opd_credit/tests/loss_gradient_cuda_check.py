"""Tiny two-GPU FSDP2 proof of cached loss probes and exact Adam continuation.

Run explicitly with torchrun; this does not load production weights or launch
rollouts. The small causal PPM fixture uses CUDA autocast, actual nested FSDP2
and NCCL, including resident decoder gradients and two learner microbatches.
"""
import copy
import datetime
import functools
import json
import os
from pathlib import Path
import time
from unittest import mock

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

from finetune.pose_opd_credit import runtime
from finetune.pose_opd_credit.config import TrainingConfig
from finetune.pose_opd_credit.loss_gradient_probe import measure_loss_gradient_norms
from finetune.pose_opd_credit.train import optimizer_parameter_groups
from finetune.pose_opd_credit.tests.test_runtime import ToyPPM, ToyDecoder, fixture


def _cuda_factories(function):
    # The CPU fixture creates arange indexes without explicit devices. Scope
    # only its forward factories to CUDA; host diagnostic buffers stay on CPU.
    @functools.wraps(function)
    def wrapped(self, *args, **kwargs):
        with torch.device(next(self.parameters()).device):
            result = function(self, *args, **kwargs)
            # The production depth forward exposes FP32 logits even under
            # autocast; the CPU-only toy's einsum otherwise returns BF16 here.
            return (result[0].float(), result[1]) if isinstance(self, ToyDecoder) else result
    return wrapped


def _local(value):
    return value.to_local() if hasattr(value, 'to_local') else value


def main():
    rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    torch.set_num_threads(1)
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=90))
    started = time.monotonic()
    try:
        torch.manual_seed(424)
        ToyPPM.forward = _cuda_factories(ToyPPM.forward)
        ToyDecoder.forward = _cuda_factories(ToyDecoder.forward)
        student = ToyPPM().to(device)
        control = copy.deepcopy(student)
        audio, pose = ToyPPM().to(device).requires_grad_(False), ToyPPM().to(device).requires_grad_(False)
        mesh = init_device_mesh('cuda', (dist.get_world_size(),))
        for model in (student, control, audio, pose):
            for child in model.children():
                fully_shard(child, mesh=mesh, reshard_after_forward=True)
            fully_shard(model, mesh=mesh, reshard_after_forward=True)
            model.eval()
        config = TrainingConfig(decoder_chunk_frames=3, text_chunk_frames=2,
            effective_batch_per_rank=2, physical_microbatch=1, teacher_batch_per_rank=2,
            teacher_depth_cache='cpu', pose_cfg_batch=True, pose_teacher_cfg_scale=3,
            pose_teacher_silence_path='fixture.npy', pose_teacher_silence_sha256='0' * 64,
            decoder_gradient_sync='per_replay', pose_light_learning_rate=8e-6)
        rows, latents = map(list, zip(*(fixture(frames=frames + rank) for frames in (6, 8))))
        normalization = runtime.update_normalization(rows, device)
        with mock.patch.object(runtime, 'load_silence_codes', return_value=np.zeros((32, 32), np.int64)):
            prepared = runtime.prepare_teacher_update(audio, pose, rows, latents, config, None, device,
                                                      world=dist.get_world_size(), normalization=normalization)
            caches = []
            for index in range(2):
                cache = {}
                for model in (student, control):
                    runtime.replay_update(model, audio, pose, rows[index:index + 1], latents[index:index + 1],
                        config, None, device, world=dist.get_world_size(), normalization=normalization,
                        prepared_teachers=prepared.slice_rows(index, index + 1),
                        loss_gradient_cache=cache if model is student else None)
                caches.append(cache)
        original = [(None if p.grad is None else _local(p.grad).clone(),
                     None if p.grad is None else (p.grad.placements, p.grad.shape, p.grad.stride()))
                    for p in student.parameters()]
        for actual, expected in zip(student.parameters(), control.parameters(), strict=True):
            torch.testing.assert_close(_local(actual.grad), _local(expected.grad), rtol=0., atol=0.)

        def callback(term):
            for cache in caches:
                runtime.replay_loss_gradient(student, cache, term, None, device)

        with mock.patch.object(audio, 'forward', side_effect=AssertionError('Teacher forward during probe')), \
                mock.patch.object(pose, 'forward', side_effect=AssertionError('Teacher forward during probe')), \
                mock.patch.object(runtime, 'scored_categorical_statistics', side_effect=AssertionError('Student rescored')), \
                mock.patch.object(runtime, 'causal_credit_weights', side_effect=AssertionError('Credit recomputed')):
            metrics = measure_loss_gradient_norms(student, callback, device)
        for parameter, (value, layout) in zip(student.parameters(), original, strict=True):
            if value is None:
                assert parameter.grad is None
            else:
                torch.testing.assert_close(_local(parameter.grad), value, rtol=0., atol=0.)
                assert (parameter.grad.placements, parameter.grad.shape, parameter.grad.stride()) == layout
        for model in (student, control):
            optimizer = torch.optim.AdamW(optimizer_parameter_groups(model, config),
                                         betas=(.9, .95), eps=1e-8, weight_decay=0., fused=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
        for actual, expected in zip(student.parameters(), control.parameters(), strict=True):
            torch.testing.assert_close(_local(actual), _local(expected), rtol=0., atol=0.)
        torch.cuda.synchronize(device)
        receipt = {'rank': rank, 'passed': True, 'world': dist.get_world_size(),
                   'device': torch.cuda.get_device_name(device), 'seconds': time.monotonic() - started,
                   'diagnostic_seconds': metrics['loss_gradient_diagnostics_seconds'],
                   'extra_backward_passes': metrics['loss_gradient_diagnostics_extra_backwards'],
                   'peak_allocated_mib': torch.cuda.max_memory_allocated(device) / 2 ** 20,
                   'exact_gradient_restore': True, 'exact_next_adam_update': True,
                   'teacher_cfg_scale': 3, 'learner_microbatches': 2,
                   'decoder_gradient_sync': 'per_replay'}
        Path(f'rank{rank}.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print(json.dumps(receipt), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
