"""Two-GPU FSDP2 AdamW parity proof for host-parking optimizer moments."""
from __future__ import annotations

import argparse
import datetime
import io
import json
from pathlib import Path
import tempfile

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import DTensor
import torch.multiprocessing as mp

from .optimizer_parking import OptimizerStateParking


class _Model(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.first = torch.nn.Linear(256, 256, device=device)
        self.later = torch.nn.Linear(256, 256, device=device)

    def forward(self, inputs, *, both):
        hidden = self.first(inputs).tanh()
        return self.later(hidden) if both else hidden


def _compare(models, optimizers):
    checks = 0
    for left, right in zip(models[0].parameters(), models[1].parameters()):
        torch.testing.assert_close(left.to_local(), right.to_local(), atol=0, rtol=0)
        checks += 1
        lstate, rstate = optimizers[0].state.get(left, {}), optimizers[1].state.get(right, {})
        assert lstate.keys() == rstate.keys()
        for key in lstate:
            lv, rv = lstate[key], rstate[key]
            assert type(lv) is type(rv), key
            if isinstance(lv, DTensor):
                assert lv.shape == rv.shape and lv.stride() == rv.stride(), key
                assert lv.placements == rv.placements and lv.device_mesh == rv.device_mesh, key
                lv, rv = lv.to_local(), rv.to_local()
            torch.testing.assert_close(lv, rv, atol=0, rtol=0)
            checks += 1
    return checks


def _worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    dist.init_process_group('nccl', init_method='file://' + rendezvous, rank=rank, world_size=2,
                            timeout=datetime.timedelta(seconds=90))
    try:
        mesh = init_device_mesh('cuda', (2,))
        models = []
        for _ in range(2):
            torch.manual_seed(498)
            model = _Model(device)
            policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
            for child in model.children():
                fully_shard(child, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
            fully_shard(model, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
            models.append(model)
        optimizers = [torch.optim.AdamW(m.parameters(), lr=3e-6, betas=(.9, .95),
                                       eps=1e-8, weight_decay=0., fused=True) for m in models]
        checks = 0
        records = []
        for step in range(3):
            torch.manual_seed(910 + step + rank)
            inputs = torch.randn(8, 256, device=device)
            target = torch.randn(8, 256, device=device)
            # Park the second optimizer at state sizes0, partially initialized,
            # and fully initialized. One restoration follows a rollout exception.
            parking = OptimizerStateParking(optimizers[1], pin_memory=bool(step % 2))
            before = torch.cuda.memory_allocated(device)
            try:
                with parking:
                    torch.cuda.synchronize(device)
                    during = torch.cuda.memory_allocated(device)
                    for state in optimizers[1].state.values():
                        for value in state.values():
                            if isinstance(value, torch.Tensor):
                                assert type(value) is torch.Tensor and value.device.type == 'cpu'
                    for operation in (optimizers[1].step, optimizers[1].state_dict):
                        try:
                            operation()
                            raise AssertionError('Parked optimizer operation was not guarded')
                        except RuntimeError as error:
                            assert 'parked' in str(error)
                    if step == 2:
                        raise ValueError('simulated rollout exception')
            except ValueError as error:
                assert str(error) == 'simulated rollout exception'
            assert not parking.parked
            checks += _compare(models, optimizers)
            if step:
                # CUDA allocation bins round scalar steps and small moments up
                # to512-byte blocks; logical tensor bytes are a lower bound.
                assert before - during >= parking.stats['cuda_bytes_released'], (before, during, parking.stats)
            records.append(dict(step=step, bytes_before=before, bytes_while_parked=during, **parking.stats))
            for model, optimizer in zip(models, optimizers):
                result = model(inputs, both=step > 0)
                (result.float() - target).square().mean().backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            checks += _compare(models, optimizers)

        buffer = io.BytesIO()
        torch.save(optimizers[1].state_dict(), buffer)
        buffer.seek(0)
        loaded = torch.load(buffer, weights_only=False)
        optimizers[1].load_state_dict(loaded)
        checks += _compare(models, optimizers)
        result = {'rank': rank, 'status': 'passed', 'exact_tensor_comparisons': checks,
                  'updates': 3, 'fused_adamw': True, 'fsdp2': True, 'bf16_forward_fp32_reduce': True,
                  'empty_partial_full_state_tested': True, 'exception_restoration_tested': True,
                  'checkpoint_roundtrip_after_restore': True, 'parking': records}
        Path(output, f'rank-{rank}.json').write_text(json.dumps(result, indent=2) + '\n')
    finally:
        dist.destroy_process_group()


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args(argv)
    if torch.cuda.device_count() < 2:
        raise RuntimeError('This check requires two visible CUDA devices')
    with tempfile.TemporaryDirectory(prefix='pose-opd-optimizer-parking-') as directory:
        mp.spawn(_worker, args=(directory + '/rendezvous', directory), nprocs=2, join=True)
        results = [json.loads(Path(directory, f'rank-{rank}.json').read_text()) for rank in range(2)]
    report = {'status': 'passed', 'ranks': results}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
