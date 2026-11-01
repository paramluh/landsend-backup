"""CPU two-rank DTensor proof; run with torchrun --standalone --nproc-per-node=2.

Covers old-world3 to new-world2 row boundaries, node-local file ownership,
DTensor AdamW state restoration and an exact subsequent optimizer update.
"""
import json
import os
from pathlib import Path
import shutil
import tempfile

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from finetune.duplex_distill.checkpoints import load_checkpoint, save_checkpoint, resume_checkpoint


def main():
    torch.set_num_threads(1)
    dist.init_process_group('gloo')
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 2
    root = [tempfile.mkdtemp(prefix='pose-distill-checkpoint-proof-') if rank == 0 else None]
    dist.broadcast_object_list(root, src=0)
    root = Path(root[0])
    mesh = init_device_mesh('cpu', (world,))
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(distribute_tensor(torch.zeros(7, 3), mesh, [Shard(0)]))
            self.replica = nn.Parameter(torch.zeros(2))
            self.register_buffer('counter', torch.tensor(3))
    if rank == 0:
        src = root / 'source';src.mkdir()
        weight = torch.arange(21.).reshape(7, 3) / 7
        for i in range(3):
            torch.save({'weight': weight[i * 3:(i + 1) * 3].bfloat16()}, src / f'rank{i}.pt')
        torch.save({'replica': torch.tensor([1., 2.]).bfloat16(), 'counter': torch.tensor(3)}, src / 'rep.pt')
        (src / 'meta.json').write_text(json.dumps({'format': 'ppm-local-bf16-shards-v1', 'world': 3, 'step': 10}))
        (src / 'COMPLETE').write_text('10')
    dist.barrier()
    a = Model();load_checkpoint(a, root / 'source', 'cpu')
    torch.testing.assert_close(a.weight.full_tensor(), (torch.arange(21.).reshape(7, 3) / 7).bfloat16().float(), atol=0, rtol=0)
    oa = torch.optim.AdamW(a.parameters(), lr=.01, foreach=False)
    for p in a.parameters():p.grad = torch.ones_like(p)
    oa.step();oa.zero_grad(set_to_none=True)
    saved = save_checkpoint(a, {'main': oa}, root, 1, {'checkpoint_reserve_gb': 0})
    b = Model();ob = torch.optim.AdamW(b.parameters(), lr=.9, foreach=False)
    resume_checkpoint(b, {'main': ob}, saved['resume_path'], 'cpu')
    for x, y in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(x.to_local() if hasattr(x, 'to_local') else x,
                                   y.to_local() if hasattr(y, 'to_local') else y, atol=0, rtol=0)
    for model, optimizer in ((a, oa), (b, ob)):
        for p in model.parameters():p.grad = torch.ones_like(p) * 2
        optimizer.step()
    for x, y in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(x.to_local() if hasattr(x, 'to_local') else x,
                                   y.to_local() if hasattr(y, 'to_local') else y, atol=0, rtol=0)
    c = Model();load_checkpoint(c, saved['export_path'], 'cpu')
    latest = save_checkpoint(a, {'main': oa}, root, 2,
                             {'checkpoint_reserve_gb': 0, 'keep_latest_optimizer_state': True})
    assert not Path(saved['resume_path']).exists()
    assert (Path(saved['export_path']) / 'NODE_COMPLETE').is_file()
    assert json.loads((root / 'checkpoints/LATEST_RESUME.json').read_text())['step'] == 2
    d = Model(); od = torch.optim.AdamW(d.parameters(), lr=.9, foreach=False)
    resume_checkpoint(d, {'main': od}, latest['resume_path'], 'cpu')
    for model, optimizer in ((a, oa), (d, od)):
        for p in model.parameters():p.grad = torch.ones_like(p) * 3
        optimizer.step()
    for x, y in zip(a.parameters(), d.parameters()):
        torch.testing.assert_close(x.to_local() if hasattr(x, 'to_local') else x,
                                   y.to_local() if hasattr(y, 'to_local') else y, atol=0, rtol=0)
    dist.barrier()
    if rank == 0:
        print('PASS: world3→world2 load; DTensor save/resume; exact AdamW updates; latest-resume retention; old export preserved', flush=True)
        shutil.rmtree(root)
    dist.destroy_process_group()


if __name__ == '__main__':main()
