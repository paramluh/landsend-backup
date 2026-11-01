"""
Checkpoints when only global rank 0 has a run dir (no shared filesystem). Same files as utils.harness
(model.safetensors, model_ema.safetensors, optim_rank{r}.pt, optim_meta.pt), all on rank 0: every
rank's EMA shadows and ZeRO-1 shard (masters, moments, EMA, RNG states) travel to rank 0 over the gloo
group (init_gloo) as CPU tensors, one rank at a time. On resume, rank 0 reads the meta and the shards and
sends each shard to its owner; trainable weights come back from the masters, so ranks start from the
initial checkpoint_path. Once an optimizer save is complete, the optimizer files of older steps in the
run dir are deleted.
Cross-node NCCL p2p of these buffers died on landsend with IBV_WC_LOC_ACCESS_ERR while training
collectives were fine, so no saver transfer goes over NCCL.
"""
import shutil
from datetime import timedelta

import safetensors.torch
import torch
import torch.distributed as dist

GLOO_TIMEOUT = timedelta(hours=1)
gloo = None


def init_gloo():
    global gloo
    gloo = dist.new_group(backend='gloo', timeout=GLOO_TIMEOUT)


class _Ref:
    def __init__(self, tensor):
        self.shape, self.dtype = tensor.shape, tensor.dtype


def _split(obj, tensors):
    if torch.is_tensor(obj):
        tensors.append(obj)
        return _Ref(obj)
    if isinstance(obj, dict):
        return {k: _split(v, tensors) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_split(v, tensors) for v in obj)
    return obj


def _join(obj, fill):
    if isinstance(obj, _Ref):
        return fill(obj)
    if isinstance(obj, dict):
        return {k: _join(v, fill) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_join(v, fill) for v in obj)
    return obj


def _send(obj, dst):
    """Pickling multi-GB shards is slow, so only the skeleton is pickled; tensors go raw."""
    tensors = []
    dist.send_object_list([_split(obj, tensors)], dst=dst, group=gloo)
    for tensor in tensors:
        dist.send(tensor.detach().cpu().contiguous(), dst=dst, group=gloo)


def _recv(src):
    box = [None]
    dist.recv_object_list(box, src=src, group=gloo)

    def fill(ref):
        buffer = torch.empty(ref.shape, dtype=ref.dtype)
        dist.recv(buffer, src=src, group=gloo)
        return buffer
    return _join(box[0], fill)


def _to_rank0(obj, src):
    rank = dist.get_rank()
    if src == 0:
        return obj
    if rank == src:
        _send(obj, 0)
    elif rank == 0:
        return _recv(src)
    return None


def _from_rank0(obj, dst):
    rank = dist.get_rank()
    if dst == 0:
        return obj
    if rank == 0:
        _send(obj, dst)
    elif rank == dst:
        return _recv(0)
    return None


def save(save_dir, model, layered, ema, optimizer, config, config_path, save_optim, step, examples,
         steps_per_epoch, wandb_run=None):
    rank, world = dist.get_rank(), dist.get_world_size()
    if rank == 0:
        print(f'Saving model to {save_dir}')
        save_dir.mkdir(parents=True, exist_ok=True)
        state_dict = {
            p.original_name: p.detach().to('cpu', config['save_dtype'])
            for p in layered.parameters() if hasattr(p, 'original_name')
        }
        model.save_model(save_dir, state_dict)
        shutil.copy(config_path, save_dir)
    ema_state = {}
    for src in range(world):
        shard = None
        if ema is not None:
            shard = {name: s.to('cpu', config['save_dtype']) for name, _, s in ema.triples} if rank == src else None
            shard = _to_rank0(shard, src)
            if rank == 0:
                ema_state.update(shard)
        if save_optim:
            if rank == src:
                shard = {'optim': optimizer.state_dict(), 'rng': (torch.get_rng_state(), torch.cuda.get_rng_state())}
                if ema is not None:
                    shard['ema'] = ema.state_dict()
            shard = _to_rank0(shard, src)
            if rank == 0:
                torch.save(shard, save_dir / f'optim_rank{src:04d}.pt')
    if rank == 0:
        if ema is not None:
            safetensors.torch.save_file(ema_state, save_dir / 'model_ema.safetensors', metadata={'format': 'pt'})
        if save_optim:
            meta = {'world_size': world, 'step': step, 'examples': examples, 'steps_per_epoch': steps_per_epoch}
            if wandb_run is not None:
                meta['wandb_id'] = wandb_run.id
                meta['wandb_step'] = wandb_run.step - 1
            torch.save(meta, save_dir / 'optim_meta.pt')
    dist.barrier(group=gloo)
    if rank == 0 and save_optim:
        for old in save_dir.parent.glob('step*/optim_*.pt'):
            if old.parent != save_dir:
                old.unlink()


def load_meta(resume_dir):
    box = [torch.load(resume_dir / 'optim_meta.pt', weights_only=True) if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(box, src=0)
    return box[0]


def load(resume_dir, meta, optimizer, ema, steps_per_epoch):
    assert meta['world_size'] == dist.get_world_size(), \
        f"optimizer state was saved at world_size={meta['world_size']}, now {dist.get_world_size()}"
    assert meta['steps_per_epoch'] == steps_per_epoch, \
        f"saved steps_per_epoch={meta['steps_per_epoch']}, now {steps_per_epoch}: dataset or batch size changed"
    mine = None
    for dst in range(dist.get_world_size()):
        shard = torch.load(resume_dir / f'optim_rank{dst:04d}.pt', map_location='cpu', weights_only=True) \
            if dist.get_rank() == 0 else None
        shard = _from_rank0(shard, dst)
        if dist.get_rank() == dst:
            mine = shard
        del shard
    dist.barrier(group=gloo)
    optimizer.load_state_dict(mine['optim'])
    if ema is not None:
        ema.load_state_dict(mine['ema'])
    torch.set_rng_state(mine['rng'][0])
    torch.cuda.set_rng_state(mine['rng'][1])
