import math
import os
import shutil
from contextlib import contextmanager
from pathlib import Path

import safetensors.torch
import torch
import torch.distributed as dist
from optimi import StableAdamW

from utils.common import is_main_process


def get_next_run_dir(output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    existing = [int(d.name.split('_')[1]) for d in out.iterdir()
                if d.is_dir() and d.name.startswith('run_') and d.name.split('_')[1].isdigit()]
    return str(out / f'run_{max(existing, default=0) + 1}')


def get_most_recent_run_dir(output_dir):
    return str(sorted(
        (d for d in Path(output_dir).iterdir() if d.is_dir() and d.name.startswith('run_')),
        key=lambda d: int(d.name.split('_')[1]),
    )[-1])


class Zero1FP32MasterAdamW:
    """
    Params and DDP gradient buckets keep their dtype. Each rank owns a numel-balanced
    shard (balanced per dtype, largest params first), holding fp32 masters + optimizer
    state for it only; whole tensors are assigned, never split, so StableAdamW's
    per-tensor update clipping is exact under sharding. Every parameter's storage is a
    view into one flat buffer per dtype laid out [world, shard], so step() ends in a
    single in-place all_gather per dtype with no staging copies: bf16 grads up to the
    masters, inner step, masters down into the flat, all_gather. make_inner builds the
    inner optimizer over this rank's master groups; groups that own nothing on this
    rank are dropped (optimi rejects empty groups) and group_ids records the survivors.
    """

    def __init__(self, param_groups, make_inner):
        self.rank = dist.get_rank()
        world = dist.get_world_size()
        owner_of = {}
        by_dtype = {}
        for group in param_groups:
            for p in group['params']:
                by_dtype.setdefault(p.dtype, []).append(p)
        self.flats = {}  # dtype -> flat [world * shard] buffer that every parameter of that dtype views into
        for dtype, params in by_dtype.items():
            sizes = [0] * world
            shards = [[] for _ in range(world)]
            for p in sorted(params, key=lambda p: -p.numel()):
                owner = min(range(world), key=sizes.__getitem__)
                sizes[owner] += p.numel()
                shards[owner].append(p)
                owner_of[p] = owner
            shard = max(sizes)
            flat = torch.empty(world * shard, dtype=dtype, device=params[0].device)
            for owner, owned in enumerate(shards):
                offset = owner * shard
                for p in owned:
                    view = flat[offset:offset + p.numel()].view_as(p)
                    view.copy_(p.data)
                    p.data = view
                    offset += p.numel()
            self.flats[dtype] = flat

        self.per_rank_params = [[] for _ in range(world)]
        self.pairs = []  # (parameter, fp32 master) for this rank's shard
        self.group_ids = []
        master_groups = []
        for i, group in enumerate(param_groups):
            master_group = {k: v for k, v in group.items() if k != 'params'} | {'params': []}
            for p in group['params']:
                self.per_rank_params[owner_of[p]].append(p)
                if owner_of[p] == self.rank:
                    m = p.detach().clone().float()
                    master_group['params'].append(m)
                    self.pairs.append((p, m))
            if master_group['params']:
                self.group_ids.append(i)
                master_groups.append(master_group)
        self.inner = make_inner(master_groups)

    def step(self):
        for p, m in self.pairs:
            m.grad = p.grad.detach().float()
        self.inner.step()
        for p, m in self.pairs:
            p.data.copy_(m.data)
            m.grad = None  # the fp32 grad copy is dead until the next step; give the memory back to activations
        self._gather_shards()

    def _gather_shards(self):
        world = dist.get_world_size()
        for flat in self.flats.values():
            shard = flat.numel() // world
            dist.all_gather_into_tensor(flat, flat[self.rank * shard:(self.rank + 1) * shard])

    def state_dict(self):
        return {'masters': [m.detach().cpu() for _, m in self.pairs], 'inner': self.inner.state_dict()}

    def load_state_dict(self, sd):
        assert len(sd['masters']) == len(self.pairs), \
            f"shard holds {len(sd['masters'])} params, this rank owns {len(self.pairs)}"
        for i, (p, m) in enumerate(self.pairs):
            assert sd['masters'][i].shape == m.shape, \
                f"shard param {i}: shape {tuple(sd['masters'][i].shape)} != {tuple(m.shape)}"
            m.data.copy_(sd['masters'][i])
        self.inner.load_state_dict(sd['inner'])  # torch moves state tensors onto the master's device (fused AdamW's step included)
        for p, m in self.pairs:
            p.data.copy_(m.data)
        self._gather_shards()

    def zero_grad(self):
        # In place: the grads are DDP bucket views, and setting them to None would make the next no_sync()
        # backward allocate a second full copy of the grads alongside the buckets.
        torch._foreach_zero_([p.grad for params in self.per_rank_params for p in params])


class EMA:
    """fp32 shadows of this rank's masters; the union over ranks covers every original_name parameter."""

    def __init__(self, decay, pairs):
        self.decay = decay
        named = [(p, m) for p, m in pairs if hasattr(p, 'original_name')]
        self.triples = [(p.original_name, m, m.clone()) for p, m in named]
        self._params = [p for p, _ in named]
        self._masters = [m for _, m, _ in self.triples]
        self._shadows = [s for _, _, s in self.triples]

    def update(self):
        torch._foreach_mul_(self._shadows, self.decay)
        torch._foreach_add_(self._shadows, self._masters, alpha=1 - self.decay)

    @contextmanager
    def swapped_in(self, optimizer, save_dtype):
        """
        Every rank's parameters hold model_ema.safetensors (the shadows cast to save_dtype, as saved),
        written in place through the ZeRO-1 flats like optimizer.step(); the raw flats wait on the CPU
        and come back bit-exact on exit. Parameters outside the optimizer are frozen and stay as they are.
        """
        parked = [flat.cpu() for flat in optimizer.flats.values()]
        for p, s in zip(self._params, self._shadows):
            p.data.copy_(s.to(save_dtype))
        optimizer._gather_shards()
        try:
            yield
        finally:
            for flat, raw in zip(optimizer.flats.values(), parked):
                flat.copy_(raw)

    def state_dict(self):
        return [s.cpu() for _, _, s in self.triples]

    def load_state_dict(self, shadows):
        assert len(shadows) == len(self.triples), \
            f'shard holds {len(shadows)} EMA shadows, this rank owns {len(self.triples)}'
        for saved, (_, _, s) in zip(shadows, self.triples):
            s.copy_(saved)

    def save(self, save_dir, save_dtype):
        tmp_dir = save_dir / 'tmp_ema'
        if is_main_process():
            tmp_dir.mkdir(parents=True, exist_ok=True)
        dist.barrier()
        shard = {name: s.to('cpu', save_dtype) for name, _, s in self.triples}
        torch.save(shard, tmp_dir / f'state_dict_{dist.get_rank()}.bin')
        dist.barrier()
        if is_main_process():
            state_dict = {}
            for path in tmp_dir.glob('*.bin'):
                state_dict.update(torch.load(path, map_location='cpu', weights_only=True))
            safetensors.torch.save_file(state_dict, save_dir / 'model_ema.safetensors', metadata={'format': 'pt'})
            shutil.rmtree(tmp_dir)


def build_ema(config, optimizer, layered):
    if 'ema_decay' not in config:
        return None
    # model_ema.safetensors must carry the same keys as model.safetensors: a frozen
    # original_name parameter would sit outside the optimizer and be missing from the merge.
    assert (sum(hasattr(p, 'original_name') for ps in optimizer.per_rank_params for p in ps)
            == sum(hasattr(p, 'original_name') for p in layered.parameters())), \
        'every original_name parameter must be trainable, or model_ema would miss keys'
    return EMA(config['ema_decay'], optimizer.pairs)


def build_optimizer(config, model, parameters):
    optim_config = config['optimizer']
    kwargs = {k: v for k, v in optim_config.items() if k not in ('type', 'groups')}
    if optim_config['type'] == 'adamw':
        make_inner = lambda groups: torch.optim.AdamW(groups, lr=config['lr'], fused=True, **kwargs)
    elif optim_config['type'] == 'stableadamw':
        kwargs['betas'] = tuple(kwargs['betas'])
        make_inner = lambda groups: StableAdamW(groups, lr=config['lr'], triton=True, **kwargs)
    else:
        raise ValueError(f"optimizer.type must be adamw or stableadamw, got {optim_config['type']}")
    return Zero1FP32MasterAdamW(model.get_param_groups(parameters), make_inner)


def build_lr_scheduler(config, optimizer, steps_per_epoch):
    """
    Per-group warmup / hold / cosine. Each [[optimizer.groups]] entry may set
    warmup_steps and decay_start: ramp to peak over warmup_steps, hold until
    decay_start, cosine to min_lr at decay_end. Lambda order matches
    get_param_groups (default group first, then config groups), filtered to the
    groups this rank's inner optimizer actually holds.
    """
    total_steps = config['epochs'] * steps_per_epoch
    decay_end = config['lr_decay_epochs'] * steps_per_epoch if 'lr_decay_epochs' in config else total_steps

    def make_cosine(lr, min_lr, warmup_steps, decay_start=None):
        start = decay_start if decay_start is not None else warmup_steps
        assert start >= warmup_steps
        floor = min_lr / lr if lr > 0 else 0.0
        decay_steps = decay_end - start

        def fn(step):
            if warmup_steps > 0 and step < warmup_steps:
                return step / warmup_steps
            if step < start:
                return 1.0
            p = min((step - start) / decay_steps, 1.0) if decay_steps > 0 else 1.0
            return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p))

        return fn

    groups = config['optimizer'].get('groups', [])
    lambdas = [make_cosine(config['lr'], config['min_lr'], config['warmup_steps'])]
    for g in groups:
        lambdas.append(make_cosine(
            g.get('lr', config['lr']), g.get('min_lr', config['min_lr']),
            g.get('warmup_steps', config['warmup_steps']), g.get('decay_start'),
        ))
    lambdas = [lambdas[i] for i in optimizer.group_ids]
    return torch.optim.lr_scheduler.LambdaLR(optimizer.inner, lambdas)


def save_model_dir(save_dir, model, layered, ema, config, config_path):
    if is_main_process():
        print(f'Saving model to {save_dir}')
        save_dir.mkdir(parents=True, exist_ok=False)
        state_dict = {
            p.original_name: p.detach().to('cpu', config['save_dtype'])
            for p in layered.parameters() if hasattr(p, 'original_name')
        }
        model.save_model(save_dir, state_dict)
        shutil.copy(config_path, save_dir)
    dist.barrier()
    if ema is not None:
        ema.save(save_dir, config['save_dtype'])


def save_optim_state(save_dir, optimizer, ema, step, examples, steps_per_epoch, wandb_run=None):
    """Each rank writes its ZeRO-1 shard (masters, moments, EMA shadows); rank 0 adds optim_meta.pt,
    with the wandb run id and its last logged history row so a resume can rewind the run to here.
    Once every rank has written, each deletes its own files from older steps in the run dir."""
    shard = {'optim': optimizer.state_dict()}
    if ema is not None:
        shard['ema'] = ema.state_dict()
    torch.save(shard, save_dir / f'optim_rank{optimizer.rank:04d}.pt')
    if optimizer.rank == 0:
        meta = {'world_size': dist.get_world_size(), 'step': step, 'examples': examples, 'steps_per_epoch': steps_per_epoch}
        if wandb_run is not None:
            meta['wandb_id'] = wandb_run.id
            meta['wandb_step'] = wandb_run.step - 1
        torch.save(meta, save_dir / 'optim_meta.pt')
    dist.barrier()
    mine = {f'optim_rank{optimizer.rank:04d}.pt'} | ({'optim_meta.pt'} if optimizer.rank == 0 else set())
    for old in save_dir.parent.glob('step*/optim_*.pt'):
        if old.name in mine and old.parent != save_dir:
            old.unlink()


def wandb_resume_kwargs(config, meta):
    """Rewind truncates the run's history after the checkpoint's row and continues it; a hand-set
    [wandb] id (checkpoints from before wandb_id was saved) appends to that run instead."""
    if 'wandb_id' in meta:
        return {'resume_from': f"{meta['wandb_id']}?_step={meta['wandb_step']}"}
    if 'id' in config.get('wandb', {}):
        return {'id': config['wandb']['id'], 'resume': 'must'}
    return {}


def load_optim_state(resume_dir, optimizer, ema, steps_per_epoch):
    meta = torch.load(resume_dir / 'optim_meta.pt', weights_only=True)
    assert meta['world_size'] == dist.get_world_size(), \
        f"optimizer state was saved at world_size={meta['world_size']}, now {dist.get_world_size()}"
    # The data position is step * GA items: a changed dataset or batch size would resume at the wrong item.
    assert meta['steps_per_epoch'] == steps_per_epoch, \
        f"saved steps_per_epoch={meta['steps_per_epoch']}, now {steps_per_epoch}: dataset or batch size changed"
    shard = torch.load(resume_dir / f'optim_rank{optimizer.rank:04d}.pt', map_location='cpu', weights_only=True)
    optimizer.load_state_dict(shard['optim'])
    if ema is not None:
        ema.load_state_dict(shard['ema'])
    return meta


MANUAL_SIGNALS = ('save', 'save_o', 'save_q')


def check_manual_signals(run_dir):
    """Rank 0 stats and removes the signal files; the flags are broadcast so every rank takes the same branch."""
    signals = torch.zeros(len(MANUAL_SIGNALS), dtype=torch.int64, device='cuda')
    if is_main_process():
        for i, name in enumerate(MANUAL_SIGNALS):
            if (run_dir / name).exists():
                signals[i] = 1
                os.remove(run_dir / name)
    dist.broadcast(signals, src=0)
    return tuple(bool(flag) for flag in signals.tolist())
