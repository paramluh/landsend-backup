"""
DMD distillation of the bimodal model (student, frozen teacher, flow-matching critic; all from one checkpoint).

One outer step per rank, rows [sink | GT 1..3 | targets 4..7] (distillation/bimodal_dmd.py):
  1. critic pool: the student rolls out pool_rows rows without gradients (stabilised and resolved frames);
  2. critic_updates critic updates of critic_rows rows each: FM on the stabilising window (generated frames at
     history_t, re-noised into the bands) weighted 0.8 and on the resolving windows (resolved frames, t ~ U) 0.2;
  3. one student update of student_kinds microbatches of one row: 'stab' rolls out to history_t only and is scored
     in the stabilising window [sink | GT history at history_t | 4 present]; 'res' rolls out fully and each frame is
     scored independently in its resolving window. Velocity DMD with the per-frame teacher-error normalizer.
Weights are replicated bf16 for compute; fp32 masters and AdamW state are ZeRO-1 sharded (utils.harness), the
gradients reduce-scattered to their owners. critic_warmup_steps outer steps run the critic only.

  torchrun ... learn_dmd_bimodal.py --config experiments/distillation/aloe-studio-bimodal-dmd.toml
"""

import argparse
import gc
import json
import math
import os
import random
import shutil
import time
import tomllib
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch.utils.data import DataLoader

from dataloading.ds_aloe import AloeChunkSource, ShuffledDenseWindows
from distillation.bimodal_dmd import (
    H, P, Rollout, UniformRollout, UniformStabilizerRollout, both_pass, dmd_loss, fm_loss, noised, resolve_pass, resolve_times, stab_pass, stab_times,
)
from distillation.bimodal_net import audio_latents, build_net, load_mimi, load_state_dict
from distillation.checkpoints import (
    RESUME_KEYS, compatibility, complete_checkpoint, distribution_fingerprint, file_sha256, load_model_weights, load_rank_state,
    prune_optimizer_state, read_metadata, save_ema, save_rank_states, tensor_bytes, validate_resume,
)
from utils import saver
from utils.harness import EMA, Zero1FP32MasterAdamW
from distillation.lora import inject_lora

STAB_BUCKETS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
RES_BUCKETS = [0.02, 0.1, 0.2, 0.3, 0.4]


class SeededWindows(ShuffledDenseWindows):
    """ShuffledDenseWindows' global shuffle over full rows only: frames start-3..start-1 as GT seeds and
    start..start+3 as targets, start = 4, 8, ...; no startup windows, pads or history dropout."""

    def __init__(self, source, rank, world_size, rows, min_over=None):
        super().__init__(source, H, P, [0.0] * H, rank, world_size, rows, 1)
        counts = np.maximum(0, (source.lengths - 1 - H) // P)
        self._prefix = np.zeros(len(counts) + 1, dtype=np.int64)
        np.cumsum(counts, out=self._prefix[1:])
        self._windows_per_epoch = int(self._prefix[-1]) // (rows * world_size) * (rows * world_size)
        if min_over is not None:
            count = torch.tensor(self._windows_per_epoch)
            dist.all_reduce(count, op=dist.ReduceOp.MIN, group=min_over)
            self._windows_per_epoch = int(count)
        if not self._windows_per_epoch:
            raise ValueError('No full DMD batch in the smallest data shard')
        self._perm_epoch = None

    def _window(self, epoch, window_index):
        clip = int(np.searchsorted(self._prefix, window_index, side='right') - 1)
        start = 1 + H + (window_index - int(self._prefix[clip])) * P
        return clip, torch.cat((torch.zeros(1, dtype=torch.long), torch.arange(start - H, start + P)))


class GradFlats:
    """Gradients as views into flats laid out like the optimizer's parameter flats; reduce-scatter to owners."""

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.world, self.rank = dist.get_world_size(), dist.get_rank()
        self.offsets = {}
        for dtype, flat in optimizer.flats.items():
            for params in optimizer.per_rank_params:
                for p in params:
                    if p.dtype == dtype:
                        self.offsets[p] = (dtype, (p.data_ptr() - flat.data_ptr()) // flat.element_size())
        self.grads = None

    def allocate(self):
        self.grads = {dtype: torch.zeros_like(flat) for dtype, flat in self.optimizer.flats.items()}
        for p, (dtype, offset) in self.offsets.items():
            p.grad = self.grads[dtype][offset:offset + p.numel()].view_as(p)

    def reduce_clip(self, max_norm):
        for g in self.grads.values():
            shard = g.numel() // self.world
            out = torch.empty(shard, dtype=g.dtype, device=g.device)
            dist.reduce_scatter_tensor(out, g, op=dist.ReduceOp.AVG)
            g[self.rank * shard:(self.rank + 1) * shard].copy_(out)
        owned = [p.grad for p, _ in self.optimizer.pairs]
        norm_sq = torch.zeros((), device=owned[0].device, dtype=torch.float32)
        for g in owned:
            norm_sq += g.float().square().sum()
        dist.all_reduce(norm_sq)
        norm = norm_sq.sqrt()
        scale = (max_norm / (norm + 1e-6)).clamp(max=1.0)
        torch._foreach_mul_(owned, scale.item())
        return norm

    def release(self):
        for p in self.offsets:
            p.grad = None
        self.grads = None


class Sums:
    def __init__(self):
        self.values = {}

    def add(self, name, total, count):
        total = torch.as_tensor(total, dtype=torch.float64, device='cuda').detach().double()
        count = torch.as_tensor(count, dtype=torch.float64, device='cuda').double()
        if name in self.values:
            self.values[name][0].add_(total)
            self.values[name][1].add_(count)
        else:
            self.values[name] = [total.clone(), count.clone()]

    def frames(self, prefix, values):
        """values [B, P] -> prefix/loss (mean) and prefix_frame/0..3."""
        self.add(f'{prefix}/loss', values.sum(), values.numel())
        for j in range(values.shape[1]):
            self.add(f'{prefix}_frame/{j}', values[:, j].sum(), values.shape[0])

    def buckets(self, prefix, t, values, edges):
        for lo, hi in zip(edges[:-1], edges[1:]):
            inside = (t > lo) & (t <= hi) if lo > edges[0] else (t >= lo) & (t <= hi)
            self.add(f'{prefix}/{lo:.2f}-{hi:.2f}', torch.where(inside, values, 0).sum(), inside.sum())

    def report(self):
        if not self.values:
            return {}
        names = list(self.values)
        packed = torch.stack([torch.stack(self.values[n]) for n in names])
        dist.all_reduce(packed)
        out = {}
        for name, (total, count) in zip(names, packed.cpu().tolist()):
            if count > 0:
                out[name] = total / count
        self.values = {}
        return out


def load_config(path, overrides):
    with open(path, 'rb') as handle:
        config = tomllib.load(handle)
    for item in overrides:
        key, value = item.split('=', 1)
        node = config
        *parents, leaf = key.split('.')
        for parent in parents:
            node = node.setdefault(parent, {})
        try:
            node[leaf] = tomllib.loads(f'v = {value}')['v']
        except tomllib.TOMLDecodeError:  # bare words, e.g. student_kinds=stab,res (quotes do not survive ssh)
            node[leaf] = value.split(',') if ',' in value or isinstance(node.get(leaf), list) else value
    return config


def memory(tag):
    if os.environ.get('DEBUG_MEMORY') and dist.get_rank() == 0:
        print(f'[memory] {tag}: allocated {torch.cuda.memory_allocated() / 2**30:.1f} GiB, '
              f'peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB', flush=True)


def parse_value(value):
    try:
        return tomllib.loads(f'v = {value}')['v']
    except tomllib.TOMLDecodeError:
        return value


def lr_lambda(warmup):
    return lambda step: min(1.0, (step + 1) / warmup) if warmup else 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--set', action='append', default=[], help='key=toml_value overrides')
    parser.add_argument('--benchmark', type=int, default=0, help='outer steps to time (no warmup, saves or W&B)')
    parser.add_argument('--benchmark-skip', type=int, default=2)
    parser.add_argument('--variants', default='', help='benchmark: "|"-separated variants, each "key=value,key=value" '
                        'applied for --variant-steps steps in turn (one launch, warm graphs)')
    parser.add_argument('--variant-steps', type=int, default=3)
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    if config.get('resume_from') and config.get('critic_resume_from'):
        raise ValueError('resume_from and critic_resume_from are mutually exclusive')
    if args.benchmark and config.get('resume_from'):
        raise ValueError('benchmark does not resume a student training run')

    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    dist.init_process_group('nccl', device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    main_process = rank == 0
    node_sharded = config.get('node_sharded', False)
    if node_sharded or config.get('calibrate_updates', False):
        saver.init_gloo()
    torch.manual_seed(config['seed'] + rank)
    random.seed(config['seed'] + rank)
    np.random.seed(config['seed'] + rank)
    torch._dynamo.config.cache_size_limit = 1024
    torch._dynamo.config.accumulated_cache_size_limit = 4096
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    audio_config = config['model']['audio']
    rows_per_rank = config['critic_updates'] * config['critic_rows'] + len(config['student_kinds'])
    source = AloeChunkSource(config['root_dir'], config['clipnames_root'], config['train_csv'],
                             pose_file=config['pose_latents'], mimi_file=config['mimi_codes'],
                             bad_audio=config.get('bad_audio', 0))
    data_rank, data_world = (local_rank, int(os.environ['LOCAL_WORLD_SIZE'])) if node_sharded else (rank, world)
    dataset = SeededWindows(source, data_rank, data_world, rows_per_rank, saver.gloo if node_sharded else None)
    first = dataset[0]
    latent_hw = tuple(first['latents_low'].shape[-2:])
    csv_path = Path(config['clipnames_root']) / config['train_csv']
    csv_path = csv_path if csv_path.suffix == '.csv' else csv_path.with_suffix('.csv')
    data_sources = [None] * world
    dist.all_gather_object(data_sources, file_sha256(csv_path))
    local_rows = []
    if (local_rank == 0 if node_sharded else main_process):
        local_rows = [(os.path.relpath(clip.video_path, config['root_dir']), clip.length,
                       bool(source.bad_audio[i]) if source.bad_audio is not None else False)
                      for i, clip in enumerate(source.clips)]
    all_rows = [None] * world
    dist.all_gather_object(all_rows, local_rows)
    data_hash = distribution_fingerprint(all_rows)
    checkpoint_hash = [file_sha256(config['model']['checkpoint_path']) if main_process else None]
    dist.broadcast_object_list(checkpoint_hash, src=0)
    signature = compatibility(config, checkpoint_hash[0], latent_hw, data_hash)
    resume_path = config.get('resume_from') or config.get('critic_resume_from')
    critic_only = bool(config.get('critic_resume_from'))
    resume_meta = read_metadata(resume_path, critic_only=critic_only,
                                optimizer_policy=config.get('critic_resume_optimizer', 'restore')) if resume_path else None
    if resume_meta is not None:
        validate_resume(resume_meta, config, signature, world, data_sources, len(dataset), critic_only)

    # Models: one CPU state dict, three replicated nets.
    start = time.time()
    state_dict = load_state_dict(config['model']['checkpoint_path'])
    mimi, template = load_mimi(audio_config, device)
    nets = {role: build_net(state_dict, audio_config, latent_hw, device, template)
            for role in ('student', 'teacher', 'critic')}
    del state_dict
    reset_critic = critic_only and config.get('critic_resume_optimizer', 'restore') == 'reset'
    if reset_critic:
        load_model_weights(Path(resume_path), 'critic', nets['critic'], node_sharded)
    node_group, _ = dist.new_subgroups(group_size=int(os.environ['LOCAL_WORLD_SIZE']))
    if config.get('shard_teacher', True):
        nets['teacher'].shard_blocks(node_group)
        torch.cuda.empty_cache()
    for role, net in nets.items():
        trainable = role != 'teacher'
        net.requires_grad_(trainable)
        if trainable and config.get('lora'):
            with torch.random.fork_rng(devices=[local_rank]):
                torch.manual_seed(config['seed'])
                inject_lora(net, **config['lora'])
        if resume_path and not critic_only and config.get('lora'):
            if trainable:
                load_model_weights(Path(resume_path), role, net, node_sharded)
        for name, p in net.original_named_parameters():
            p.original_name = name
        settings = config.get('checkpoint_by_role', {}).get(role, {})
        net.configure(config['compile_blocks'], settings.get('mode', config['checkpoint']), config.get('compile_mode', 'parts'),
                      checkpoint_skip_blocks=settings.get('skip_blocks', ()))
    if main_process:
        print(f'models loaded in {time.time() - start:.0f}s; '
              f'{sum(p.numel() for p in nets["student"].parameters()):,} parameters per net; '
              f'{sum(p.numel() for p in nets["student"].parameters() if p.requires_grad):,} trainable', flush=True)

    optim = config['optimizer']
    make_inner = lambda groups: torch.optim.AdamW(groups, lr=optim['lr'], betas=tuple(optim['betas']),
                                                  weight_decay=optim['weight_decay'], eps=optim['eps'], fused=True)
    optimizers = {role: Zero1FP32MasterAdamW([{'params': [p for p in nets[role].parameters() if p.requires_grad]}], make_inner)
                  for role in ('student', 'critic')}
    grads = {role: GradFlats(optimizer) for role, optimizer in optimizers.items()}
    schedulers = {
        'student': torch.optim.lr_scheduler.LambdaLR(optimizers['student'].inner, lr_lambda(config['lr_warmup_steps'])),
        'critic': torch.optim.lr_scheduler.LambdaLR(optimizers['critic'].inner, lr_lambda(config['critic_lr_warmup_steps'])),
    }
    ema = EMA(math.exp(-1 / config['ema_decay_steps']), optimizers['student'].pairs) if config['ema_decay_steps'] > 0 else None
    counters = dict(outer_step=0, warmup_step=0, critic_updates=0, student_updates=0, data_steps=0)
    if resume_path:
        restored = resume_meta['counters'] if reset_critic else load_rank_state(
            Path(resume_path), optimizers, schedulers, ema, node_sharded, critic_only)
        if critic_only:
            counters['warmup_step'] = config['critic_warmup_steps']
            counters['critic_updates'] = restored['critic_updates']
        else:
            counters.update(restored)
    memory('models + optimizers')
    make_rollout = lambda kind: {'uniform': UniformRollout, 'uniform_stab': UniformStabilizerRollout, 'fused': Rollout}[kind](
        config['band_steps'], config['resolve_steps'], config['history_t'], config['t_start'])
    rollout = make_rollout(config.get('rollout', 'fused'))
    # The no-grad batch-16 pool is compute-bound, so padded frames cost real time there: trimmed per-pass layouts.
    pool_rollout = make_rollout(config.get('pool_rollout', config.get('rollout', 'fused')))
    t_bounds = dict(stab=(config['stab_t_min'], config['stab_t_max']), res=(config['res_t_min'], config['res_t_max']))
    fm_weights = dict(stab=config['fm_weight_stab'], res=config['fm_weight_res'])
    epsilon = config['normalizer_epsilon']
    h = config['history_t']

    benchmark = args.benchmark > 0
    warmup_steps = 0 if benchmark else max(0, config['critic_warmup_steps'] - counters['warmup_step'])
    student_steps = args.benchmark if benchmark else max(0, config['max_outer_steps'] - counters['outer_step'])
    total_outer = warmup_steps + (0 if config.get('critic_warmup_only', False) else student_steps)
    loader = DataLoader(dataset, batch_size=None, sampler=range(counters['data_steps'], counters['data_steps'] + total_outer),
                        num_workers=config['num_dataloader_workers'], pin_memory=True,
                        prefetch_factor=config.get('prefetch_factor', 4), persistent_workers=True,
                        generator=torch.Generator().manual_seed(config['seed']))

    run_dir = tracker = None
    if not benchmark:
        paths = [None]
        if main_process:
            run_dir = Path(config['output_dir']) / datetime.now(timezone.utc).strftime('run-%Y%m%d-%H%M%S')
            run_dir.mkdir(parents=True)
            (run_dir / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
            paths = [str(run_dir)]
        dist.broadcast_object_list(paths, src=0)
        run_dir = Path(paths[0])
        if main_process and config['wandb']['enabled']:
            import wandb
            tracker = wandb.init(project=config['wandb']['project'], name=config['wandb']['run_name'], config=config,
                                 dir=str(run_dir))

    def prepare(raw, lo, hi, codebooks):
        rows = {
            'latents': raw['latents_low'][lo:hi].to(device, non_blocking=True).float(),
            'pose': raw['pose_latents_low'][lo:hi].to(device, non_blocking=True).float(),
            'ofi': raw['original_frame_indices'][lo:hi].to(device, non_blocking=True),
            'history_t': h,
        }
        bad = raw.get('bad_audio')
        rows['drop'] = (bad[lo:hi].to(device) if bad is not None
                        else torch.zeros(hi - lo, dtype=torch.bool, device=device))
        rows['audio'] = audio_latents(mimi, raw['codes'][lo:hi].to(device, non_blocking=True), codebooks)
        return rows

    def codebook_count():
        return int(torch.randint(audio_config['min_codebooks'], audio_config['num_codebooks'] + 1, ()))

    def generation_noise(rows):
        return torch.randn_like(rows['latents']), torch.randn_like(rows['latents'])

    def stab_history(rows):
        gt = rows['latents'][:, :, 1:1 + H]
        return (1 - h) * gt + h * torch.randn_like(gt)

    def resolve_clean(rows, y0):
        return torch.cat((rows['latents'][:, :, 2:1 + H], y0[:, :, :P - 1].detach()), dim=2)

    sums = Sums()
    timers = {}
    update_metrics = {}
    probe = {}

    def prepare_probe(raw):
        if not probe:
            probe['rows'] = prepare(raw, 0, 1, audio_config['num_codebooks'])
            rng = torch.Generator(device=device).manual_seed(config['seed'] + 100003)
            shape = probe['rows']['latents'].shape
            probe['noise'] = torch.randn(shape, device=device, generator=rng)
            probe['renoise'] = torch.randn(shape, device=device, generator=rng)
            probe['rollout'] = UniformStabilizerRollout(config['band_steps'], config['resolve_steps'], h, config['t_start'])

    @torch.no_grad()
    def probe_student(raw):
        if not main_process or not config.get('calibrate_updates', False):
            return None
        prepare_probe(raw)
        net = nets['student']
        compiled = net.block_fn
        net.block_fn = net._block
        try:
            y, _ = probe['rollout'](net, probe['rows'], net.audio_tokens(probe['rows']['audio']),
                                   probe['noise'], probe['renoise'], stab_only=True)
        finally:
            net.block_fn = compiled
        return y.float().cpu()

    @torch.no_grad()
    def probe_critic(raw):
        if not main_process or not config.get('calibrate_updates', False):
            return None
        prepare_probe(raw)
        rows, net = probe['rows'], nets['critic']
        t = torch.tensor([[.3, .5, .7, .9]], device=device)
        x, _ = noised(rows['latents'][:, :, 4:], 0., t, probe['noise'][:, :, 4:])
        history = (1 - h) * rows['latents'][:, :, 1:4] + h * probe['renoise'][:, :, 1:4]
        compiled = net.block_fn
        net.block_fn = net._block
        try:
            audio = net.frame_audio(net.audio_tokens(rows['audio']), rows['ofi'][:, 4:], rows['drop'])
            velocity = stab_pass(net, rows, x, t, history, audio, torch.ones(1, P, dtype=torch.bool, device=device))
        finally:
            net.block_fn = compiled
        return velocity.float().cpu()

    def optimizer_step(role, calibrate=True):
        if not config.get('calibrate_updates', False) or not calibrate:
            optimizers[role].step()
            return
        from distillation.update_scale import capture, measure
        targets = config['lora']['targets'] if config.get('lora') else config['calibration_targets']
        before = capture(nets[role], optimizers[role], targets, bool(config.get('lora')))
        optimizers[role].step()
        after = capture(nets[role], optimizers[role], targets, bool(config.get('lora')))
        measured = measure(before, after)
        if main_process:
            update_metrics.update({f'calibration/{role}/{key}': value for key, value in measured.items()})

    def timed(name):
        class Timer:
            def __enter__(self):
                self.start = torch.cuda.Event(enable_timing=True)
                self.end = torch.cuda.Event(enable_timing=True)
                self.start.record()

            def __exit__(self, *exc):
                self.end.record()
                timers.setdefault(name, []).append((self.start, self.end))
        return Timer()

    def critic_phase(raw):
        critic_net, student_net = nets['critic'], nets['student']
        n = config['critic_rows']
        groups = [prepare(raw, i * n, (i + 1) * n, codebook_count()) for i in range(config['critic_updates'])]
        with timed('pool'), torch.no_grad():
            pool = {key: torch.cat([g[key] for g in groups]) if torch.is_tensor(groups[0][key]) else groups[0][key]
                    for key in groups[0]}
            tokens = student_net.audio_tokens(pool['audio'])
            noise, renoise = generation_noise(pool)
            outputs = []
            size = config['pool_micro_batch']
            for i in range(0, len(pool['latents']), size):
                part = {k: v[i:i + size] if torch.is_tensor(v) else v for k, v in pool.items()}
                outputs.append(pool_rollout(student_net, part, tokens[i:i + size], noise[i:i + size], renoise[i:i + size],
                                            stab_only=fm_weights['res'] == 0))
            y02 = torch.cat([o[0] for o in outputs])
            y0 = torch.cat([o[1] for o in outputs]) if fm_weights['res'] else None
        memory('after pool')
        with timed('critic'):
            critic_probe_before = probe_critic(raw)
            for u, rows in enumerate(groups):
                lo, hi = u * n, (u + 1) * n
                grads['critic'].allocate()
                B = hi - lo
                if config.get('critic_combined', False) and fm_weights['res']:
                    t_s = stab_times(B, device, *t_bounds['stab'])
                    x_s, v_s = noised(y02[lo:hi], h, t_s, torch.randn_like(y02[lo:hi]))
                    t_r = resolve_times(B, device, *t_bounds['res'])
                    x_r, v_r = noised(y0[lo:hi], 0.0, t_r, torch.randn_like(y0[lo:hi]))
                    tokens = critic_net.audio_tokens(rows['audio'])
                    audio = critic_net.frame_audio(tokens, rows['ofi'][:, 4:], rows['drop'])
                    mask = torch.ones(B, 2 * P, dtype=torch.bool, device=device)
                    vel_s, vel_r = both_pass(critic_net, rows, x_s, t_s, stab_history(rows), x_r, t_r,
                                             resolve_clean(rows, y0[lo:hi]), torch.cat((audio, audio), dim=1), mask)
                    loss_stab, frames_stab = fm_loss(vel_s, v_s)
                    loss_res, frames_res = fm_loss(vel_r, v_r)
                    (fm_weights['stab'] * loss_stab + fm_weights['res'] * loss_res).backward()
                    sums.frames('fm_stabilizer', frames_stab)
                    sums.buckets('fm_stabilizer_tbucket', t_s, frames_stab, STAB_BUCKETS)
                    sums.frames('fm_resolver', frames_res)
                    sums.buckets('fm_resolver_tbucket', t_r, frames_res, RES_BUCKETS)
                else:
                    loss_stab, loss_res = separate_critic_losses(critic_net, rows, y02[lo:hi], y0[lo:hi] if y0 is not None else None, B)
                sums.add('critic/loss', fm_weights['stab'] * loss_stab.detach() + fm_weights['res'] * loss_res.detach(), 1)
                norm = grads['critic'].reduce_clip(config['gradient_clipping'])
                optimizer_step('critic', calibrate=u == 0)
                schedulers['critic'].step()
                grads['critic'].release()
                sums.add('critic/grad_norm', norm, 1)
                counters['critic_updates'] += 1
            if critic_probe_before is not None:
                critic_probe_after = probe_critic(raw)
                probe.setdefault('critic_initial', critic_probe_before)
                update_metrics['calibration/critic_probe/update_rms'] = (critic_probe_after - critic_probe_before).square().mean().sqrt().item()
                update_metrics['calibration/critic_probe/cumulative_rms'] = (critic_probe_after - probe['critic_initial']).square().mean().sqrt().item()
                update_metrics['calibration/critic_probe/output_rms'] = critic_probe_before.square().mean().sqrt().item()

    def separate_critic_losses(critic_net, rows, y02, y0, B):
        """Stabilising and resolving FM as two passes, each backpropagated on its own."""
        # stabilising window: generated frames at history_t, re-noised into the bands
        t = stab_times(B, device, *t_bounds['stab'])
        x, v_star = noised(y02, h, t, torch.randn_like(y02))
        tokens = critic_net.audio_tokens(rows['audio'])
        audio = critic_net.frame_audio(tokens, rows['ofi'][:, 4:], rows['drop'])
        mask = torch.ones(B, P, dtype=torch.bool, device=device)
        velocity = stab_pass(critic_net, rows, x, t, stab_history(rows), audio, mask)
        loss_stab, frames_stab = fm_loss(velocity, v_star)
        (fm_weights['stab'] * loss_stab).backward()
        sums.frames('fm_stabilizer', frames_stab)
        sums.buckets('fm_stabilizer_tbucket', t, frames_stab, STAB_BUCKETS)
        if fm_weights['res'] == 0:
            return loss_stab, loss_stab.new_zeros(())
        # resolving windows: resolved frames, each at its own t
        t = resolve_times(B, device, *t_bounds['res'])
        x, v_star = noised(y0, 0.0, t, torch.randn_like(y0))
        tokens = critic_net.audio_tokens(rows['audio'])
        audio = critic_net.frame_audio(tokens, rows['ofi'][:, 4:], rows['drop'])
        velocity = resolve_pass(critic_net, rows, x, t, resolve_clean(rows, y0), audio, mask)
        loss_res, frames_res = fm_loss(velocity, v_star)
        (fm_weights['res'] * loss_res).backward()
        sums.frames('fm_resolver', frames_res)
        sums.buckets('fm_resolver_tbucket', t, frames_res, RES_BUCKETS)
        return loss_stab, loss_res

    def student_phase(raw):
        student_net, teacher_net, critic_net = nets['student'], nets['teacher'], nets['critic']
        offset = config['critic_updates'] * config['critic_rows']
        kinds = config['student_kinds']
        m = config.get('student_micro_rows', 1)  # same-kind rows batched into one microbatch: identical objective
        micro = [(kinds[i], i) for i in range(0, len(kinds), m)]
        assert all(len(set(kinds[i:i + m])) == 1 for _, i in micro), 'a microbatch must be a single kind'
        with timed('student'):
            grads['student'].allocate()
            for kind, i in micro:
                rows = prepare(raw, offset + i, offset + i + m, codebook_count())
                B = m
                noise, renoise = generation_noise(rows)
                tokens = student_net.audio_tokens(rows['audio'])
                memory(f'student {kind} before rollout')
                with timed('student_rollout'):
                    y02, y0 = rollout(student_net, rows, tokens, noise, renoise, stab_only=kind == 'stab')
                memory(f'student {kind} after rollout')
                mask = torch.ones(B, P, dtype=torch.bool, device=device)
                with torch.no_grad(), timed('student_scoring'):
                    if kind == 'stab':
                        y, t0, (t_min, t_max) = y02, h, t_bounds['stab']
                        t = stab_times(B, device, t_min, t_max)
                    else:
                        y, t0, (t_min, t_max) = y0, 0.0, t_bounds['res']
                        t = resolve_times(B, device, t_min, t_max)
                    x, v_star = noised(y, t0, t, torch.randn_like(y))
                    velocities = []
                    history = stab_history(rows) if kind == 'stab' else resolve_clean(rows, y0)
                    for net in (teacher_net, critic_net):
                        audio = net.frame_audio(net.audio_tokens(rows['audio']), rows['ofi'][:, 4:], rows['drop'])
                        layout = stab_pass if kind == 'stab' else resolve_pass
                        velocities.append(layout(net, rows, x, t, history, audio, mask))
                loss, frames, normalizer, difference = dmd_loss(y, velocities[0], velocities[1], v_star, epsilon)
                with timed('student_backward'):
                    (loss * B / len(kinds)).backward()
                memory(f'student {kind} after backward')
                name = 'dmd_stabilizer' if kind == 'stab' else 'dmd_resolver'
                edges = STAB_BUCKETS if kind == 'stab' else RES_BUCKETS
                sums.frames(name, frames)
                sums.add(f'{name}/normalizer', normalizer.sum(), normalizer.numel())
                sums.add(f'{name}/difference', difference.sum(), difference.numel())
                sums.buckets(f'{name}_tbucket_loss', t, frames, edges)
                sums.buckets(f'{name}_tbucket_normalizer', t, normalizer, edges)
                sums.add('student/loss', loss.detach() * B, B)
            with timed('student_sync_step'):
                norm = grads['student'].reduce_clip(config['gradient_clipping'])
                probe_before = probe_student(raw)
                optimizer_step('student')
                if probe_before is not None:
                    probe_after = probe_student(raw)
                    probe.setdefault('initial', probe_before)
                    update_metrics['calibration/probe/update_rms'] = (probe_after - probe_before).square().mean().sqrt().item()
                    update_metrics['calibration/probe/cumulative_rms'] = (probe_after - probe['initial']).square().mean().sqrt().item()
                    update_metrics['calibration/probe/output_rms'] = probe_before.square().mean().sqrt().item()
                schedulers['student'].step()
                grads['student'].release()
                if ema is not None:
                    ema.update()
            sums.add('student/grad_norm', norm, 1)
            counters['student_updates'] += 1

    def report():
        torch.cuda.synchronize()
        out = dict(counters)
        out.update(sums.report())
        for name, events in timers.items():
            seconds = torch.tensor(sum(s.elapsed_time(e) for s, e in events) / 1000 / len(events), device=device)
            every = [torch.zeros_like(seconds) for _ in range(world)]
            dist.all_gather(every, seconds)
            every = torch.stack(every)
            out[f'seconds/{name}'] = every.max().item()
            out[f'seconds_spread/{name}_median'] = every.median().item()
            out[f'seconds_spread/{name}_slowest_rank'] = int(every.argmax())
        timers.clear()
        peak = torch.tensor(torch.cuda.max_memory_allocated() / 2**30, device=device)
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        out['memory/peak_allocated_gib'] = peak.item()
        torch.cuda.reset_peak_memory_stats()
        out['student/lr'] = optimizers['student'].inner.param_groups[0]['lr']
        out['critic/lr'] = optimizers['critic'].inner.param_groups[0]['lr']
        out.update(update_metrics)
        return out

    def save(name):
        directory = run_dir / name
        owned_bytes = sum(sum(m.numel() * m.element_size() for _, m in opt.pairs) + tensor_bytes(opt.inner.state)
                          for opt in optimizers.values())
        owned_bytes += sum(s.numel() * s.element_size() for _, _, s in ema.triples) if ema is not None else 0
        required = torch.tensor(owned_bytes + 1024**2, device=device, dtype=torch.int64)
        dist.all_reduce(required)
        disk_error = [None]
        if main_process:
            weight_bytes = sum(p.numel() * p.element_size() for role in ('student', 'critic')
                               for _, p in nets[role].original_named_parameters())
            if ema is not None:
                weight_bytes += sum(p.numel() * 2 for _, p in nets['student'].original_named_parameters())
            needed = int((int(required) + weight_bytes) * 1.1) + 1024**3
            free = shutil.disk_usage(run_dir).free
            if free < needed:
                disk_error[0] = f'Checkpoint needs approximately {needed / 2**30:.1f} GiB free; {free / 2**30:.1f} GiB available at {run_dir}'
        dist.broadcast_object_list(disk_error, src=0)
        if disk_error[0]:
            raise RuntimeError(disk_error[0])
        if main_process:
            directory.mkdir(parents=True, exist_ok=True)
            (directory / 'checkpoint.complete.json').unlink(missing_ok=True)
            (directory / 'metadata.json').unlink(missing_ok=True)
        dist.barrier()
        if main_process:
            for role in ('student', 'critic'):
                tensors = {n: p.detach().cpu().contiguous() for n, p in nets[role].original_named_parameters()}
                save_file(tensors, str(directory / f'{role}.safetensors'), metadata={'format': 'pt'})
        save_ema(directory, ema, node_sharded)
        save_rank_states(directory, optimizers, schedulers, ema, counters, node_sharded)
        if main_process:
            meta = {'world_size': world, 'compatibility': signature, 'counters': dict(counters),
                    'lora': config.get('lora'),
                    'critic_bootstrap': {'path': resume_path, 'checkpoint_id': resume_meta['checkpoint_id'],
                                         'optimizer_pruned_from_checkpoint_id': resume_meta.get('optimizer_pruned_from_checkpoint_id'),
                                         'optimizer': config.get('critic_resume_optimizer', 'restore'),
                                         'allow_data_change': config.get('critic_resume_allow_data_change', False)}
                                        if critic_only else (resume_meta or {}).get('critic_bootstrap'),
                    'training': {key: config[key] for key in RESUME_KEYS}, 'data_sources': data_sources,
                    'dataset_length': len(dataset), 'optimizer_state_retained': True,
                    'critic_warmup_complete': counters['warmup_step'] >= config['critic_warmup_steps']
                                              and counters['critic_updates'] > 0}
            complete_checkpoint(directory, meta)
            os.chmod(directory, 0o775)
            for path in directory.iterdir():
                os.chmod(path, 0o664)
            # Optimizer / EMA shards only in the newest checkpoint (~5 GB per rank); weights stay in every step dir.
            for older in [*run_dir.glob('step-*'), run_dir / 'critic-warmup']:
                if older.name < directory.name:
                    shards = list(older.glob('rank-*.pt'))
                    if shards:
                        old_meta = json.loads((older / 'metadata.json').read_text())
                        prune_optimizer_state(older.absolute(), old_meta['checkpoint_id'], verify_weight_hashes=False)
            print(f'saved {directory}', flush=True)

    step_times = []
    # As learn_aloe / learn_bimodal: compiled graphs and models are long-lived; left unfrozen, every gen-2 collection
    # rescans them on one rank and stalls the others at the next collective. Refreeze while new graphs appear
    # (the first steps of each phase).
    torch.cuda.empty_cache()
    gc.freeze()
    # A collection pausing one rank mid-backward stalls all 40 at the gradient reduce-scatter: no automatic
    # collections; every rank collects together at a step boundary instead.
    gc.disable()
    variants = [dict(item.split('=', 1) for item in v.split(',') if item) for v in args.variants.split('|')] if args.variants else []
    variant_times = {}
    for index, raw in enumerate(loader):
        if variants:
            variant = variants[min(index // args.variant_steps, len(variants) - 1)]
            for key, value in variant.items():
                config[key] = parse_value(value)
                if key == 'checkpoint':
                    for net in nets.values():
                        net.checkpoint = config[key]
            name = ','.join(f'{k}={v}' for k, v in variant.items()) or 'base'
        wall = time.time()
        if config.get('compile_mode') == 'block_cudagraph':
            torch.compiler.cudagraph_mark_step_begin()
        warmup = index < warmup_steps
        profiling = main_process and not warmup and counters['outer_step'] + 1 == config.get('profile_step')
        profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                          record_shapes=False) if profiling else nullcontext()
        with profiler, timed('outer'):
            critic_phase(raw)
            if not warmup:
                student_phase(raw)
        if warmup:
            counters['warmup_step'] += 1
        else:
            counters['outer_step'] += 1
        counters['data_steps'] += 1
        torch.cuda.synchronize()
        step_times.append(time.time() - wall)
        if counters['warmup_step'] <= 4 and counters['outer_step'] == 0 or 0 < counters['outer_step'] <= 4:
            gc.freeze()
        elif (counters['warmup_step'] + counters['outer_step']) % 50 == 0:
            gc.collect()
        if variants:
            variant_times.setdefault(name, []).append(step_times[-1])
        out = report()
        out['seconds/wall'] = step_times[-1]
        out['step'] = counters['warmup_step'] + counters['outer_step']
        if profiling:
            out['profiled'] = True
            profile_dir = Path(config['profile_dir'])
            profile_dir.mkdir(parents=True, exist_ok=True)
            profiler.export_chrome_trace(str(profile_dir / 'trace.json'))
            (profile_dir / 'operators.txt').write_text(profiler.key_averages().table(sort_by='self_device_time_total', row_limit=60))
            print(f'profile saved to {profile_dir}', flush=True)
        if main_process:
            print(json.dumps({k: round(v, 6) if isinstance(v, float) and not k.startswith('calibration/')
                              and not k.endswith('/lr') else v for k, v in out.items()}), flush=True)
            if tracker is not None:
                tracker.log(out, step=out['step'])
        if not benchmark and warmup and counters['warmup_step'] == config['critic_warmup_steps'] \
                and config.get('save_warmed_critic', False):
            save('critic-warmup')
        if not benchmark and not warmup and (counters['outer_step'] % config['save_every_n_steps'] == 0
                                                or counters['outer_step'] == config['max_outer_steps']):
            save(f'step-{counters["warmup_step"] + counters["outer_step"]:06d}')
        if not benchmark and run_dir is not None:
            flags = torch.zeros(1, device=device)
            if main_process and (run_dir / 'save_q').exists():
                (run_dir / 'save_q').unlink()
                flags[0] = 1
            dist.broadcast(flags, src=0)
            if flags.item():
                save(f'step-{counters["warmup_step"] + counters["outer_step"]:06d}')
                break
    if variants and main_process:
        for name, times in variant_times.items():
            print(json.dumps({'variant': name, 'steps': times, 'best': min(times),
                              'median_after_first': float(np.median(times[1:])) if len(times) > 1 else times[0]}), flush=True)
    if benchmark and main_process:
        measured = step_times[args.benchmark_skip:]
        print(json.dumps({'benchmark': True, 'steps': measured, 'median': float(np.median(measured)),
                          'mean': float(np.mean(measured)), 'min': float(np.min(measured))}), flush=True)
    if tracker is not None:
        tracker.finish()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
