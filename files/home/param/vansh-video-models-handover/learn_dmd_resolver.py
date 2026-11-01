"""One-step resolver DMD, with GT-noised or frozen-stabilizer inputs.

All trainable forwards and scoring are resolver-only. Global target batch is
fixed independently of microbatching; optimizer masters/moments use ZeRO-1.
"""
import argparse
import gc
import json
import os
import random
import shutil
import time
import tomllib
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch.utils.data import DataLoader

from dataloading.ds_aloe import AloeChunkSource
from distillation.bimodal_dmd import dmd_loss, fm_loss, noised, resolve_times
from distillation.bimodal_net import audio_latents, build_net, load_mimi, load_state_dict
from distillation.checkpoints import (complete_checkpoint, distribution_fingerprint, file_sha256,
    load_model_weights, load_rank_state, prune_optimizer_state, read_metadata, save_rank_states, tensor_bytes)
from distillation.resolver_data import ResolverWindows, learning_rate_factor, rank_row_units
from distillation.resolver_dmd import GTConditionedStabilizer, gt_noised_input, one_step_resolve, resolve_gt_pass, target_start
from learn_dmd_bimodal import GradFlats, Sums, load_config
from utils import saver
from utils.harness import Zero1FP32MasterAdamW


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--set', action='append', default=[])
    parser.add_argument('--benchmark', type=int, default=0)
    parser.add_argument('--benchmark-skip', type=int, default=2)
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    method = config['resolver_method']
    if method not in ('gt', 'onpolicy'):
        raise ValueError(method)
    config.setdefault('total_steps', 2000)
    config.setdefault('global_batch_size', 160)
    config.setdefault('critic_warmup_steps', 150)
    config.setdefault('lr_final_ratio', .1)
    config.setdefault('lr_warmup_steps', 50)
    config.setdefault('critic_lr_warmup_steps', 100)
    if config['global_batch_size'] != 160:
        raise ValueError('This experiment fixes every optimizer update at global batch 160')
    if config.get('dataset_history') != 6 or config.get('t_start') != .2 or config.get('history_t') != .2:
        raise ValueError('This experiment uses six GT context frames and t_start=history_t=.2')
    if config.get('resolve_steps') != 1 or config.get('student_rows') != 1 or config.get('critic_rows') != 1:
        raise ValueError('One-step resolver uses one four-target row per rank and optimizer update')
    if config.get('lr_decay') != 'cosine' or config.get('lora'):
        raise ValueError('This experiment uses dense training with cosine LR decay')
    with (Path(__file__).parent / config['history_dropout_source']).open('rb') as handle:
        fine_tuning_dropout = tomllib.load(handle)['stabilising']['history_dropout']
    if config['history_dropout'] != fine_tuning_dropout:
        raise ValueError('History dropout must match the pose-v3 fine-tuning parent-window probabilities')
    benchmark = args.benchmark > 0
    local_rank, local_world = int(os.environ['LOCAL_RANK']), int(os.environ['LOCAL_WORLD_SIZE'])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    dist.init_process_group('nccl', device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    nodes = world // local_world
    if nodes != 5 or local_world != 8:
        raise ValueError('This launch requires five nodes with eight healthy GPUs each')
    saver.init_gloo()
    torch.manual_seed(config['seed'] + rank)
    np.random.seed(config['seed'] + rank)
    random.seed(config['seed'] + rank)
    torch._dynamo.config.cache_size_limit = 1024
    torch._dynamo.config.accumulated_cache_size_limit = 4096
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    main_process = rank == 0
    local_rows, _, _ = rank_row_units(config['global_batch_size'], nodes, local_world, local_rank)
    updates = config['critic_updates']
    loss_rank_weight = world * local_rows * 4 / config['global_batch_size']
    source = AloeChunkSource(config['root_dir'], config['clipnames_root'], config['train_csv'],
        pose_file=config['pose_latents'], mimi_file=config['mimi_codes'], bad_audio=config.get('bad_audio', 0))
    dataset = ResolverWindows(source, local_rank, local_world, config['global_batch_size'], nodes,
                              updates + 1, saver.gloo, history_dropout=config['history_dropout'])
    latent_hw = tuple(dataset[0]['latents_low'].shape[-2:])
    data_sources = [None] * world
    dist.all_gather_object(data_sources, file_sha256(config['train_csv']))
    rows_metadata = [(os.path.relpath(clip.video_path, config['root_dir']), clip.length,
        bool(source.bad_audio[i]) if source.bad_audio is not None else False)
        for i, clip in enumerate(source.clips)] if local_rank == 0 else []
    all_rows = [None] * world
    dist.all_gather_object(all_rows, rows_metadata)
    data_hash = distribution_fingerprint(all_rows)
    hashes = [None]
    if main_process:
        hashes[0] = {'teacher': file_sha256(config['model']['checkpoint_path'])}
        if method == 'onpolicy':
            hashes[0]['stabilizer'] = file_sha256(config['stabilizer_checkpoint_path'])
    dist.broadcast_object_list(hashes, src=0)
    signature = {'method': method, 'checkpoint_sha256': hashes[0], 'data_distribution_sha256': data_hash,
                 'data_sources': data_sources, 'latent_hw': latent_hw, 'global_batch_size': config['global_batch_size'],
                 'history': 'ground_truth', 'target_start': 7, 'resolver_t_start': .2, 'resolver_steps': 1,
                 'band_steps': config['band_steps'], 'audio': config['model']['audio'],
                 'training': {key: config[key] for key in ('optimizer', 'total_steps', 'critic_warmup_steps',
                     'critic_updates', 'lr_warmup_steps', 'critic_lr_warmup_steps', 'lr_final_ratio',
                     'res_t_min', 'res_t_max', 'normalizer_epsilon', 'gradient_clipping', 'seed',
                     'pose_latents', 'mimi_codes', 'bad_audio', 'history_dropout', 'history_dropout_source')}}
    start = time.time()
    node_group, _ = dist.new_subgroups(group_size=local_world)
    audio_config = config['model']['audio']
    mimi, template = load_mimi(audio_config, device)
    state = load_state_dict(config['model']['checkpoint_path'])
    nets = {}
    # Shard each frozen model immediately; never hold all four full replicas.
    nets['teacher'] = build_net(state, audio_config, latent_hw, device, template).requires_grad_(False)
    if config.get('shard_teacher', True):
        nets['teacher'].shard_blocks(node_group)
        torch.cuda.empty_cache()
    for role in ('student', 'critic'):
        nets[role] = build_net(state, audio_config, latent_hw, device, template)
    del state
    if method == 'onpolicy':
        state = load_state_dict(config['stabilizer_checkpoint_path'])
        nets['stabilizer'] = build_net(state, audio_config, latent_hw, device, template).requires_grad_(False)
        del state
        if config.get('shard_stabilizer', True):
            nets['stabilizer'].shard_blocks(node_group)
            torch.cuda.empty_cache()
    resume = config.get('resume_from')
    meta = read_metadata(resume) if resume else None
    if meta:
        if meta['world_size'] != world or meta['compatibility'] != json.loads(json.dumps(signature)):
            raise ValueError('Resume world size, data, conditioning, or model identity differs')
        for role in ('student', 'critic'):
            load_model_weights(Path(resume), role, nets[role], True)
    for role, net in nets.items():
        net.requires_grad_(role in ('student', 'critic'))
        for name, parameter in net.original_named_parameters():
            parameter.original_name = name
        settings = config.get('checkpoint_by_role', {}).get(role, {})
        net.configure(config['compile_blocks'], settings.get('mode', config['checkpoint']),
                      config.get('compile_mode', 'block'), checkpoint_skip_blocks=settings.get('skip_blocks', ()))
    optim = config['optimizer']
    def make_inner(groups):
        return torch.optim.AdamW(groups, lr=optim['lr'], betas=tuple(optim['betas']),
            weight_decay=optim['weight_decay'], eps=optim['eps'], fused=True)
    optimizers = {role: Zero1FP32MasterAdamW([{'params': list(nets[role].parameters())}], make_inner)
                  for role in ('student', 'critic')}
    grads = {role: GradFlats(opt) for role, opt in optimizers.items()}
    warmup = config['critic_warmup_steps']
    totals = {'student': config['total_steps'] - warmup, 'critic': config['total_steps'] * updates}
    schedulers = {role: torch.optim.lr_scheduler.LambdaLR(opt.inner,
        lambda step, role=role: learning_rate_factor(step, totals[role],
            config['lr_warmup_steps'] if role == 'student' else config['critic_lr_warmup_steps'], config['lr_final_ratio']))
        for role, opt in optimizers.items()}
    counters = dict(outer_step=0, warmup_step=0, student_updates=0, critic_updates=0, data_steps=0)
    if resume:
        counters.update(load_rank_state(Path(resume), optimizers, schedulers, None, True, False))
    if main_process:
        print(json.dumps({'event': 'models_ready', 'seconds': time.time()-start, 'method': method,
            'world_size': world, 'nodes': nodes, 'global_student_batch': 160, 'global_critic_batch': 160,
            'global_pool_batch': 160 * updates, 'local_targets': local_rows*4,
            'allocated_gib': torch.cuda.memory_allocated()/2**30, 'signature': signature}), flush=True)
    total_steps = args.benchmark if benchmark else config['total_steps'] - counters['data_steps']
    loader_kwargs = dict(batch_size=None, sampler=range(counters['data_steps'], counters['data_steps'] + total_steps),
        num_workers=config['num_dataloader_workers'], pin_memory=True, generator=torch.Generator().manual_seed(config['seed']))
    if config['num_dataloader_workers']:
        loader_kwargs.update(prefetch_factor=config.get('prefetch_factor', 4), persistent_workers=True)
    loader = DataLoader(dataset, **loader_kwargs)
    run_dir = tracker = None
    if not benchmark:
        paths = [None]
        if main_process:
            paths[0] = str(Path(config['output_dir']) / datetime.now(timezone.utc).strftime('run-%Y%m%d-%H%M%S'))
            Path(paths[0]).mkdir(parents=True)
            (Path(paths[0])/'config.json').write_text(json.dumps(config, indent=2)+'\n')
        dist.broadcast_object_list(paths, src=0)
        run_dir = Path(paths[0])
        if main_process and config['wandb']['enabled']:
            import wandb
            tracker = wandb.init(project=config['wandb']['project'], name=config['wandb']['run_name'],
                                 config={**config, 'world_size': world, 'signature': signature}, dir=str(run_dir))
            (run_dir/'wandb.json').write_text(json.dumps({'url': tracker.url, 'id': tracker.id})+'\n')
    input_generator = GTConditionedStabilizer(config['band_steps'], history_t=.2)
    sums, timers, step_times = Sums(), {}, []
    @contextmanager
    def timed(name):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        yield
        end.record()
        timers.setdefault(name, []).append((begin, end))
    def prepare(raw, lo, hi):
        result = {'latents': raw['latents_low'][lo:hi].to(device, non_blocking=True).float(),
                  'pose': raw['pose_latents_low'][lo:hi].to(device, non_blocking=True).float(),
                  'ofi': raw['original_frame_indices'][lo:hi].to(device, non_blocking=True), 'history_t': .2}
        result['drop'] = raw['bad_audio'][lo:hi].to(device) if 'bad_audio' in raw else torch.zeros(hi-lo, dtype=torch.bool, device=device)
        codebooks = int(torch.randint(audio_config['min_codebooks'], audio_config['num_codebooks']+1, ()))
        result['audio'] = audio_latents(mimi, raw['codes'][lo:hi].to(device, non_blocking=True), codebooks)
        validity = result['ofi'][:, 5:10] >= 0
        sums.add('data/resolver_history_slots_dropped', (~validity[:, :4]).sum() + (~validity[:, 1:]).sum(), (hi-lo)*8)
        return result
    def subset(rows, lo, hi):
        return {k: v[lo:hi] if torch.is_tensor(v) else v for k, v in rows.items()}
    @torch.no_grad()
    def inputs(rows):
        with timed('input_generation'):
            if method == 'gt':
                return gt_noised_input(rows, torch.randn_like(rows['latents'][:, :, -4:]), t_start=.2)
            size = config.get('stabilizer_micro_rows', config['pool_micro_batch'])
            return torch.cat([input_generator(nets['stabilizer'], subset(rows, i, i+size))
                              for i in range(0, len(rows['latents']), size)])
    def scoring(net, rows, x, t):
        audio = net.frame_audio(net.audio_tokens(rows['audio']), rows['ofi'][:, target_start(rows):], rows['drop'])
        mask = torch.ones(len(rows['latents']), 4, dtype=torch.bool, device=device)
        return resolve_gt_pass(net, rows, x, t, audio, mask)
    def apply_update(role):
        norm = grads[role].reduce_clip(config['gradient_clipping'])
        if not torch.isfinite(norm):
            raise FloatingPointError(f'Nonfinite {role} gradient')
        sums.add(f'{role}/lr_used', optimizers[role].inner.param_groups[0]['lr'], 1)
        optimizers[role].step()
        schedulers[role].step()
        grads[role].release()
        sums.add(f'{role}/grad_norm', norm, 1)
        counters[role+'_updates'] += 1
    def critic_phase(raw):
        pool_rows = local_rows * updates
        rows = prepare(raw, 0, pool_rows)
        with torch.no_grad(), timed('pool'):
            x02 = inputs(rows)
            size = config['pool_micro_batch']
            generated = torch.cat([one_step_resolve(nets['student'], subset(rows, i, i+size), x02[i:i+size])
                                   for i in range(0, pool_rows, size)])
        for u in range(updates):
            with timed('critic_update'):
                grads['critic'].allocate()
                size = min(config.get('critic_micro_rows', local_rows), local_rows)
                for lo in range(0, local_rows, size):
                    count = min(size, local_rows-lo)
                    start = u*local_rows+lo
                    part, y = subset(rows, start, start+count), generated[start:start+count]
                    t = resolve_times(count, device, config['res_t_min'], config['res_t_max'])
                    x, target = noised(y, 0., t, torch.randn_like(y))
                    velocity = scoring(nets['critic'], part, x, t)
                    loss, frames = fm_loss(velocity, target)
                    (loss * count / local_rows * loss_rank_weight).backward()
                    sums.frames('fm_resolver', frames)
                apply_update('critic')
    def student_phase(raw):
        offset = local_rows * updates
        rows = prepare(raw, offset, offset+local_rows)
        x02 = inputs(rows)
        grads['student'].allocate()
        size = min(config['student_micro_rows'], local_rows)
        with timed('student'):
            for lo in range(0, local_rows, size):
                count = min(size, local_rows-lo)
                part = subset(rows, lo, lo+count)
                with timed('student_forward'):
                    y = one_step_resolve(nets['student'], part, x02[lo:lo+count])
                with torch.no_grad(), timed('teacher_critic_scoring'):
                    t = resolve_times(count, device, config['res_t_min'], config['res_t_max'])
                    x, target = noised(y, 0., t, torch.randn_like(y))
                    vt = scoring(nets['teacher'], part, x, t)
                    vf = scoring(nets['critic'], part, x, t)
                loss, frames, normalizer, difference = dmd_loss(y, vt, vf, target, config['normalizer_epsilon'])
                with timed('student_backward'):
                    (loss * count / local_rows * loss_rank_weight).backward()
                sums.frames('dmd_resolver', frames)
                sums.add('dmd_resolver/normalizer', normalizer.sum(), normalizer.numel())
                sums.add('dmd_resolver/difference', difference.sum(), difference.numel())
                with torch.no_grad():
                    cos = torch.nn.functional.cosine_similarity(vt.flatten(1), vf.flatten(1), dim=1)
                    sums.add('diagnostic/teacher_critic_velocity_cosine', cos.sum(), cos.numel())
                    sums.add('diagnostic/teacher_critic_velocity_mse', (vt-vf).square().sum(), vt.numel())
                    sums.add('diagnostic/resolver_output_rms_squared', y.detach().square().sum(), y.numel())
            apply_update('student')
    def save_checkpoint():
        directory = run_dir / f'step-{counters["data_steps"]:06d}'
        owned = sum(sum(m.numel()*m.element_size() for _, m in opt.pairs) + tensor_bytes(opt.inner.state) for opt in optimizers.values())
        need = torch.tensor(owned, dtype=torch.int64, device=device)
        dist.all_reduce(need)
        error = [None]
        if main_process:
            weights = sum(p.numel()*p.element_size() for role in ('student','critic') for p in nets[role].parameters())
            required = (int(need)+weights)*1.1 + 1024**3
            if shutil.disk_usage(run_dir).free < required:
                error[0] = f'Insufficient disk for atomic checkpoint: need {required/2**30:.1f}GiB'
            else:
                directory.mkdir(parents=True, exist_ok=False)
        dist.broadcast_object_list(error, src=0)
        if error[0]:
            raise RuntimeError(error[0])
        dist.barrier()
        if main_process:
            for role in ('student', 'critic'):
                tensors = {n: p.detach().cpu().contiguous() for n,p in nets[role].original_named_parameters()}
                save_file(tensors, str(directory/f'{role}.safetensors'))
                del tensors
        save_rank_states(directory, optimizers, schedulers, None, counters, True)
        if main_process:
            complete_checkpoint(directory, {'world_size': world, 'compatibility': signature, 'counters': dict(counters),
                'training': config, 'data_sources': data_sources, 'dataset_length': len(dataset),
                'optimizer_state_retained': True, 'critic_warmup_complete': counters['warmup_step']>=warmup,
                'lora': None, 'raw_weights': True})
            # Commit new complete state before removing superseded optimizer shards.
            for older in sorted(run_dir.glob('step-*')):
                if older != directory and list(older.glob('rank-*.pt')):
                    old = json.loads((older/'metadata.json').read_text())
                    prune_optimizer_state(older.absolute(), old['checkpoint_id'], verify_weight_hashes=False)
            print(json.dumps({'event': 'checkpoint_complete', 'path': str(directory), 'step': counters['data_steps']}), flush=True)
        dist.barrier()
    torch.cuda.empty_cache()
    gc.freeze()
    gc.disable()
    for index, raw in enumerate(loader):
        wall = time.time()
        warming = not benchmark and counters['warmup_step'] < warmup
        if config.get('compile_mode') == 'block_cudagraph':
            torch.compiler.cudagraph_mark_step_begin()
        with timed('outer'):
            critic_phase(raw)
            if not warming:
                student_phase(raw)
        counters['warmup_step' if warming else 'outer_step'] += 1
        counters['data_steps'] += 1
        torch.cuda.synchronize()
        elapsed = torch.tensor(time.time()-wall, device=device)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        step_times.append(elapsed.item())
        out = {'step': counters['data_steps'], **counters, 'seconds/wall': elapsed.item(),
               'global/student_batch': 160, 'global/critic_batch': 160, 'global/pool_batch': 160*updates,
               'phase': 'critic_warmup' if warming else 'joint', **sums.report()}
        if 'diagnostic/teacher_critic_velocity_mse' in out:
            out['diagnostic/teacher_critic_velocity_rms'] = out.pop('diagnostic/teacher_critic_velocity_mse') ** .5
            out['diagnostic/resolver_output_rms'] = out.pop('diagnostic/resolver_output_rms_squared') ** .5
        for name, pairs in timers.items():
            seconds = torch.tensor(sum(s.elapsed_time(e) for s,e in pairs)/1000, device=device)
            dist.all_reduce(seconds, op=dist.ReduceOp.MAX)
            out['seconds/'+name] = seconds.item()
        timers.clear()
        peak = torch.tensor(torch.cuda.max_memory_allocated()/2**30, device=device)
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        out['memory/peak_allocated_gib'] = peak.item()
        torch.cuda.reset_peak_memory_stats()
        for role in optimizers:
            out[role+'/lr_next'] = optimizers[role].inner.param_groups[0]['lr']
        if main_process:
            print(json.dumps(out), flush=True)
            if run_dir:
                with (run_dir/'metrics.jsonl').open('a') as handle:
                    handle.write(json.dumps(out)+'\n')
            if tracker:
                tracker.log(out, step=out['step'])
        if not benchmark and (counters['data_steps'] % config['save_every_n_steps'] == 0 or counters['data_steps'] == config['total_steps']):
            save_checkpoint()
        if index < 4 or counters['data_steps'] in (warmup+1, warmup+2):
            gc.freeze()
        elif counters['data_steps'] % 50 == 0:
            gc.collect()
    if benchmark and main_process:
        measured = step_times[args.benchmark_skip:]
        if not measured:
            raise ValueError('Benchmark needs measured steps after skips')
        print(json.dumps({'benchmark': True, 'method': method, 'world_size': world, 'steps': measured,
            'median': float(np.median(measured)), 'min': min(measured), 'mean': float(np.mean(measured))}), flush=True)
    if tracker:
        tracker.finish()
    if run_dir and main_process:
        (run_dir/'training.complete.json').write_text(json.dumps({'complete': True, 'counters': counters,
            'final_lr_ratio': config['lr_final_ratio'], 'completed_at': datetime.now(timezone.utc).isoformat()})+'\n')
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
