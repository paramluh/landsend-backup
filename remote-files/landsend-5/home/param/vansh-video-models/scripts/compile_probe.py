"""Independent node-local compile probes, with real model shapes and no updates.

Run with torchrun --standalone --nproc_per_node=8 scripts/compile_probe.py ...
Each cycle exercises four critic FM backward pathways and one student DMD
backward pathway, at the actual per-GPU microbatch size of two. There is one
microbatch per objective (16 source rows across eight GPUs), no optimizer/EMA,
and no parameter update. This measures compilation and activation residency;
it is explicitly not a production-throughput or optimizer-memory benchmark.
"""

import argparse
from contextlib import contextmanager
import copy
from datetime import datetime, timedelta, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import types

import numpy as np
import torch
import torch.distributed as dist

# Direct script invocation otherwise exposes scripts/, not the source root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

VARIANTS = ('block_shared', 'parts_dynamic', 'block_untrimmed', 'block_role_isolated', 'block_dynamic')


@contextmanager
def variant_context(variant):
    """Change only compile/layout callables inside this disposable process."""
    from distillation import bimodal_net
    from distillation.full_gt_dmd import FullGTStabilizer

    if variant not in VARIANTS:
        raise ValueError(f'Unknown compile probe variant: {variant}')
    original_factory, original_init = bimodal_net.compiled_whole_block, FullGTStabilizer.__init__
    original_whole = dict(bimodal_net._WHOLE)
    original_parts = (bimodal_net.PRE, bimodal_net.POST, bimodal_net.ROPE)
    original_coalesce = torch._inductor.config.triton.coalesce_tiling_analysis
    bimodal_net._WHOLE.clear()
    created = []
    description = {'variant': variant, 'compile_mode': 'parts' if variant == 'parts_dynamic' else 'block',
                   'trim_stabilizer_future_slots': variant != 'block_untrimmed',
                   'whole_block_dynamic_shapes': variant == 'block_dynamic',
                   'role_dynamo_code_isolation': variant == 'block_role_isolated',
                   'inductor_disk_cache_may_reuse_kernels': True, 'compiled_wrappers_created': created}
    if variant == 'block_dynamic':
        torch._inductor.config.triton.coalesce_tiling_analysis = False
        def factory(mode=bimodal_net.TORCH_COMPILE_MODE):
            if mode not in bimodal_net._WHOLE:
                bimodal_net._WHOLE[mode] = torch.compile(bimodal_net.block_forward, dynamic=True, mode=mode)
                created.append({'mode': mode, 'shared': True, 'dynamic': True})
            return bimodal_net._WHOLE[mode]
        bimodal_net.compiled_whole_block = factory
    elif variant == 'block_role_isolated':
        def factory(mode=bimodal_net.TORCH_COMPILE_MODE):
            index = len(created)
            role = ('teacher', 'student', 'critic')[index] if index < 3 else f'net_{index}'
            original = bimodal_net.block_forward
            name = f'block_forward_compile_probe_{role}'
            # A new wrapper alone still shares Dynamo's code-object cache.
            # Distinct code objects make this a genuine Dynamo isolation control;
            # backend kernels remain eligible for Inductor disk-cache reuse.
            code = original.__code__.replace(co_name=name, co_qualname=name)
            functional = types.FunctionType(code, original.__globals__, name,
                                            original.__defaults__, original.__closure__)
            functional.__kwdefaults__ = original.__kwdefaults__
            created.append({'mode': mode, 'role': role, 'shared': False, 'dynamic': False})
            return torch.compile(functional, dynamic=False, mode=mode)
        bimodal_net.compiled_whole_block = factory
    if variant == 'block_untrimmed':
        def initializer(self, *args, **kwargs):
            kwargs['trim'] = False
            original_init(self, *args, **kwargs)
        FullGTStabilizer.__init__ = initializer
    try:
        yield description
    finally:
        bimodal_net.compiled_whole_block = original_factory
        FullGTStabilizer.__init__ = original_init
        bimodal_net._WHOLE.clear()
        bimodal_net._WHOLE.update(original_whole)
        bimodal_net.PRE, bimodal_net.POST, bimodal_net.ROPE = original_parts
        torch._inductor.config.triton.coalesce_tiling_analysis = original_coalesce


def compiler_stats():
    counters = torch._dynamo.utils.counters
    return {'unique_graphs': int(counters['stats']['unique_graphs']),
            'graph_breaks': int(sum(counters['graph_break'].values())),
            'frames_total': int(counters['frames']['total']), 'frames_ok': int(counters['frames']['ok']),
            'compile_seconds_by_timer': {key: float(sum(values)) for key, values in
                                         torch._dynamo.utils.compilation_time_metrics.items()}}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=2)
    parser.add_argument('--expected-world-size', type=int, default=8)
    parser.add_argument('--recompile-logs', action='store_true', help='Print guard failures on local rank zero only')
    args = parser.parse_args()
    if args.iterations < 2:
        parser.error('At least two complete cycles are needed to inspect compilation stability')
    from distillation.bimodal_net import audio_latents, build_net, load_mimi, load_state_dict
    from distillation.bimodal_dmd import noised
    from distillation.full_gt_data import build_full_gt_data
    from distillation.full_gt_dmd import FullGTStabilizer, generate_resolver, score_resolver
    from distillation.joint_dmd import endpoint_fm_loss
    from learn_dmd_bimodal import load_config
    from learn_dmd_joint import dmd_frames, validate_config

    config = load_config(args.config, [])
    validate_config(config, full_gt=True)
    if config['micro_batch_size_per_gpu'] != 2:
        raise ValueError('This compile experiment must retain the real per-GPU microbatch size of two')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
    dist.init_process_group('nccl', device_id=device, timeout=timedelta(hours=2))
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != args.expected_world_size or world != int(os.environ['LOCAL_WORLD_SIZE']):
        raise ValueError('Each independent variant must use exactly one whole node')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config['seed']+rank)
    np.random.seed(config['seed']+rank)
    random.seed(config['seed']+rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch._dynamo.config.cache_size_limit = config.get('compile_cache_size_limit', 64)
    torch._dynamo.config.accumulated_cache_size_limit = config.get('compile_accumulated_cache_size_limit', 256)
    torch._dynamo.config.fail_on_recompile_limit_hit = True
    if args.recompile_logs and rank == 0:
        torch._logging.set_logs(recompiles=True)
    records = []
    started = time.monotonic()

    def announce(event, **details):
        if rank == 0:
            print(json.dumps({'event': event, 'variant': args.variant, **details}), flush=True)

    @contextmanager
    def measure(cycle, phase, role):
        torch.cuda.synchronize()
        # Compilation can skew rank arrival by minutes. Align each node-local
        # phase before timing; report maxima across ranks in the final receipt.
        dist.barrier()
        before, at = compiler_stats(), time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        announce('phase_started', cycle=cycle, phase=phase, role=role)
        observations = {}
        yield observations
        torch.cuda.synchronize()
        after = compiler_stats()
        record = {'cycle': cycle, 'phase': phase, 'role': role, 'seconds': time.monotonic()-at,
                  'new_graphs': after['unique_graphs']-before['unique_graphs'], **after,
                  'new_compile_seconds_by_timer': {key: value-before['compile_seconds_by_timer'].get(key, 0.)
                       for key, value in after['compile_seconds_by_timer'].items()},
                  'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
                  'peak_reserved_gib': torch.cuda.max_memory_reserved()/2**30,
                  'allocated_after_gib': torch.cuda.memory_allocated()/2**30, **observations}
        records.append(record)
        write_json(output/f'rank-{rank:02d}.progress.json', {'rank': rank, 'records': records})
        announce('phase_complete', **record)

    announce('compile_probe_start', world_size=world, microbatch_per_gpu=2,
             rows_per_objective_per_cycle=world*2, parameter_updates=0, optimizer_and_ema=False)
    try:
        with variant_context(args.variant) as variant:
            data_config = copy.deepcopy(config)
            data_config['global_batch_size'] = world*2
            data = build_full_gt_data(data_config, rank, world)
            example = data.group(0)[0]
            latent_hw = tuple(example['latents_low'].shape[-2:])
            del example
            mimi, template = load_mimi(config['model']['audio'], device)
            state = load_state_dict(config['model']['checkpoint_path'])
            nets = {}
            for role in ('teacher', 'student', 'critic'):
                net = build_net(state, config['model']['audio'], latent_hw, device, template, num_refs=9)
                net.train(role != 'teacher').requires_grad_(role != 'teacher')
                if role == 'teacher':
                    net.shard_blocks(dist.group.WORLD)
                net.configure(True, 'none' if role == 'teacher' else 'block', variant['compile_mode'])
                net.cache_references = True
                nets[role] = net
                announce('model_ready', role=role)
            del net, state, template
            variant['coalesce_tiling_analysis'] = torch._inductor.config.triton.coalesce_tiling_analysis
            rollout = FullGTStabilizer(config['band_steps'], config['history_t'], config['stabilizer_history_dropout'])

            def prepare(raw):
                row = {key: raw[source].to(device, non_blocking=True) for key, source in
                       (('latents', 'latents_low'), ('pose', 'pose_latents_low'), ('ofi', 'original_frame_indices'),
                        ('refs', 'refs'), ('ref_keep', 'ref_keep'))}
                for key in ('latents', 'pose', 'refs'):
                    row[key] = row[key].float()
                row['drop'] = raw.get('bad_audio', torch.zeros(2, dtype=torch.bool)).to(device)
                audio = config['model']['audio']
                codebooks = int(torch.randint(audio['min_codebooks'], audio['num_codebooks']+1, ()))
                row['audio'] = audio_latents(mimi, raw['codes'].to(device), codebooks)
                return row

            def generate(row, kind):
                noise, context = rollout.prepare_context(row)
                return (rollout.generate(nets['student'], row, noise=noise, context=context) if kind == 'stabilizer'
                        else generate_resolver(nets['student'], row, context, noise=noise[:, :, -1:]))

            def score(net, row, kind, x, times, context):
                return (rollout.score_pass(net, row, x, times, context) if kind == 'stabilizer'
                        else score_resolver(net, row, x, times, context))

            def corrupt(y, kind):
                low, high = ((.200001, .999999) if kind == 'stabilizer'
                             else (config['res_t_min'], config['res_t_max']))
                times = low + torch.rand((y.shape[0], 1), device=device)*(high-low)
                noise = torch.randn_like(y)
                x, target = noised(y.detach(), .2 if kind == 'stabilizer' else 0., times, noise)
                return x, times, target, noise

            gc.collect()
            dist.barrier()
            announce('models_and_data_ready', initialization_seconds=time.monotonic()-started,
                     variant_settings=variant, production_global_batch_size=config['global_batch_size'],
                     probe_global_batch_size=data_config['global_batch_size'])
            for cycle in range(args.iterations):
                for update in range(config['critic_updates']):
                    nets['critic'].zero_grad(set_to_none=True)
                    group = data.critic_group(cycle, update)
                    for raw in group:
                        kind, row = raw['objective'], prepare(raw)
                        with measure(cycle, f'critic_{update}_{kind}_pool', 'student'), torch.no_grad():
                            y, context = generate(row, kind)
                        x, times, target, noise = corrupt(y, kind)
                        with measure(cycle, f'critic_{update}_{kind}_fm_backward', 'critic') as observations:
                            velocity = score(nets['critic'], row, kind, x, times, context)
                            loss = endpoint_fm_loss(velocity, y, times, noise,
                                                    history_t=.2 if kind == 'stabilizer' else 0.)[0]
                            observations['loss'] = float(loss.detach())
                            if not np.isfinite(observations['loss']):
                                raise FloatingPointError('Nonfinite endpoint FM probe loss')
                            observations['source_window_ids'] = raw['epoch_window_id'].tolist()
                            (loss*config['objective_weights'][kind]).backward()
                        del velocity, loss, x, times, target, noise, y, context, row
                    nets['critic'].zero_grad(set_to_none=True)
                    del group
                nets['student'].zero_grad(set_to_none=True)
                for raw in data.group(cycle):
                    kind, row = raw['objective'], prepare(raw)
                    with measure(cycle, f'student_{kind}_rollout', 'student'):
                        y, context = generate(row, kind)
                    with torch.no_grad():
                        x, times, target, noise = corrupt(y, kind)
                        with measure(cycle, f'teacher_{kind}_scoring', 'teacher'):
                            teacher = score(nets['teacher'], row, kind, x, times, context)
                        with measure(cycle, f'critic_{kind}_scoring', 'critic'):
                            critic = score(nets['critic'], row, kind, x, times, context)
                    with measure(cycle, f'student_{kind}_dmd_backward', 'student') as observations:
                        frames = dmd_frames(y, teacher, critic, target, config['normalizer_epsilon'])
                        observations['loss'] = float(frames.detach().mean())
                        if not np.isfinite(observations['loss']):
                            raise FloatingPointError('Nonfinite DMD probe loss')
                        observations['source_window_ids'] = raw['epoch_window_id'].tolist()
                        (frames.mean()*config['objective_weights'][kind]).backward()
                    del frames, y, context, x, times, target, noise, teacher, critic, row
                nets['student'].zero_grad(set_to_none=True)
                gc.collect()
                dist.barrier()
                announce('cycle_complete', cycle=cycle, **compiler_stats())
            summary = {'format': 'node-local-full-gt-compile-probe-v1', 'complete': True,
                       'variant': variant, 'rank': rank, 'world_size': world,
                       'microbatch_per_gpu': 2, 'global_rows_per_objective': world*2,
                       'cycles': args.iterations, 'critic_pathways_per_cycle': config['critic_updates'],
                       'optimizer_and_ema': False, 'parameter_updates': 0,
                       'not_a_production_throughput_benchmark': True,
                       'new_graphs_last_cycle': sum(row['new_graphs'] for row in records if row['cycle'] == args.iterations-1),
                       'records': records, 'compiler': compiler_stats(),
                       'compiler_counters': {key: dict(value) for key, value in torch._dynamo.utils.counters.items()},
                       'fail_on_recompile_limit_hit': True,
                       'compile_timer_note': 'PyTorch timers are nested and must not be added together',
                       'peak_allocated_gib': max(row['peak_allocated_gib'] for row in records),
                       'seconds_total': time.monotonic()-started, 'torch_version': torch.__version__,
                       'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(),
                       'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       'completed_at_utc': datetime.now(timezone.utc).isoformat()}
            write_json(output/f'rank-{rank:02d}.complete.json', summary)
            summaries = [None]*world
            dist.all_gather_object(summaries, {key: value for key, value in summary.items() if key != 'records'})
            if rank == 0:
                all_records = [json.loads((output/f'rank-{index:02d}.complete.json').read_text())['records']
                               for index in range(world)]
                maxima = [{'cycle': row[0]['cycle'], 'phase': row[0]['phase'], 'role': row[0]['role'],
                           'seconds_max': max(value['seconds'] for value in row),
                           'seconds_median': float(np.median([value['seconds'] for value in row])),
                           'peak_allocated_gib_max': max(value['peak_allocated_gib'] for value in row),
                           'new_graphs_max': max(value['new_graphs'] for value in row)}
                          for row in zip(*all_records)]
                write_json(output/'probe.complete.json', {'complete': True, 'variant': variant,
                    'all_ranks_stable': all(value['new_graphs_last_cycle'] == 0 for value in summaries),
                    'phase_rank_maxima': maxima, 'ranks': summaries})
                announce('compile_probe_complete', all_ranks_stable=all(value['new_graphs_last_cycle'] == 0 for value in summaries))
    except BaseException as error:
        write_json(output/f'rank-{rank:02d}.failed.json', {'variant': args.variant, 'rank': rank,
                   'error': f'{type(error).__name__}: {error}', 'records': records, 'compiler': compiler_stats()})
        raise
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
