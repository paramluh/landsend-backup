"""Standalone four-GPU benchmark of frozen-partner conversational rollouts.

Run as a module. Student uses two GPUs and the frozen speech-only partner uses
two other GPUs. This measures actual sampled rollouts, not optimizer updates.
No learner, KL-scoring teachers, gradients, or Adam state are resident: its
memory results must NOT be presented as evidence that a training update fits.

Example:
  python -m finetune.pose_opd_credit.teacher_partner_benchmark \
    --config finetune/pose_opd_credit/recipe.teacher_partner.json \
    --output /path/to/new-benchmark --instances 16 --frames 125 --repeats 2

Cache capacity defaults to the recipe's full two-minute shape even when only
a short waveform prefix is sampled. Use --profile-calls 0 for an uninstrumented
timing comparison; event profiles otherwise cover a bounded number of calls.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import functools
import gc
import hashlib
import itertools
import json
from pathlib import Path
import pickle
import socket
import time
import traceback

import numpy as np


def arguments(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--instances', type=int, default=16)
    parser.add_argument('--batch-sizes', help='Reuse loaded weights across these batches, e.g. 2,16,32')
    parser.add_argument('--frames', type=int, default=125)
    parser.add_argument('--capacity-frames', type=int)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--row-offset', type=int, default=0)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--student-devices', default='0,1')
    parser.add_argument('--partner-devices', default='2,3')
    parser.add_argument('--student-backbone-cut', type=int)
    parser.add_argument('--partner-backbone-cut', type=int)
    parser.add_argument('--profile-calls', type=int, default=32)
    parser.add_argument('--profile-warmup', type=int, default=25)
    parser.add_argument('--profile-kernel-graphs', action='store_true')
    parser.add_argument('--disable-partner-text-history', action='store_true')
    parser.add_argument('--teacher-replay', choices=('sequential', 'block'), default='sequential')
    parser.add_argument('--replay-backends', help='Compare sequential,block with the same loaded weights and rows')
    parser.add_argument('--delivery-delays', help='Compare one-way PCM delivery delays in samples, e.g. 0,72')
    parser.add_argument('--prepared-rows', type=Path,
                        help='Trusted admitted-row cache from the layout benchmark; avoids another eligibility scan')
    parser.add_argument('--rollout-seeds', type=Path,
                        help='Explicit seed list, indexed by row offset, for a single listening trial')
    parser.add_argument('--save-conversations', action='store_true',
                        help='Save paired float32 audio and all generated codes after each completed rollout')
    parser.add_argument('--verify-block', action='store_true')
    parser.add_argument('--pack-partner-projections', choices=('none', 'qkv', 'mlp', 'all'), default='none')
    parser.add_argument('--mimi-placement', choices=('frontend', 'audio', 'student-audio'), default='frontend')
    parser.add_argument('--reserve-gib', type=float, default=0.,
                        help='Synthetic allocation per GPU to bound resident training-shard memory')
    parser.add_argument('--no-graphs', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--qwen32-config', default='/home/param/models/imported-weights/ppm-811-17850-base-bf16/configs/Qwen3-32B')
    parser.add_argument('--qwen4-config', default='/home/param/models/imported-weights/ppm-811-17850-base-bf16/configs/Qwen3-4B')
    parser.add_argument('--voxtral', default='/home/param/models/imported-weights/aloe-ft-vox17127-nopersona/assets/voxtral-realtime')
    parser.add_argument('--mimi', default='/home/param/models/mimi')
    args = parser.parse_args(argv)
    if not np.isfinite(args.reserve_gib) or args.reserve_gib < 0:
        parser.error('--reserve-gib must be finite and nonnegative')
    if not 1 <= args.instances <= 64:
        parser.error('--instances must be between1 and64')
    try:
        args.batch_sizes = [int(value) for value in args.batch_sizes.split(',')] if args.batch_sizes else [args.instances]
    except ValueError:
        parser.error('--batch-sizes must be comma-separated integers')
    if (not args.batch_sizes or len(set(args.batch_sizes)) != len(args.batch_sizes)
            or any(not 1 <= value <= 64 for value in args.batch_sizes)):
        parser.error('--batch-sizes must contain distinct values between1 and64')
    args.replay_backends = args.replay_backends.split(',') if args.replay_backends else [args.teacher_replay]
    if (not args.replay_backends or len(set(args.replay_backends)) != len(args.replay_backends)
            or any(value not in ('sequential', 'block') for value in args.replay_backends)):
        parser.error('--replay-backends must contain distinct sequential and/or block values')
    if args.delivery_delays is not None:
        try:
            args.delivery_delays = [int(value) for value in args.delivery_delays.split(',')]
        except ValueError:
            parser.error('--delivery-delays must be comma-separated integer sample counts')
        if (not args.delivery_delays or len(set(args.delivery_delays)) != len(args.delivery_delays)
                or any(not 0 <= value < 1920 for value in args.delivery_delays)):
            parser.error('--delivery-delays must contain distinct values between 0 and 1919')
    if args.frames < 1 or args.repeats < 1 or args.row_offset < 0 or args.profile_calls < 0 or args.profile_warmup < 0:
        parser.error('Positive frames/repeats and nonnegative offset/profile counts are required')
    if args.rollout_seeds is not None and args.repeats != 1:
        parser.error('Explicit listening seeds require --repeats 1')
    for name in ('student_devices', 'partner_devices'):
        try:
            devices = tuple(int(value) for value in getattr(args, name).split(','))
        except ValueError:
            parser.error('GPU devices must be two comma-separated integers')
        if len(devices) != 2 or any(value < 0 for value in devices) or len(set(devices)) != 2:
            parser.error('Each actor requires two distinct nonnegative GPU indices')
        setattr(args, name, devices)
    if set(args.student_devices) & set(args.partner_devices):
        parser.error('This baseline requires four distinct GPUs')
    return args


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + '\n')
    temporary.replace(path)


def _prepared_benchmark_rows(cfg, args, maximum_batch, log):
    """Reuse trusted admitted items; delivery delay does not affect row selection."""
    from .data import ProductionRows, prefix_seed
    if args.prepared_rows is None:
        data = ProductionRows(cfg)
        items = data.items()
        selected = items[args.row_offset:args.row_offset + maximum_batch]
        rows = [prefix_seed(data.load(item, rank=0, update=0, slot=slot), args.frames)
                for slot, item in enumerate(selected)]
        return rows, selected, len(items), False
    with args.prepared_rows.open('rb') as source:
        cached = pickle.load(source)
    if cached.get('format') != 'teacher-partner-prepared-rows-v1':
        raise ValueError('Unknown prepared-row cache format')
    key = cached['key']
    wanted = json.loads(json.dumps(cfg.to_dict()))
    recorded = dict(key['config'])
    for config in (wanted, recorded):
        config.pop('partner_audio_delay_samples', None)
    if wanted != recorded:
        raise ValueError('Prepared rows have a different data/training recipe')
    if key['data_sha256'] != hashlib.sha256(Path(__file__).with_name('data.py').read_bytes()).hexdigest():
        raise ValueError('Prepared rows use a different data implementation')
    begin = args.row_offset - key['row_offset']
    end = begin + maximum_batch
    if begin < 0 or end > len(cached['rows']):
        raise ValueError('Prepared cache does not cover the requested row range')
    selected = cached['preparation']['selected_items'][begin:end]
    if key['frames'] >= args.frames:
        rows = [prefix_seed(row, args.frames) for row in cached['rows'][begin:end]]
        log('prepared_rows_cache_hit', path=str(args.prepared_rows), selected_rows=len(rows))
    else:
        data = ProductionRows(cfg)
        rows = [prefix_seed(data.load(item, rank=0, update=0, slot=begin + slot), args.frames)
                for slot, item in enumerate(selected)]
        log('prepared_items_reloaded', path=str(args.prepared_rows), selected_rows=len(rows), frames=args.frames)
    return rows, selected, cached['preparation']['eligible_rows'], True


def _sync(torch, devices):
    for device in devices:
        torch.cuda.synchronize(device)


def memory_report(torch, devices):
    report = []
    for device in devices:
        free, total = torch.cuda.mem_get_info(device)
        report.append({
            'device': str(device), 'name': torch.cuda.get_device_name(device),
            'allocated_gib': torch.cuda.memory_allocated(device) / 2 ** 30,
            'reserved_gib': torch.cuda.memory_reserved(device) / 2 ** 30,
            'peak_allocated_gib': torch.cuda.max_memory_allocated(device) / 2 ** 30,
            'peak_reserved_gib': torch.cuda.max_memory_reserved(device) / 2 ** 30,
            'device_free_gib': free / 2 ** 30, 'device_total_gib': total / 2 ** 30,
        })
    return report


class CudaPhaseRecorder:
    """Bounded primary-stream event samples with no per-frame synchronization.

    Per-device durations can overlap and must not be added as wall time. Events
    cover work joined to each device's primary stream by the actor method; the
    runtime's own timers additionally cover frontends, Mimi and transfers.
    """
    def __init__(self, torch, *, calls=32, warmup=0):
        self.torch, self.limit, self.warmup = torch, int(calls), int(warmup)
        self.events = defaultdict(list)
        self.counts = defaultdict(int)
        self.host_seconds = defaultdict(float)
        self.original = []

    def attach(self, engine, role, devices):
        for name in ('sample_outputs', 'advance_backbone', 'rewind'):
            if not hasattr(engine, name):
                raise AttributeError(f'Actor lacks required partner-rollout method {name}')
            original = getattr(engine, name)

            @functools.wraps(original)
            def wrapped(*args, _name=name, _original=original, **kwargs):
                label = f'{role}.{_name}'
                if _name == 'sample_outputs' and (kwargs.get('force') or {}).get('codes') is not None:
                    label += '.forced'
                self.counts[label] += 1
                capture = self.counts[label] > self.warmup and len(self.events[label]) < self.limit
                pairs = []
                if capture:
                    for device in devices:
                        with self.torch.cuda.device(device):
                            start = self.torch.cuda.Event(enable_timing=True)
                            end = self.torch.cuda.Event(enable_timing=True)
                            start.record()
                        pairs.append((device, start, end))
                began = time.perf_counter()
                try:
                    return _original(*args, **kwargs)
                finally:
                    self.host_seconds[label] += time.perf_counter() - began
                    if capture:
                        for device, _, end in pairs:
                            with self.torch.cuda.device(device):
                                end.record()
                        self.events[label].append(pairs)

            self.original.append((engine, name, original))
            setattr(engine, name, wrapped)

    def detach(self):
        for engine, name, original in reversed(self.original):
            setattr(engine, name, original)
        self.original.clear()

    def report(self):
        result = {}
        for label, count in self.counts.items():
            timings = defaultdict(list)
            for sample in self.events[label]:
                for device, start, end in sample:
                    timings[str(device)].append(start.elapsed_time(end))
            result[label] = {
                'calls': count, 'sampled_calls': len(self.events[label]),
                'host_submit_seconds': self.host_seconds[label],
                'device_primary_stream_ms': {
                    device: {'mean': float(np.mean(values)), 'median': float(np.median(values)),
                             'min': min(values), 'max': max(values), 'total_sampled': sum(values)}
                    for device, values in timings.items()},
            }
        return result


class KernelGraphRecorder:
    """Time each single-device graph after its input-copy dependencies."""
    def __init__(self, torch, calls=32, warmup=25):
        self.torch, self.calls, self.warmup = torch, calls, warmup
        self.counts, self.events, self.original = defaultdict(int), defaultdict(list), []

    def run(self, name, device, fn):
        self.counts[name] += 1
        if not self.warmup < self.counts[name] <= self.warmup + self.calls:
            return fn()
        with self.torch.cuda.device(device):
            start, end = self.torch.cuda.Event(enable_timing=True), self.torch.cuda.Event(enable_timing=True)
            start.record()
            result = fn()
            end.record()
        self.events[name].append((start, end))
        return result

    def attach(self, engine, role):
        recorder = self
        class Proxy:
            def __init__(self, name, graph):
                self.name, self.graph = name, graph
            def replay(self):
                return recorder.run(self.name, recorder.torch.cuda.current_device(), self.graph.replay)
            def __getattr__(self, name):
                return getattr(self.graph, name)
        for name, graph in list(engine.graphs.items()):
            self.original.append((engine.graphs, name, graph))
            engine.graphs[name] = Proxy(role + '.' + str(name), graph)
        engine._opd_graph_profiler = self

    def detach(self, engines):
        for mapping, name, graph in self.original:
            mapping[name] = graph
        for engine in engines:
            engine._opd_graph_profiler = None

    def report(self):
        return {name: {'samples': len(events), 'median_ms': float(np.median([a.elapsed_time(b) for a, b in events])),
            'mean_ms': float(np.mean([a.elapsed_time(b) for a, b in events])),
            'p90_ms': float(np.percentile([a.elapsed_time(b) for a, b in events], 90))}
            for name, events in self.events.items()}


def _build_model(torch, M, spec, devices, *, backbone_cut, require_pose, log):
    from .actor import build_actor
    from finetune.duplex_distill.checkpoints import load_checkpoint
    cfgs = (M.cfg32(spec['layers']), M.cfg4b(spec['tower_layers']), M.cfg06(spec['dec_layers']))
    with torch.device('meta'):
        template = M.PPM(*cfgs, proj_r=spec['proj_r'], tower_vocab=spec['tower_vocab'],
                         pose_head=spec['pose_head'], decp_bb=require_pose,
                         vadapter=spec['vadapter'], nopose=not require_pose)
    model, layer_devices = build_actor(template, cfgs, devices, spec['vadapter'],
        spec['decp_bb_alpha'], backbone_cut=backbone_cut, require_pose=require_pose)
    del template
    receipt = load_checkpoint(model, Path(spec['path']), devices[0], gain_policy=spec['gain_policy'])
    model._opd_actor_published = True
    log('checkpoint_loaded', role=spec['role'], receipt=receipt)
    return model, layer_devices, cfgs


def _audit_result(result, rows):
    rollouts = result['rollouts']
    if len(rollouts) != len(rows):
        raise ValueError('Partner rollout did not return every student row')
    fingerprints = []
    for row, rollout in zip(rows, rollouts):
        if (rollout.conversation_frames != int(row['conversation_frames'])
                or rollout.prefill_frames != int(row['prefill_frames'])):
            raise ValueError('Returned student rollout changed its frame/prefill contract')
        digest = hashlib.sha256()
        for stream, values in (('text', rollout.text), ('audio', rollout.audio), ('pose', rollout.pose)):
            digest.update(np.asarray(values).tobytes())
            mask = np.asarray(rollout.sampled_masks[stream])
            logp = np.asarray(rollout.behavior_logprobs[stream])
            if not np.isfinite(logp[mask]).all():
                raise ValueError(f'Nonfinite/missing actual behavior logprobs for {stream}')
        fingerprints.append(digest.hexdigest())
    if len(result['latents']) != len(rows):
        raise ValueError('Partner rollout did not return the exact heard student input latents')
    def causal_digest(value):
        digest = hashlib.sha256()
        def visit(item):
            if isinstance(item, dict):
                for key in sorted(item):
                    digest.update(str(key).encode()); visit(item[key])
            elif isinstance(item, (list, tuple)):
                digest.update(str(len(item)).encode())
                for entry in item:
                    visit(entry)
            else:
                if hasattr(item, 'detach'):
                    item = item.detach().cpu()
                    if str(item.dtype) == 'torch.bfloat16':
                        item = item.float()
                array = np.asarray(item)
                digest.update(str(array.dtype).encode()); digest.update(str(array.shape).encode())
                if np.issubdtype(array.dtype, np.inexact) and np.isnan(array).any():
                    array = array.copy(); array[np.isnan(array)] = np.nan
                digest.update(array.tobytes())
        visit(value)
        return digest.hexdigest()
    causal = {key: causal_digest(result[key]) for key in ('waveforms', 'latents', 'partner_rollouts')}
    causal['student_masks_and_behavior'] = causal_digest([
        {'sampled_masks': rollout.sampled_masks, 'behavior_logprobs': rollout.behavior_logprobs}
        for rollout in rollouts])
    return {'rows': len(rows), 'rollout_sha256': fingerprints, 'causal_sha256': causal,
            'prefill_frames': [int(row['prefill_frames']) for row in rows]}


def _storage_identity(model):
    """No GPU copy: ensure cache disposal never replaces model weight storage."""
    return tuple((name, value.data_ptr(), tuple(value.shape), str(value.device), str(value.dtype))
                 for name, value in model.state_dict().items())


def _release_engines(torch, engines, loaded, devices):
    """Remove instance hooks/cycles and caches, preserving separately owned weights."""
    _sync(torch, devices)
    for engine in engines.values():
        blocks = getattr(engine, '_opd_partner_blocks', {})
        for block in blocks.values():
            block.graphs.clear()
        blocks.clear()
        engine.close()
        engine.graphs.clear()
    engines.clear()
    # An OOM partway through the constructor can leave its already-registered
    # bound input hook holding a partial engine alive. Only remove hooks created
    # since the corresponding bare model was loaded by this harness.
    for role in loaded.values():
        hooks = role['model'].inp._forward_hooks
        for key in set(hooks) - role['initial_hook_ids']:
            del hooks[key]
    engine = block = blocks = None
    gc.collect()
    _sync(torch, devices)
    for device in devices:
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    for name, role in loaded.items():
        if _storage_identity(role['model']) != role['storage_identity']:
            raise RuntimeError(f'{name}: disposing actor caches changed bare model tensor storage')


def _build_engine_pair(torch, cfg, loaded, *, instances, capacity, graphs, log):
    engines, setup = {}, {}
    try:
        for name, role in loaded.items():
            began = time.perf_counter()
            log('actor_cache_build_begin', role=name, instances=instances)
            engines[name] = role['actor_type'](
                role['model'], role['layer_devices'], role['cfgs'], role['devices'], n_inst=instances,
                sampler_backend=cfg.actor_sampler_backend, depth_cache_positions=cfg.actor_depth_cache_positions,
                attention_precision=cfg.actor_attention_precision, linear_backend=cfg.actor_linear_backend,
                max_s=(capacity + 8) / 12.5, p_max=role['prefix_capacity'] + 8)
            engine = engines[name]
            if not role['has_pose']:
                engine.enable_rewind(window=8)
            _sync(torch, role['devices'])
            setup[f'{name}_cache_build_seconds'] = time.perf_counter() - began
            if graphs:
                began = time.perf_counter()
                log('actor_capture_begin', role=name, instances=instances)
                engine.capture(f=25, P=role['prefix_capacity'])
                _sync(torch, role['devices'])
                setup[f'{name}_capture_seconds'] = time.perf_counter() - began
            log('actor_ready', role=name, instances=instances, memory=memory_report(torch, role['devices']))
    except Exception:
        engine = None
        _release_engines(torch, engines, loaded, tuple(device for role in loaded.values() for device in role['devices']))
        raise
    return engines, setup


def main(argv=None):
    args = arguments(argv)
    from .config import TrainingConfig
    from .data import ProductionRows, prefix_seed
    from .train import checkpoint_spec, rollout_seed, source_digest
    cfg = TrainingConfig.from_dict(json.loads(args.config.read_text()))
    if args.delivery_delays is None:
        args.delivery_delays = [cfg.partner_audio_delay_samples]
    if cfg.rollout_mode != 'teacher_partner':
        raise ValueError('Benchmark config must explicitly select teacher_partner rollouts')
    if args.frames <= cfg.partner_prefill_max_frames:
        raise ValueError('Benchmark length must exceed the largest random prefill')
    if args.seed is not None:
        cfg = replace(cfg, seed=args.seed).validate()
    capacity = args.capacity_frames or cfg.conversation_prefix_frames or args.frames
    if capacity < args.frames:
        raise ValueError('Cache capacity cannot be shorter than the benchmark')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'result.json').exists():
        raise ValueError('Use a fresh benchmark output directory')

    def log(event, **fields):
        record = {'event': event, 'wall_time': time.time(), **fields}
        line = json.dumps(record, default=str, allow_nan=False)
        print(line, flush=True)
        with (args.output / 'events.jsonl').open('a') as stream:
            stream.write(line + '\n')

    began = time.perf_counter()
    identity = {'format': 'pose-opd-teacher-partner-benchmark-v1', 'host': socket.gethostname(),
        'source_sha256': source_digest(), 'config': cfg.to_dict(), 'arguments': vars(args),
        'cache_capacity_frames': capacity, 'benchmark_only': True,
        'resident_models': ['student_actor', 'frozen_partner_actor', 'two_voxtral_frontends', 'mimi_decoders'],
        'learner_optimizer_or_kl_scorers_resident': False,
        'memory_scope': 'rollout_only_not_training_update_fit',
        'timing_scope': 'local_four_gpu_conversation_batch_not_global_update'}
    _write_json(args.output / 'identity.json', identity)
    log('data_begin')
    maximum_batch = max(args.batch_sizes)
    rows, selected, eligible_rows, reused_selection = _prepared_benchmark_rows(cfg, args, maximum_batch, log)
    if len(selected) != maximum_batch:
        raise ValueError('Selected dataset range does not contain a full benchmark batch')
    if any(int(row['conversation_frames']) != args.frames for row in rows):
        raise ValueError('Requested benchmark frames exceed the selected fixed-length source rows')
    preparation = {'seconds': time.perf_counter() - began, 'eligible_rows': eligible_rows,
                   'reused_admitted_selection': reused_selection,
                   'selected_items': selected, 'prefill_frames': [int(row['prefill_frames']) for row in rows]}
    log('data_ready', **preparation)
    if args.prepare_only:
        _write_json(args.output / 'result.json', {'identity': identity, 'preparation': preparation,
                                                'gpu_benchmark_run': False})
        return

    import torch
    from . import runtime as R
    from .rollout import make_fastppm_class
    from .train import load_actor_native
    from .partner_rollout import rollout_teacher_partner
    student_devices = tuple(torch.device(f'cuda:{i}') for i in args.student_devices)
    partner_devices = tuple(torch.device(f'cuda:{i}') for i in args.partner_devices)
    devices = student_devices + partner_devices
    if not torch.cuda.is_available() or torch.cuda.device_count() <= max(args.student_devices + args.partner_devices):
        raise ValueError('Four requested CUDA devices must be visible')
    # Deliberately retained until benchmark exit; this is a conservative byte
    # reservation, not a claim that a distributed training update was tested.
    reservations = [torch.empty(int(np.ceil(args.reserve_gib * 2 ** 30)), dtype=torch.uint8, device=device)
                    for device in devices] if args.reserve_gib else []
    torch.manual_seed(cfg.seed)
    M = R.load_native()
    M.QWEN32, M.QWEN4B_CFG = args.qwen32_config, args.qwen4_config
    native = load_actor_native(M)
    specs = {
        'student': checkpoint_spec(cfg.student, 'student'),
        'partner': checkpoint_spec(cfg.resolved_partner_checkpoint, 'audio_teacher', gain_policy=cfg.audio_teacher_gain_policy),
    }
    max_prefix = max(int(row['P'][0]) for row in rows)
    partner_prefix = max(int(row['partner']['P'][0]) for row in rows)
    if cfg.fixed_sequence_frames is not None and cfg.conversation_prefix_frames is not None:
        prefix_capacity = cfg.fixed_sequence_frames - cfg.conversation_prefix_frames - cfg.speech_delay
        max_prefix, partner_prefix = max(max_prefix, prefix_capacity), max(partner_prefix, prefix_capacity)
    loaded, setup = {}, {}
    for role, role_devices, has_pose, prefix in (
            ('student', student_devices, True, max_prefix), ('partner', partner_devices, False, partner_prefix)):
        t0 = time.perf_counter()
        log('model_build_begin', role=role, devices=role_devices)
        override = args.student_backbone_cut if has_pose else args.partner_backbone_cut
        model, layer_devices, cfgs = _build_model(torch, M, specs[role], role_devices,
            backbone_cut=(cfg.actor_backbone_cut if has_pose else cfg.partner_actor_backbone_cut or cfg.actor_backbone_cut)
                if override is None else override,
            require_pose=has_pose, log=log)
        if role == 'partner' and args.pack_partner_projections != 'none':
            from .partner_block import pack_block_projections
            log('partner_projections_packed', **pack_block_projections(model,
                attention=args.pack_partner_projections in ('qkv', 'all'),
                mlp=args.pack_partner_projections in ('mlp', 'all')))
        _sync(torch, role_devices)
        setup[f'{role}_build_load_seconds'] = time.perf_counter() - t0
        loaded[role] = {'model': model, 'layer_devices': layer_devices, 'cfgs': cfgs,
                        'devices': role_devices, 'has_pose': has_pose, 'prefix_capacity': prefix,
                        'actor_type': make_fastppm_class(native, require_pose=has_pose,
                            partner_sampling='natural' if has_pose else cfg.partner_sampling),
                        'initial_hook_ids': set(model.inp._forward_hooks),
                        'storage_identity': _storage_identity(model)}
        log('model_ready', role=role, memory=memory_report(torch, role_devices))
    from voxtral_front import VoxtralFront
    t0 = time.perf_counter()
    log('frontends_build_begin')
    fronts = {role: VoxtralFront(args.voxtral, device=description['devices'][0])
              for role, description in loaded.items()}
    _sync(torch, devices)
    setup['frontend_build_seconds'] = time.perf_counter() - t0
    log('frontends_ready', memory=memory_report(torch, devices))
    runs, candidates, batch_setups, failures = [], [], {}, []
    engines = {}
    verified_candidates = set()
    try:
        for instances in args.batch_sizes:
            candidate_rows = rows[:instances]
            try:
                engines, batch_setup = _build_engine_pair(torch, cfg, loaded, instances=instances,
                    capacity=capacity, graphs=not args.no_graphs, log=log)
                if args.disable_partner_text_history:
                    from .partner_faststate import disable_partner_text_history
                    log('partner_text_history_disabled', **disable_partner_text_history(engines['partner']))
                batch_setups[str(instances)] = batch_setup
                for backend, delivery_delay in itertools.product(args.replay_backends, args.delivery_delays):
                    candidate_id = f'batch-{instances:03d}-{backend}-delay{delivery_delay:03d}'
                    candidate_dir = args.output / 'candidates' / candidate_id
                    candidate_dir.mkdir(parents=True, exist_ok=True)
                    candidate = {'id': candidate_id, 'instances': instances, 'teacher_replay': backend,
                        'delivery_delay_samples': delivery_delay,
                        'cache_capacity_frames': capacity, 'setup': batch_setup, 'runs': [], 'status': 'running'}
                    candidates.append(candidate)
                    for repeat in range(args.repeats):
                        _sync(torch, devices)
                        for device in devices:
                            torch.cuda.reset_peak_memory_stats(device)
                        recorder = CudaPhaseRecorder(torch, calls=args.profile_calls,
                            warmup=cfg.partner_prefill_max_frames + 8 + args.profile_warmup)
                        recorder.attach(engines['student'], 'student', student_devices)
                        recorder.attach(engines['partner'], 'partner', partner_devices)
                        kernel_recorder = KernelGraphRecorder(torch, calls=args.profile_calls,
                            warmup=cfg.partner_prefill_max_frames + 8 + args.profile_warmup)
                        if args.profile_kernel_graphs:
                            for role, engine in engines.items():
                                kernel_recorder.attach(engine, role)
                        # Same row and repeat seeds across candidate sizes/backends.
                        seeds = [rollout_seed(cfg.seed, repeat, 0, slot) for slot in range(instances)]
                        if args.rollout_seeds is not None:
                            seeds = json.loads(args.rollout_seeds.read_text())[args.row_offset:args.row_offset + instances]
                            if len(seeds) != instances or any(type(seed) is not int or not 0 <= seed < 2 ** 32 for seed in seeds):
                                raise ValueError('Explicit rollout seeds must cover every selected row with uint32 seeds')
                        verify_this_run = args.verify_block and candidate_id not in verified_candidates
                        log('rollout_begin', candidate=candidate_id, repeat=repeat, cold=repeat == 0,
                            frames=args.frames, instances=instances, teacher_replay=backend,
                            verify_block=verify_this_run, delivery_delay_samples=delivery_delay)
                        started = time.perf_counter()
                        try:
                            with torch.inference_mode():
                                result = rollout_teacher_partner(engines['student'], engines['partner'], candidate_rows, seeds,
                                    fronts['student'], fronts['partner'], args.mimi, graphs=not args.no_graphs,
                                    progress=lambda **fields: log('progress', candidate=candidate_id, repeat=repeat, **fields),
                                    teacher_replay=backend, verify_block=verify_this_run,
                                    profile_calls=args.profile_calls, profile_warmup=args.profile_warmup,
                                    mimi_placement=args.mimi_placement, delivery_delay_samples=delivery_delay)
                            _sync(torch, devices)
                            elapsed = time.perf_counter() - started
                        finally:
                            recorder.detach()
                            if args.profile_kernel_graphs:
                                kernel_recorder.detach(engines.values())
                        verified_candidates.add(candidate_id)
                        audit = _audit_result(result, candidate_rows)
                        if args.save_conversations:
                            from .trial_audio import save_trial_audio
                            save_trial_audio(result, candidate_rows, seeds, cfg,
                                candidate_dir / f'conversations-{repeat}', row_offset=args.row_offset)
                        generated = sum(args.frames - int(row['prefill_frames']) for row in candidate_rows)
                        run = {'candidate': candidate_id, 'instances': instances, 'teacher_replay': backend,
                            'delivery_delay_samples': delivery_delay,
                            'repeat': repeat, 'cold': repeat == 0, 'wall_seconds': elapsed,
                            'block_verification_included': verify_this_run,
                            'student_generated_audio_frames': generated,
                            'student_audio_frames_per_wall_second': generated / elapsed,
                            'student_generated_audio_seconds_per_wall_second': generated / 12.5 / elapsed,
                            'runtime_timings': result.get('timings', {}), 'cuda_actor_phases': recorder.report(),
                            'cuda_kernel_graphs': kernel_recorder.report(),
                            'memory': memory_report(torch, devices), 'audit': audit}
                        runs.append(run)
                        candidate['runs'].append(run)
                        log('rollout_complete', **run)
                        _write_json(args.output / 'partial.json', {'identity': identity, 'setup': setup,
                            'batch_setups': batch_setups, 'candidates': candidates, 'runs': runs, 'failures': failures})
                        _write_json(candidate_dir / 'partial.json', candidate)
                        del result, recorder
                    candidate['status'] = 'complete'
                    candidate['warm_wall_seconds_median'] = (float(np.median([
                        run['wall_seconds'] for run in candidate['runs'][1:]])) if args.repeats > 1 else None)
                    _write_json(candidate_dir / 'result.json', candidate)
                    log('candidate_complete', candidate=candidate_id,
                        warm_wall_seconds_median=candidate['warm_wall_seconds_median'])
            except torch.cuda.OutOfMemoryError as exc:
                failure = {'instances': instances, 'error': str(exc), 'type': 'cuda_out_of_memory',
                           'memory': memory_report(torch, devices)}
                failures.append(failure)
                if candidates and candidates[-1]['instances'] == instances and candidates[-1]['status'] == 'running':
                    candidates[-1]['status'] = 'out_of_memory'
                    candidates[-1]['failure'] = failure
                log('candidate_out_of_memory', **failure)
                _write_json(args.output / f'oom-batch-{instances:03d}.json', failure)
            finally:
                result = recorder = None
                _release_engines(torch, engines, loaded, devices)
                log('actor_caches_released', instances=instances, bare_weights_retained=True,
                    memory=memory_report(torch, devices))
    except Exception as exc:
        _write_json(args.output / 'failure.json', {'error': f'{type(exc).__name__}: {exc}',
                    'traceback': traceback.format_exc(), 'setup': setup, 'runs': runs,
                    'batch_setups': batch_setups, 'candidates': candidates, 'failures': failures,
                    'memory': memory_report(torch, devices)})
        raise
    summary = {'identity': identity, 'preparation': preparation, 'setup': setup, 'runs': runs,
        'batch_setups': batch_setups, 'candidates': candidates, 'failures': failures,
        'gpu_benchmark_run': True, 'all_candidates_succeeded': not failures,
        'warm_wall_seconds_median': candidates[0].get('warm_wall_seconds_median') if len(candidates) == 1 else None,
        'full_training_update_seconds': None,
        'full_training_update_memory_validated': False,
        'extrapolation_note': 'Short runs do not establish full-length attention cost or distributed scheduling throughput.'}
    _write_json(args.output / 'result.json', summary)
    log('benchmark_complete', output=str(args.output / 'result.json'))


if __name__ == '__main__':
    main()
