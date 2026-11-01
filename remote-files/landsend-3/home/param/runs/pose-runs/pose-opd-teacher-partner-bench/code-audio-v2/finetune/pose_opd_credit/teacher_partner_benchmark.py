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
import hashlib
import json
from pathlib import Path
import socket
import time
import traceback

import numpy as np


def arguments(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--instances', type=int, default=16)
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
    parser.add_argument('--teacher-replay', choices=('sequential', 'block'), default='sequential')
    parser.add_argument('--verify-block', action='store_true')
    parser.add_argument('--no-graphs', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--qwen32-config', default='/home/param/models/imported-weights/ppm-811-17850-base-bf16/configs/Qwen3-32B')
    parser.add_argument('--qwen4-config', default='/home/param/models/imported-weights/ppm-811-17850-base-bf16/configs/Qwen3-4B')
    parser.add_argument('--voxtral', default='/home/param/models/imported-weights/aloe-ft-vox17127-nopersona/assets/voxtral-realtime')
    parser.add_argument('--mimi', default='/home/param/models/mimi')
    args = parser.parse_args(argv)
    if not 1 <= args.instances <= 64:
        parser.error('--instances must be between1 and64')
    if args.frames < 1 or args.repeats < 1 or args.row_offset < 0 or args.profile_calls < 0:
        parser.error('Positive frames/repeats and nonnegative offset/profile counts are required')
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
    def __init__(self, torch, *, calls=32):
        self.torch, self.limit = torch, int(calls)
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
                capture = len(self.events[label]) < self.limit
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
    return {'rows': len(rows), 'rollout_sha256': fingerprints,
            'prefill_frames': [int(row['prefill_frames']) for row in rows]}


def main(argv=None):
    args = arguments(argv)
    from .config import TrainingConfig
    from .data import ProductionRows, prefix_seed
    from .train import checkpoint_spec, rollout_seed, source_digest
    cfg = TrainingConfig.from_dict(json.loads(args.config.read_text()))
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
    data = ProductionRows(cfg)
    items = data.items()
    selected = items[args.row_offset:args.row_offset + args.instances]
    if len(selected) != args.instances:
        raise ValueError('Selected dataset range does not contain a full benchmark batch')
    rows = [prefix_seed(data.load(item, rank=0, update=0, slot=slot), args.frames)
            for slot, item in enumerate(selected)]
    preparation = {'seconds': time.perf_counter() - began, 'eligible_rows': len(items),
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
    engines, setup = {}, {}
    for role, role_devices, has_pose, prefix in (
            ('student', student_devices, True, max_prefix), ('partner', partner_devices, False, partner_prefix)):
        t0 = time.perf_counter()
        log('actor_build_begin', role=role, instances=args.instances, devices=role_devices)
        override = args.student_backbone_cut if has_pose else args.partner_backbone_cut
        model, layer_devices, cfgs = _build_model(torch, M, specs[role], role_devices,
            backbone_cut=cfg.actor_backbone_cut if override is None else override,
            require_pose=has_pose, log=log)
        actor_type = make_fastppm_class(native, require_pose=has_pose)
        engine = actor_type(model, layer_devices, cfgs, role_devices, n_inst=args.instances,
            sampler_backend=cfg.actor_sampler_backend, depth_cache_positions=cfg.actor_depth_cache_positions,
            attention_precision=cfg.actor_attention_precision, linear_backend=cfg.actor_linear_backend,
            max_s=(capacity + 8) / 12.5, p_max=prefix + 8)
        if not has_pose:
            engine.enable_rewind(window=8)
        _sync(torch, role_devices)
        setup[f'{role}_build_load_seconds'] = time.perf_counter() - t0
        if not args.no_graphs:
            t0 = time.perf_counter()
            log('actor_capture_begin', role=role)
            engine.capture(f=25, P=prefix)
            _sync(torch, role_devices)
            setup[f'{role}_capture_seconds'] = time.perf_counter() - t0
        engines[role] = engine
        log('actor_ready', role=role, memory=memory_report(torch, role_devices))
    from voxtral_front import VoxtralFront
    t0 = time.perf_counter()
    log('frontends_build_begin')
    fronts = {role: VoxtralFront(args.voxtral, device=engine.d0)
              for role, engine in engines.items()}
    _sync(torch, devices)
    setup['frontend_build_seconds'] = time.perf_counter() - t0
    log('frontends_ready', memory=memory_report(torch, devices))
    runs = []
    try:
        for repeat in range(args.repeats):
            _sync(torch, devices)
            for device in devices:
                torch.cuda.reset_peak_memory_stats(device)
            recorder = CudaPhaseRecorder(torch, calls=args.profile_calls)
            recorder.attach(engines['student'], 'student', student_devices)
            recorder.attach(engines['partner'], 'partner', partner_devices)
            seeds = [rollout_seed(cfg.seed, repeat, 0, slot) for slot in range(args.instances)]
            log('rollout_begin', repeat=repeat, cold=repeat == 0, frames=args.frames, instances=args.instances)
            started = time.perf_counter()
            try:
                with torch.inference_mode():
                    result = rollout_teacher_partner(engines['student'], engines['partner'], rows, seeds,
                        fronts['student'], fronts['partner'], args.mimi, graphs=not args.no_graphs,
                        progress=lambda **fields: log('progress', repeat=repeat, **fields),
                        teacher_replay=args.teacher_replay, verify_block=args.verify_block)
                _sync(torch, devices)
                elapsed = time.perf_counter() - started
            finally:
                recorder.detach()
            audit = _audit_result(result, rows)
            generated = sum(args.frames - int(row['prefill_frames']) for row in rows)
            run = {'repeat': repeat, 'cold': repeat == 0, 'wall_seconds': elapsed,
                'student_generated_audio_frames': generated,
                'student_audio_frames_per_wall_second': generated / elapsed,
                'student_generated_audio_seconds_per_wall_second': generated / 12.5 / elapsed,
                'runtime_timings': result.get('timings', {}), 'cuda_actor_phases': recorder.report(),
                'memory': memory_report(torch, devices), 'audit': audit}
            runs.append(run)
            log('rollout_complete', **run)
            _write_json(args.output / 'partial.json', {'identity': identity, 'setup': setup, 'runs': runs})
            del result, recorder
    except Exception as exc:
        _write_json(args.output / 'failure.json', {'error': f'{type(exc).__name__}: {exc}',
                    'traceback': traceback.format_exc(), 'setup': setup, 'runs': runs,
                    'memory': memory_report(torch, devices)})
        raise
    summary = {'identity': identity, 'preparation': preparation, 'setup': setup, 'runs': runs,
        'gpu_benchmark_run': True,
        'warm_wall_seconds_median': float(np.median([run['wall_seconds'] for run in runs[1:]])) if len(runs) > 1 else None,
        'full_training_update_seconds': None,
        'full_training_update_memory_validated': False,
        'extrapolation_note': 'Short runs do not establish full-length attention cost or distributed scheduling throughput.'}
    _write_json(args.output / 'result.json', summary)
    log('benchmark_complete', output=str(args.output / 'result.json'))


if __name__ == '__main__':
    main()
