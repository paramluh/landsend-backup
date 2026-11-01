"""Two-teacher full-row OPD: current student rollout, analytic KL, finite credit.

Launch with torchrun, an even number of visible GPUs per host, and an explicit
JSON TrainingConfig. --prepare-only validates dataset membership without CUDA.
No training processes are submitted to other machines by this entry point.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import replace
import datetime
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import sys
import time

import numpy as np

from .config import TrainingConfig
from .data import ProductionRows, replay_from_rollout
from .schedule import resolve_schedule, apply_learning_rate

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def canonical_json(value):
    """The same representation before and after checkpoint JSON round-trips."""
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def checkpoint_deadline(config, now):
    return None if config.checkpoint_interval_seconds is None else now + config.checkpoint_interval_seconds


def checkpoint_due(config, *, completed_update, limit, now, deadline):
    """Step intervals use the absolute resume cursor; final updates always save."""
    return (completed_update == limit
            or (config.checkpoint_interval_steps is not None and completed_update > 0
                and completed_update % config.checkpoint_interval_steps == 0)
            or (deadline is not None and now >= deadline))


def checkpoint_export_config(config, student_spec, *, identity, data_config, next_update, plan_digest):
    """Preserve reconstruction metadata and the configured resume-state retention."""
    return dict(student_spec['config'], opd_identity=identity,
                keep_latest_optimizer_state=config.keep_latest_optimizer_state,
                vadapter=student_spec['vadapter'], decp_bb_alpha=student_spec['decp_bb_alpha'],
                data=data_config,
                **{name: student_spec[name] for name in
                   ('layers', 'tower_layers', 'dec_layers', 'proj_r', 'tower_vocab')},
                _runtime_state={'next_update': next_update, 'plan_digest': plan_digest})


def source_digest(repo=REPO):
    """Hash all local Python/config dependencies, including checkpoint helpers."""
    repo = Path(repo)
    digest = hashlib.sha256()
    paths = []
    for root in (repo / 'finetune' / 'pose_opd_credit', repo / 'finetune' / 'duplex_distill',
                 repo / 'infer' / 'engine' / 'ppm'):
        paths.extend(p for p in root.rglob('*') if p.is_file() and p.suffix in ('.py', '.json'))
    paths.append(repo / 'finetune' / 'storage_paths.py')
    for path in sorted(set(paths)):
        name = path.relative_to(repo).as_posix().encode()
        content = path.read_bytes()
        digest.update(len(name).to_bytes(8, 'big')); digest.update(name)
        digest.update(len(content).to_bytes(8, 'big')); digest.update(content)
    return digest.hexdigest()


def make_plan(items, *, world, epochs, seed, rows_per_rank=1, frame_lengths=None):
    """Full-row coverage, independently shuffled epochs, explicit zero-weight tail slots."""
    if (not items or type(world) is not int or world < 1 or type(epochs) is not int or epochs < 1
            or type(seed) is not int or seed < 0 or type(rows_per_rank) is not int or rows_per_rank < 1):
        raise ValueError('Nonempty data and positive world/epochs are required')
    normalized = [tuple(map(int, item)) for item in items]
    if any(len(item) != 3 for item in normalized) or len(set(normalized)) != len(normalized):
        raise ValueError('Data plan requires unique (store, row, orientation) items')
    if frame_lengths is not None:
        if set(frame_lengths) != set(normalized) or any(type(value) is not int or value <= 0 for value in frame_lengths.values()):
            raise ValueError('Balancing requires one positive frame length for every selected orientation')
    updates = []
    batch = world * rows_per_rank
    for epoch in range(epochs):
        order = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(len(items))
        rows = [list(normalized[i]) for i in order]
        rows += [None] * ((-len(rows)) % batch)
        for start in range(0, len(rows), batch):
            update = rows[start:start + batch]
            if frame_lengths is not None:
                # Preserve the shuffled global batch exactly; only distribute
                # its rows across ranks. Each rank still has the same slot count.
                groups, loads = [[] for _ in range(world)], [0] * world
                length = lambda row: 0 if row is None else frame_lengths[tuple(row)]
                for row in sorted(update, key=length, reverse=True):
                    rank = min((rank for rank in range(world) if len(groups[rank]) < rows_per_rank),
                               key=lambda rank: (loads[rank], len(groups[rank]), rank))
                    groups[rank].append(row)
                    loads[rank] += length(row)
                update = [row for group in groups for row in group]
            updates.append(update)
    digest = hashlib.sha256(json.dumps(updates, separators=(',', ':')).encode()).hexdigest()
    return {'format': 'pose-opd-credit-plan-v1', 'updates': updates, 'digest': digest,
            'rows_per_epoch': len(items), 'epochs': epochs, 'world': world, 'seed': seed,
            'rows_per_rank': rows_per_rank, 'effective_batch': batch,
            'rank_assignment': 'balanced_frames' if frame_lengths is not None else 'shuffled'}


def rollout_seed(base_seed, absolute_update, rank, slot):
    return int(np.random.SeedSequence([base_seed, absolute_update, rank, slot]).generate_state(1)[0])


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)


def _file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_spec(path, role, *, gain_policy='require_fp32'):
    """Resolve each model's own adapter, dimensions and effective D33 scale.

    Export-side explicit alpha is already resolved. Older local exports keep
    the initial alpha/fade schedule in cfg, which must be evaluated at their
    checkpoint step. A teacher must never inherit the student's scale.
    """
    path = Path(path)
    if role not in ('student', 'audio_teacher', 'pose_teacher'):
        raise ValueError(f'Unknown model role: {role}')
    if gain_policy not in ('require_fp32', 'exported') or (role != 'audio_teacher' and gain_policy != 'require_fp32'):
        raise ValueError('Only the audio teacher may explicitly retain its exported gain')
    metadata_path = path / 'train_cfg.json'
    if not metadata_path.is_file():
        metadata_path = path / 'meta.json'
    if not metadata_path.is_file() or not (path / 'COMPLETE').is_file():
        raise ValueError(f'{role}: expected a complete export with train_cfg.json or meta.json at {path}')
    metadata = json.loads(metadata_path.read_text())
    cfg = dict(metadata.get('cfg', metadata))
    if cfg.get('mot'):
        raise ValueError(f'{role}: this entrypoint requires a native PPM checkpoint; MoT needs its own teacher adapter')
    step = int(metadata.get('step', -1))
    alpha = float(metadata.get('decp_bb_alpha', cfg.get('decp_bb_alpha', 1.)))
    if 'decp_bb_alpha' not in metadata:
        fade_from, fade_steps = int(cfg.get('decp_bb_fade_from', -1)), int(cfg.get('decp_bb_fade_steps', 0))
        if fade_from >= 0 and fade_steps > 0:
            if step < 0:
                raise ValueError(f'{role}: a faded D33 link needs the checkpoint step')
            alpha *= min(1., max(0., 1. - (step - fade_from) / fade_steps))
    if not math.isfinite(alpha):
        raise ValueError(f'{role}: D33 alpha must be finite')
    vadapter = cfg.get('vadapter')
    if role == 'pose_teacher':
        if vadapter:
            raise ValueError('The pose monoplex teacher must have native empty A and no audio adapter')
        vadapter = None
    elif not isinstance(vadapter, dict) or not vadapter:
        raise ValueError(f'{role}: checkpoint metadata must describe its trained Voxtral adapter')
    dims = {'layers': int(cfg.get('layers', 64)), 'tower_layers': int(cfg.get('tower_layers', 36)),
            'dec_layers': int(cfg.get('dec_layers', 28)), 'proj_r': int(cfg.get('proj_r', 2048)),
            'tower_vocab': int(cfg.get('tower_vocab', 151682))}
    if any(value < 1 for value in dims.values()) or dims['tower_vocab'] != 151682:
        raise ValueError(f'{role}: incompatible architecture dimensions or executable vocabulary')
    if role != 'audio_teacher' and metadata.get('decp_bb', cfg.get('decp_bb_link', 1)) == 0:
        raise ValueError(f'{role}: the replay implementation requires the native D33 pose link')
    fingerprints = {metadata_path.name: _file_digest(metadata_path)}
    gain = None
    if vadapter:
        gain_path = path / 'vgain_theta_fp32.json'
        if gain_path.is_file():
            gain = float(json.loads(gain_path.read_text())['vgain_theta_fp32'])
            fingerprints[gain_path.name] = _file_digest(gain_path)
        elif 'vgain_theta_fp32' in metadata:
            gain = float(metadata['vgain_theta_fp32'])
        elif gain_policy != 'exported':
            raise ValueError(f'{role}: {path} lacks exact fp32 audio gain metadata; use a verified export/sidecar')
        if gain is not None and not math.isfinite(gain):
            raise ValueError(f'{role}: audio input gain must be finite')
    if (path / '.metadata').is_file():
        fingerprints['.metadata'] = _file_digest(path / '.metadata')
    return canonical_json({'path': str(path), 'role': role, 'step': step, 'config': cfg,
                           'vadapter': vadapter, 'decp_bb_alpha': alpha, **dims,
                           'vgain_theta_fp32': gain,
                           'gain_policy': gain_policy,
                           'pose_head': bool(cfg.get('pose_head_build', False)),
                           'metadata_sha256': fingerprints})


def validate_resume(saved, identity, update_count, hosts=None):
    if canonical_json(saved.get('cfg', {}).get('opd_identity')) != canonical_json(identity):
        raise ValueError('Resume changes the objective, source, sampling policy, model metadata or data schedule')
    step = saved.get('step')
    if type(step) is not int or not 0 <= step <= update_count:
        raise ValueError('Resume next-update index lies outside the data plan')
    state = saved.get('runtime_state', saved.get('cfg', {}).get('_runtime_state', {}))
    if state.get('next_update') != step or state.get('plan_digest') != identity['plan_digest']:
        raise ValueError('Resume data cursor and plan do not match the checkpoint update')
    if hosts is not None and (saved.get('world') != len(hosts) or saved.get('hosts') != hosts):
        raise ValueError('Resume requires the exact saved rank-to-host mapping')
    return step


def write_training_metadata(root, identity, plan, audit, *, resume=False):
    """Refuse to overwrite another run or silently append fresh-weight updates."""
    root = Path(root)
    current = root / 'config.json'
    if current.is_file() and canonical_json(json.loads(current.read_text())) != canonical_json(identity):
        raise ValueError(f'Run directory already contains a different training identity: {root}')
    metrics = root / 'metrics.jsonl'
    if not resume and metrics.is_file() and metrics.stat().st_size:
        raise ValueError('This run has recorded updates; use its explicit --resume checkpoint or a fresh run directory')
    atomic_json(current, canonical_json(identity))
    atomic_json(root / 'plan.json', plan)
    atomic_json(root / 'data-selection.json', {'eligible_rows': plan['rows_per_epoch'], 'excluded_rows': audit})


def load_actor_native(model_module, repo=REPO):
    """Import the real cached engine while sharing the learner's model class."""
    import importlib
    engine_dir = Path(repo) / 'infer' / 'engine' / 'ppm'
    if sys.modules.get('model_ppm') is not model_module:
        raise RuntimeError('Actor must share the loaded production model_ppm module')
    existing = sys.modules.get('fast_ppm')
    if existing is not None and Path(existing.__file__).resolve() != (engine_dir / 'fast_ppm.py').resolve():
        raise RuntimeError('Another FastPPM engine module is already loaded')
    sys.path.insert(0, str(engine_dir))
    try:
        module = importlib.import_module('fast_ppm')
    finally:
        sys.path.remove(str(engine_dir))
    if module.M is not model_module or module.IP.M is not model_module:
        raise RuntimeError('Actor and replay imported incompatible production model classes')
    return module


def args_parser(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--max-updates', type=int, help='Bound a smoke/profile run without changing its data plan')
    parser.add_argument('--no-save', action='store_true', help='Bounded smoke/profile only; no model/optimizer saves')
    parser.add_argument('--benchmark-microbatches', help='Physical batch per bounded update, e.g. 2,4; requires --no-save')
    parser.add_argument('--benchmark-replay-microbatches',
                        help='Replay one fixed generated batch at each physical size, e.g. 1,2,4; unsaved single update only')
    parser.add_argument('--benchmark-replay-variants',
                        help='JSON list of learner batch/chunk/sync/checkpoint overrides on one fixed batch')
    parser.add_argument('--benchmark-shape-only', action='store_true',
                        help='Synthetic maximum-shape stress: cycle forced prefill tokens, with no actor or policy claims')
    cache = parser.add_mutually_exclusive_group()
    cache.add_argument('--benchmark-save-rollouts', type=Path, help='Save completed CPU trajectories for benchmark reuse')
    cache.add_argument('--benchmark-load-rollouts', type=Path, help='Reuse verified CPU trajectories, skipping the audio front and actor')
    parser.add_argument('--warm-optimizer', action='store_true',
                        help='Benchmark only: allocate AdamW state with a zero-gradient step before rollout')
    parser.add_argument('--no-graphs', action='store_true', help='Eager cached actor for diagnostics')
    parser.add_argument('--profile-actor-graphs', action='store_true',
                        help='Time the first 32 captured inference steps with CUDA events')
    parser.add_argument('--profile-replay-stages', action='store_true',
                        help='Record replay host spans and current-stream CUDA event timings')
    parser.add_argument('--profile-policy-agreement', action='store_true',
                        help='Report bounded per-stream sampled likelihood and categorical actor/replay agreement')
    parser.add_argument('--policy-agreement-frames', type=int,
                        help='First sampled frames per stream/row for likelihood diagnostics (default 128)')
    parser.add_argument('--benchmark-prefix-frames', type=int,
                        help='Explicitly truncate rows for a bounded unsaved diagnostic; never a production config')
    parser.add_argument('--qwen32-config', default='/home/param/models/imported-weights/ppm-811-17850-base-bf16/configs/Qwen3-32B')
    parser.add_argument('--qwen4-config', default='/home/param/models/imported-weights/ppm-811-17850-base-bf16/configs/Qwen3-4B')
    parser.add_argument('--voxtral', default='/home/param/models/imported-weights/aloe-ft-vox17127-nopersona/assets/voxtral-realtime')
    args = parser.parse_args(argv)
    if args.max_updates is not None and args.max_updates < 1:
        parser.error('--max-updates must be positive')
    if args.no_save and args.max_updates is None:
        parser.error('--no-save requires a bounded --max-updates')
    if args.warm_optimizer and (not args.no_save or args.max_updates is None or args.resume is not None):
        parser.error('--warm-optimizer requires bounded --no-save and cannot resume a checkpoint')
    if args.prepare_only and (args.resume is not None or args.no_save or args.max_updates is not None):
        parser.error('--prepare-only cannot be combined with resume/profile execution flags')
    if args.policy_agreement_frames is not None and (not args.profile_policy_agreement or args.policy_agreement_frames <= 0):
        parser.error('--policy-agreement-frames requires --profile-policy-agreement and a positive count')
    if (args.profile_policy_agreement or args.profile_replay_stages) and args.prepare_only:
        parser.error('Replay/policy profiles require model execution')
    if args.profile_policy_agreement and args.policy_agreement_frames is None:
        args.policy_agreement_frames = 128
    if args.benchmark_prefix_frames is not None:
        if (not args.no_save or args.max_updates is None or args.resume is not None
                or args.benchmark_prefix_frames <= 0):
            parser.error('--benchmark-prefix-frames requires bounded --no-save, no resume, and a positive count')
    if args.benchmark_microbatches is not None:
        if not args.no_save or args.max_updates is None:
            parser.error('--benchmark-microbatches requires bounded --max-updates and --no-save')
        try:
            args.benchmark_microbatches = [int(value) for value in args.benchmark_microbatches.split(',')]
        except ValueError:
            parser.error('--benchmark-microbatches must be comma-separated positive integers')
        if len(args.benchmark_microbatches) != args.max_updates or any(value <= 0 for value in args.benchmark_microbatches):
            parser.error('--benchmark-microbatches needs one positive batch size per --max-updates')
    reuse = (args.benchmark_replay_microbatches is not None or args.benchmark_replay_variants is not None
             or args.benchmark_shape_only or args.benchmark_save_rollouts or args.benchmark_load_rollouts)
    if reuse and (not args.no_save or args.max_updates != 1 or args.resume is not None):
        parser.error('Replay batch comparisons and rollout caches require --no-save --max-updates 1 and no resume')
    if args.benchmark_replay_microbatches is not None:
        if args.benchmark_microbatches is not None:
            parser.error('Choose either per-update microbatches or fixed-trajectory replay microbatches')
        try:
            args.benchmark_replay_microbatches = [int(value) for value in args.benchmark_replay_microbatches.split(',')]
        except ValueError:
            parser.error('--benchmark-replay-microbatches must contain positive integers')
        if any(value <= 0 for value in args.benchmark_replay_microbatches):
            parser.error('--benchmark-replay-microbatches must contain positive integers')
    if args.benchmark_replay_variants is not None:
        from .benchmark import REPLAY_VARIANT_FIELDS
        if args.benchmark_replay_microbatches is not None or args.benchmark_microbatches is not None:
            parser.error('Replay variants cannot be combined with another microbatch schedule')
        try:
            variants = json.loads(args.benchmark_replay_variants)
        except ValueError:
            parser.error('--benchmark-replay-variants must be a JSON list')
        if (not isinstance(variants, list) or not variants or any(not isinstance(value, dict)
                or not value or set(value) - REPLAY_VARIANT_FIELDS for value in variants)):
            parser.error('Replay variants must be nonempty objects containing only learner scheduling fields')
        args.benchmark_replay_variants = variants
    if args.benchmark_shape_only and (not args.warm_optimizer or args.benchmark_prefix_frames is not None
            or args.profile_policy_agreement or args.profile_actor_graphs):
        parser.error('Shape-only benchmarks require warm Adam, full rows, and no actor/policy profiles')
    return args


def validate_benchmark_microbatches(config, schedule):
    if schedule is not None:
        if schedule[0] != config.physical_microbatch:
            raise ValueError('First benchmark microbatch must match configured physical_microbatch')
        if any(config.effective_batch_per_rank % value for value in schedule):
            raise ValueError('Every benchmark microbatch must divide effective_batch_per_rank')


def validate_replay_microbatches(config, schedule):
    if schedule is not None and any(config.effective_batch_per_rank % value for value in schedule):
        raise ValueError('Every replay benchmark microbatch must divide effective_batch_per_rank')


def warm_optimizer_state(optimizer):
    """Allocate fresh AdamW moments without changing weights, for profiling only.

    The zero-gradient, zero-decay step leaves every finite parameter unchanged
    but advances Adam's counter to one. This deliberately is not a production
    training initialization; callers must prohibit saving and resuming.
    """
    import torch
    if not isinstance(optimizer, torch.optim.AdamW):
        raise ValueError('Optimizer warm-up requires AdamW')
    if optimizer.state or any(float(group['weight_decay']) != 0. for group in optimizer.param_groups):
        raise ValueError('Optimizer warm-up requires fresh state and zero weight decay')
    parameters = [parameter for group in optimizer.param_groups for parameter in group['params']]
    if any(parameter.grad is not None for parameter in parameters):
        raise ValueError('Optimizer warm-up cannot discard existing gradients')
    for parameter in parameters:
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {'optimizer_step': 1, 'parameter_tensors': len(parameters),
            'weight_decay': 0., 'zero_gradient_warmup': True}


def move_audio_front(front, target):
    """Move only the frozen encoder, preserving BF16 weights/FP32 RoPE buffers.

    Wrapper device settings remain on the inference GPU. No encode may run
    while offloaded; restoring before the next data stage restores that device
    contract. Releasing the source allocator is needed by the paired actor,
    which may allocate on this GPU from a different process.
    """
    import torch
    target = torch.device(target)
    started = time.monotonic()
    front.fe.encoder.to(device=target)
    if target.type == 'cpu' and torch.device(front.dev).type == 'cuda':
        with torch.cuda.device(front.dev):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    elif target.type == 'cuda':
        torch.cuda.synchronize(target)
    return time.monotonic() - started


def main(argv=None):
    with ExitStack() as cleanup:
        return _main(argv, cleanup)


def _main(argv, cleanup):
    args = args_parser(argv)
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local = int(os.environ.get('LOCAL_RANK', '0'))
    # Initialize control collectives before local filesystem/data checks. One
    # missing dataset/checkpoint must fail all ranks before model collectives.
    control_group = None
    if not args.prepare_only:
        import torch
        import torch.distributed as dist
        local_world = int(os.environ.get('LOCAL_WORLD_SIZE', '0'))
        if world < 2 or world % 2 or local_world < 2 or local_world % 2 or world % local_world:
            raise ValueError('Training requires even world/local sizes: one cached actor per local GPU pair')
        device = torch.device('cuda', local)
        torch.cuda.set_device(device)
        torch.set_num_threads(1)
        dist.init_process_group('nccl', device_id=device, timeout=datetime.timedelta(minutes=60))
        control_group = dist.new_group(backend='gloo', timeout=datetime.timedelta(minutes=60))

    def agree(error, stage):
        if args.prepare_only:
            errors = [error]
        else:
            errors = [None] * world
            dist.all_gather_object(errors, error, group=control_group)
        if any(errors):
            raise RuntimeError(f'{stage}: ' + '; '.join(e for e in errors if e))

    error = None
    try:
        from finetune.storage_paths import canonical_run_dir
        config = TrainingConfig.from_dict(json.loads(args.config.read_text()))
        validate_benchmark_microbatches(config, args.benchmark_microbatches)
        validate_replay_microbatches(config, args.benchmark_replay_microbatches)
        for changes in args.benchmark_replay_variants or ():
            replace(config, **changes).validate()
        if args.benchmark_prefix_frames is not None and args.benchmark_prefix_frames <= config.prefill_frames + 2:
            raise ValueError('Benchmark prefix must exceed the prefill plus media delay')
        root = canonical_run_dir(args.run_dir)
        root.mkdir(parents=True, exist_ok=True)
        data = ProductionRows(config)
        items = data.items('train')
        plan = make_plan(items, world=world, epochs=config.epochs, seed=config.seed,
                         rows_per_rank=config.effective_batch_per_rank,
                         frame_lengths=data.frame_lengths(items) if config.balance_rows_by_length else None)
        from .shape_benchmark import worst_shape_plan, validate_shape_seed, synthetic_rollout
        if args.benchmark_shape_only:
            plan = worst_shape_plan(data.shape_specs(items), world=world, rows_per_rank=config.effective_batch_per_rank)
        learning_rate_schedule = resolve_schedule(config, len(plan['updates']))
        specs = {role: checkpoint_spec(path, role, gain_policy=config.audio_teacher_gain_policy
                                      if role == 'audio_teacher' else 'require_fp32') for role, path in (
            ('student', config.student), ('audio_teacher', config.audio_teacher), ('pose_teacher', config.pose_teacher))}
        if config.actor_backbone_cut is not None and config.actor_backbone_cut >= specs['student']['layers']:
            raise ValueError('actor_backbone_cut must leave at least one student layer on each actor device')
        assets = {str(Path(path) / 'config.json'): _file_digest(Path(path) / 'config.json')
                  for path in (args.qwen32_config, args.qwen4_config, args.voxtral)}
        identity = canonical_json({'objective': 'two-teacher-student-kl-finite-causal-credit-v1',
                    'method': config.to_dict(), 'source_sha256': source_digest(), 'plan_digest': plan['digest'],
                    'dataset': data.metadata, 'benchmark_microbatches': args.benchmark_microbatches,
                    'benchmark_warm_optimizer': args.warm_optimizer,
                    'benchmark_replay_microbatches': args.benchmark_replay_microbatches,
                    'benchmark_replay_variants': args.benchmark_replay_variants,
                    'benchmark_shape_only': args.benchmark_shape_only,
                    'benchmark_shape_spec': plan.get('shape_spec'),
                    'benchmark_save_rollouts': str(args.benchmark_save_rollouts) if args.benchmark_save_rollouts else None,
                    'benchmark_load_rollouts': str(args.benchmark_load_rollouts) if args.benchmark_load_rollouts else None,
                    'benchmark_prefix_frames': args.benchmark_prefix_frames,
                    'policy_agreement_frames': args.policy_agreement_frames,
                    'profile_replay_stages': args.profile_replay_stages,
                    'world': world, 'actor': 'synthetic-prefill-cycle-no-actor' if args.benchmark_shape_only else 'paired-cached-natural-categorical', 'models': specs,
                    'optimizer': {'type': 'AdamW', 'lr': config.learning_rate, 'schedule': learning_rate_schedule, 'betas': [.9, .95],
                                  'eps': 1e-8, 'weight_decay': 0., 'max_grad_norm': 1.},
                    'model_configs': [args.qwen32_config, args.qwen4_config], 'voxtral': args.voxtral,
                    'asset_config_sha256': assets})
        from .benchmark import cache_identity, load_rollout_rank, save_rollout_rank, commit_rollout_cache, gradient_statistics, replay_candidates
        trajectory_identity = cache_identity(identity, update=0, rank=rank)
        cached_rollouts = (load_rollout_rank(args.benchmark_load_rollouts, trajectory_identity)
                           if args.benchmark_load_rollouts else None)
    except Exception as exc:
        error = f'rank {rank}: {type(exc).__name__}: {exc}'
    agree(error, 'configuration/data/checkpoint preflight')
    if args.prepare_only:
        if rank == 0:
            atomic_json(root / 'preparation.json', {'identity': identity, 'plan': plan,
                        'eligible_rows': len(items), 'excluded_rows': data.audit})
        print(json.dumps({'prepared': True, 'rows': len(items), 'updates': len(plan['updates']), 'run_dir': str(root)}))
        return

    from . import runtime as R
    from .actor import build_actor, publish_actor, release_publication_cache
    from .rollout import make_fastppm_class, audio_features_from_latents, rollout_rows
    from .diagnostics import PolicyAgreement, CategoricalAgreement, benchmark_prefix
    from .profiling import ReplayStages
    from finetune.duplex_distill.checkpoints import load_checkpoint, save_checkpoint, resume_checkpoint

    pair = None
    for first in range(0, world, 2):
        group = dist.new_group([first, first + 1], backend='gloo', timeout=datetime.timedelta(minutes=60))
        if first <= rank < first + 2:
            pair = group
    leader = rank - rank % 2
    hosts = [None] * world
    dist.all_gather_object(hosts, socket.gethostname(), group=control_group)
    if any(hosts[i] != hosts[i + 1] for i in range(0, world, 2)):
        raise ValueError('Every actor GPU pair must reside on the same host')
    identities = [None] * world
    dist.all_gather_object(identities, identity, group=control_group)
    if any(other != identity for other in identities):
        raise ValueError('Source/config/data plan differs across ranks')
    error, start = None, 0
    try:
        if args.resume:
            if not (args.resume / 'COMPLETE').is_file():
                raise ValueError('Resume checkpoint is not committed COMPLETE')
            saved = json.loads((args.resume / 'train_state.json').read_text())
            start = validate_resume(saved, identity, len(plan['updates']), hosts)
        if rank == 0:
            write_training_metadata(root, identity, plan, data.audit, resume=bool(args.resume))
        (root / 'control').mkdir(exist_ok=True)
    except Exception as exc:
        error = f'rank {rank}: {type(exc).__name__}: {exc}'
    agree(error, 'resume/run-directory preflight')

    def log(event, **fields):
        doc = {'event': event, 'rank': rank, 'time': time.time(), **fields}
        print(json.dumps(doc, default=str), flush=True)
        with (root / f'events-rank{rank}.jsonl').open('a') as stream:
            stream.write(json.dumps(doc, default=str) + '\n')

    limit = min(len(plan['updates']), start + args.max_updates if args.max_updates else len(plan['updates']))
    prefetch, error = None, None
    try:
        if config.prefetch_rows and cached_rollouts is None and start < limit:
            from .prefetch import RowPrefetch
            prefetch = RowPrefetch(config, rank=rank, fallback_item=items[0])
            cleanup.callback(prefetch.close)
            begin = rank * config.effective_batch_per_rank
            prefetch.submit(start, plan['updates'][start][begin:begin + config.effective_batch_per_rank])
            log('row_prefetch_started', next_update=start, batches_ahead=1, worker_start_method='spawn')
    except Exception as exc:
        error = f'rank {rank}: {type(exc).__name__}: {exc}'
    agree(error, 'CPU row prefetch setup')

    from .samples import sample_slots, save_samples
    error = None
    try:
        if config.rollout_samples_per_update and not args.benchmark_shape_only:
            for update_items in plan['updates']:
                sample_slots(update_items, config.effective_batch_per_rank, config.rollout_samples_per_update)
    except Exception as exc:
        error = f'rank {rank}: {type(exc).__name__}: {exc}'
    agree(error, 'dashboard sample selection')

    torch.manual_seed(config.seed + rank)
    np.random.seed((config.seed + rank) % (2 ** 32))
    M = R.load_native()
    M.QWEN32, M.QWEN4B_CFG = args.qwen32_config, args.qwen4_config
    spec = specs['student']
    cfgs = (M.cfg32(spec['layers']), M.cfg4b(spec['tower_layers']), M.cfg06(spec['dec_layers']))
    if any(float(getattr(c, 'attention_dropout', 0)) for c in cfgs):
        raise ValueError('Replay requires deterministic, dropout-free model execution')
    vadapter, alpha = spec['vadapter'], spec['decp_bb_alpha']
    models = {}
    for role, path in (('student', config.student), ('audio_teacher', config.audio_teacher), ('pose_teacher', config.pose_teacher)):
        log('phase', phase='model_build_begin', role=role, checkpoint=str(path))
        model_spec = specs[role]
        role_cfgs = (M.cfg32(model_spec['layers']), M.cfg4b(model_spec['tower_layers']), M.cfg06(model_spec['dec_layers']))
        builder = R.build_frozen if role != 'student' and config.frozen_teacher_bf16 else M.build
        build_extra = {} if builder is R.build_frozen else {'init': False, 'dec_mode': 'fsdp'}
        model, _ = builder(*role_cfgs, device, **build_extra, decp_bb=role != 'audio_teacher',
                           pose_head=model_spec['pose_head'], proj_r=model_spec['proj_r'], tower_vocab=model_spec['tower_vocab'],
                           vadapter=model_spec['vadapter'], nopose=role == 'audio_teacher',
                           log=lambda message: log('build', role=role, message=message))
        log('phase', phase='model_load_begin', role=role, checkpoint=str(path))
        receipt = load_checkpoint(model, Path(path), device, gain_policy=model_spec['gain_policy'])
        R.install_decoder_layers(model, checkpoint_depth_decoder=config.checkpoint_depth_decoder)
        model.requires_grad_(role == 'student')
        model.eval()  # deterministic replay; gradients remain enabled for student
        if hasattr(model, 'decp_bb'):
            model.decp_bb_alpha = model_spec['decp_bb_alpha']
        models[role] = model
        log('loaded', role=role, receipt=receipt)
    student = models['student']
    optimizer = torch.optim.AdamW(student.parameters(), lr=config.learning_rate, betas=(.9, .95),
                                 eps=1e-8, weight_decay=0., fused=True)
    if args.resume:
        resumed = resume_checkpoint(student, {'main': optimizer}, args.resume, device)
        if int(resumed['step']) != start:
            raise ValueError('Loaded resume update differs from the preflighted cursor')
        log('resumed', next_update=start, checkpoint=str(args.resume))
    if args.warm_optimizer:
        log('phase', phase='optimizer_warmup_begin')
        warm_started = time.monotonic()
        receipt = warm_optimizer_state(optimizer)
        torch.cuda.synchronize(device)
        log('optimizer_warmup', seconds=time.monotonic() - warm_started, **receipt)
    front = None
    if cached_rollouts is None:
        import audio_s4q
        log('phase', phase='frontend_build_begin')
        front = audio_s4q.make_front({'s4_audio': 'flac', 's4_vox_model': args.voxtral, 's4_enc_hidden': 1280}, device)
        log('phase', phase='frontend_ready')
    front_offloaded = False
    rot = M.Rot(cfgs[0], device)
    fast_ppm = load_actor_native(M)
    actor_type = make_fastppm_class(fast_ppm)
    stop = {'requested': False}
    signal.signal(signal.SIGUSR1, lambda *_: stop.update(requested=True))
    next_save = checkpoint_deadline(config, time.monotonic())
    completed_update = start
    last_checkpoint = None
    for step in range(start, limit):
        applied_learning_rate = apply_learning_rate(optimizer, config.learning_rate, learning_rate_schedule, step)
        if args.benchmark_microbatches is not None:
            config = replace(config, physical_microbatch=args.benchmark_microbatches[step - start]).validate()
        actor_batch = config.rollout_batch_per_rank or config.physical_microbatch
        started = time.monotonic()
        log('phase', phase='data_audio_begin', update=step + 1,
            physical_microbatch=config.physical_microbatch, effective_batch_per_rank=config.effective_batch_per_rank,
            rollout_batch_per_rank=actor_batch, actor_instances=2 * actor_batch)
        torch.cuda.reset_peak_memory_stats(device)
        optimizer.zero_grad(set_to_none=True)
        selected_sample_slots = (sample_slots(plan['updates'][step], config.effective_batch_per_rank,
            config.rollout_samples_per_update) if rank == 0 and config.rollout_samples_per_update
            and not args.benchmark_shape_only and cached_rollouts is None else [])
        sample_waveforms = {}
        front_restore_seconds, front_offload_seconds = 0., 0.
        if cached_rollouts is not None:
            payload, rollouts = cached_rollouts
            cached_rollouts = None
            preparation_seconds = time.monotonic() - started
            rollout_started = time.monotonic()
            log('rollout_cache_loaded', update=step + 1, path=str(args.benchmark_load_rollouts), rows=len(rollouts))
        else:
            error, payload = None, None
            front_restore_seconds, front_offload_seconds = 0., 0.
            try:
                begin = rank * config.effective_batch_per_rank
                batch_items = plan['updates'][step][begin:begin + config.effective_batch_per_rank]
                if prefetch is not None:
                    wait_started = time.monotonic()
                    seed_rows = prefetch.take(step)
                    log('row_prefetch_ready', update=step + 1, rows=len(seed_rows),
                        wait_seconds=time.monotonic() - wait_started)
                    if step + 1 < limit:
                        prefetch.submit(step + 1, plan['updates'][step + 1][begin:begin + config.effective_batch_per_rank])
                else:
                    seed_rows = (data.load(items[0] if item is None else item, rank) for item in batch_items)
                if front_offloaded:
                    log('phase', phase='frontend_restore_begin', update=step + 1)
                    front_restore_seconds = move_audio_front(front, device)
                    front_offloaded = False
                    log('phase', phase='frontend_restore_end', update=step + 1, seconds=front_restore_seconds)
                payload = []
                for sample_slot, (item, seed_row) in enumerate(zip(batch_items, seed_rows, strict=True)):
                    if args.benchmark_prefix_frames is not None:
                        seed_row = benchmark_prefix(seed_row, args.benchmark_prefix_frames)
                    seed_row['dummy'] = item is None
                    if args.benchmark_shape_only:
                        validate_shape_seed(seed_row, plan['shape_spec'])
                    with torch.no_grad():
                        latents = front.encode([torch.from_numpy(seed_row['wav'])],
                                              [{'row_key': seed_row['row_key'], 'n': seed_row['conversation_frames']}])
                    # Both duplex paths consume these exact frozen features from
                    # here onward; avoid pickling unused ten-minute waveforms.
                    if sample_slot in selected_sample_slots:
                        sample_waveforms[sample_slot] = seed_row['wav']
                    seed_row.pop('wav')
                    payload.append((seed_row, latents.cpu()))
                    del latents
                if config.offload_audio_front:
                    log('phase', phase='frontend_offload_begin', update=step + 1)
                    front_offload_seconds = move_audio_front(front, 'cpu')
                    front_offloaded = True
                    log('phase', phase='frontend_offload_end', update=step + 1, seconds=front_offload_seconds)
            except Exception as exc:
                error = f'rank {rank}: {type(exc).__name__}: {exc}'
            agree(error, 'data/audio preparation')
            torch.cuda.synchronize(device)
            preparation_seconds = time.monotonic() - started
            rollout_started = time.monotonic()
            if args.benchmark_shape_only:
                rollouts = [synthetic_rollout(row) for row, _ in payload]
                log('synthetic_shape_inputs', update=step + 1, shape=plan['shape_spec'],
                    rows=len(rollouts), actor_executed=False, on_policy=False)
            else:
                # The paired actor allocates on both GPUs from one process.
                # After learner replay, its peer can retain tens of GiB of
                # unused CUDA allocator blocks that this process cannot reuse.
                # Each rank must return its own unused blocks before ANY actor
                # allocation; the collective agreement is the global barrier.
                error = None
                try:
                    released = release_publication_cache([device])
                    log('actor_preallocation_cache', update=step + 1, **released)
                except Exception as exc:
                    error = f'rank {rank}: {type(exc).__name__}: {exc}'
                agree(error, 'learner cache release before actor allocation')
                pair_rows = [None, None]
                dist.all_gather_object(pair_rows, payload, group=pair)
                actor, ldev, error = None, None, None
                devices = [torch.device('cuda', local - local % 2 + offset) for offset in (0, 1)]
                log('phase', phase='actor_build_begin', update=step + 1)
                if rank == leader:
                    try:
                        for actor_device in devices:
                            torch.cuda.reset_peak_memory_stats(actor_device)
                        actor, ldev = build_actor(student, cfgs, devices, vadapter, alpha,
                                                  backbone_cut=config.actor_backbone_cut)
                    except Exception as exc:
                        error = f'actor rank {rank}: {type(exc).__name__}: {exc}'
                torch.cuda.set_device(device)
                agree(error, 'actor allocation')
                log('phase', phase='actor_publish_begin', update=step + 1)
                publish_actor(student, actor)
                error = None
                try:
                    released = release_publication_cache(devices if rank == leader else [device])
                    log('actor_publication_cache', update=step + 1, **released)
                except Exception as exc:
                    error = f'rank {rank}: {type(exc).__name__}: {exc}'
                # The peer rank's allocator must release its gathers before the actor
                # process allocates KV/graphs on that peer's device.
                agree(error, 'actor publication cache release')
                log('phase', phase='actor_rollout_begin', update=step + 1)
                generated, engine, error = [[], []], None, None
                if rank == leader:
                    try:
                        all_rows = [item for rows in pair_rows for item in rows]
                        max_frames = max(row['conversation_frames'] for row, _ in all_rows)
                        max_prefix = max(int(row['P'][0]) for row, _ in all_rows)
                        if config.fixed_sequence_frames is not None and config.conversation_prefix_frames is not None:
                            # Reserve the same worst-case cache shape on every
                            # pair, independently of reference-clip lengths.
                            max_frames = config.conversation_prefix_frames
                            max_prefix = max(max_prefix, config.fixed_sequence_frames - max_frames - 2)
                        engine = actor_type(actor, ldev, cfgs, devices, n_inst=2 * actor_batch,
                                            sampler_backend=config.actor_sampler_backend,
                                            depth_cache_positions=config.actor_depth_cache_positions,
                                            attention_precision=config.actor_attention_precision,
                                            linear_backend=config.actor_linear_backend,
                                            policy_probe_frames=(config.prefill_frames + 2, config.prefill_frames + 31,
                                                                 config.prefill_frames + 95) if args.profile_policy_agreement else (),
                                            max_s=max_frames / 12.5, p_max=max_prefix + 8)
                        if not args.no_graphs:
                            log('phase', phase='actor_capture_begin', update=step + 1)
                            engine.capture(f=25, P=max_prefix)
                            log('phase', phase='actor_capture_end', update=step + 1)
                            if args.profile_actor_graphs:
                                engine.enable_stage_profile(32)
                        profile_reported = False
                        def report_progress(**fields):
                            nonlocal profile_reported
                            log('rollout_progress', update=step + 1,
                                microbatch=begin // actor_batch, **fields)
                            if (args.profile_actor_graphs and not args.no_graphs and not profile_reported
                                    and fields['current_frame'] >= config.prefill_frames + config.speech_delay + 32):
                                log('actor_graph_profile', update=step + 1, **engine.stage_profile_result())
                                profile_reported = True
                        for begin in range(0, config.effective_batch_per_rank, actor_batch):
                            selected = [(owner, slot, *pair_rows[owner][slot]) for owner in range(2)
                                        for slot in range(begin, begin + actor_batch)]
                            seeds = []
                            for instance, (owner, slot, row, lat) in enumerate(selected):
                                engine.set_audio_features(audio_features_from_latents(engine, lat),
                                                          prefix=int(row['P'][0]), instance=instance)
                                seeds.append(rollout_seed(config.seed, step, leader + owner, slot))
                            outputs = rollout_rows(engine, [value[2] for value in selected], seeds=seeds,
                                                   graphs=not args.no_graphs,
                                                   progress=report_progress)
                            log('phase', phase='actor_microbatch_end', update=step + 1,
                                microbatch=begin // actor_batch,
                                instances=len(selected), seconds=time.monotonic() - rollout_started)
                            if args.profile_policy_agreement:
                                from .policy_probe import same_root_depth_probe
                                log('actor_depth_policy_probe', update=step + 1,
                                    microbatch=begin // actor_batch,
                                    **same_root_depth_probe(engine, [value[2] for value in selected], R))
                            for (owner, _, _, _), rollout in zip(selected, outputs):
                                generated[owner].append(rollout)
                    except Exception as exc:
                        error = f'actor rank {rank}: {type(exc).__name__}: {exc}'
                    finally:
                        memory = []
                        for actor_device in devices:
                            free_bytes, total_bytes = torch.cuda.mem_get_info(actor_device)
                            memory.append({'device': str(actor_device),
                                           'process_peak_allocated_gib': torch.cuda.max_memory_allocated(actor_device) / 2 ** 30,
                                           'process_peak_reserved_gib': torch.cuda.max_memory_reserved(actor_device) / 2 ** 30,
                                           'device_used_gib_before_teardown': (total_bytes - free_bytes) / 2 ** 30})
                        log('actor_memory', update=step + 1, devices=memory,
                            seconds_including_build_publish=time.monotonic() - rollout_started)
                        if engine is not None:
                            engine.close()
                        del engine, actor
                        gc.collect()
                        for dev in devices:
                            with torch.cuda.device(dev):
                                torch.cuda.empty_cache()
                torch.cuda.set_device(device)
                agree(error, 'student rollout')
                dist.broadcast_object_list(generated, src=leader, group=pair)
                rollouts = generated[rank % 2]
                del pair_rows, generated
        torch.cuda.synchronize(device)
        rollout_seconds = time.monotonic() - rollout_started
        error = None
        if selected_sample_slots:
            try:
                receipt = save_samples(root, update=step + 1, payload=payload, rollouts=rollouts,
                    waveforms=sample_waveforms, slots=selected_sample_slots,
                    fps=config.frames_per_second, source_identity=identity['source_sha256'])
                log('rollout_samples_saved', update=step + 1, **receipt)
            except Exception as exc:
                error = f'rank {rank}: {type(exc).__name__}: {exc}'
        agree(error, 'dashboard rollout snapshot export')
        del sample_waveforms
        if args.benchmark_save_rollouts:
            cache_started = time.monotonic()
            error, receipt = None, None
            try:
                receipt = save_rollout_rank(args.benchmark_save_rollouts, trajectory_identity, payload, rollouts)
            except Exception as exc:
                error = f'rank {rank}: {type(exc).__name__}: {exc}'
            agree(error, 'completed rollout cache shard writes')
            receipts = [None] * world
            dist.all_gather_object(receipts, receipt, group=control_group)
            error = None
            try:
                commit_rollout_cache(args.benchmark_save_rollouts, trajectory_identity, receipts)
            except Exception as exc:
                error = f'rank {rank}: {type(exc).__name__}: {exc}'
            agree(error, 'rollout cache global completion')
            log('rollout_cache_saved', update=step + 1, path=str(args.benchmark_save_rollouts),
                seconds=time.monotonic() - cache_started, receipt=receipt)
        bundles = [replay_from_rollout(row, rollout.text, rollout.audio, rollout.pose)
                   for (row, _), rollout in zip(payload, rollouts)]
        latents = [lat for _, lat in payload]
        del payload
        normalization = R.update_normalization(bundles, device)
        teacher_stages = ReplayStages(device) if args.profile_replay_stages else None
        log('phase', phase='teacher_update_begin', update=step + 1,
            teacher_batch_per_rank=config.teacher_batch_per_rank or config.effective_batch_per_rank,
            pose_cfg_batch=config.pose_cfg_batch)
        teacher_started = time.monotonic()
        prepared_teachers = R.prepare_teacher_update(models['audio_teacher'], models['pose_teacher'],
                            bundles, latents, config, rot, device, world=world,
                            normalization=normalization, stage_profiler=teacher_stages)
        torch.cuda.synchronize(device)
        teacher_seconds = time.monotonic() - teacher_started
        teacher_peak = {'peak_allocated_gib': torch.cuda.max_memory_allocated(device) / 2 ** 30,
                        'peak_reserved_gib': torch.cuda.max_memory_reserved(device) / 2 ** 30}
        log('phase', phase='teacher_update_end', update=step + 1, seconds=teacher_seconds, **teacher_peak)
        if teacher_stages is not None:
            log('teacher_stage_profile', update=step + 1, **teacher_stages.collect())
        replay_trials = []
        replay_benchmark = args.benchmark_replay_microbatches is not None or args.benchmark_replay_variants is not None
        for replay_trial, config in enumerate(replay_candidates(config, args.benchmark_replay_microbatches, optimizer,
                                                                          variants=args.benchmark_replay_variants)):
            if replay_benchmark:
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                log('replay_benchmark_begin', update=step + 1, trial=replay_trial,
                    physical_microbatch=config.physical_microbatch, same_trajectories=True, optimizer_updates_since_rollout=0)
            R.install_decoder_layers(student, checkpoint_depth_decoder=config.checkpoint_depth_decoder)
            replay_started = time.monotonic()
            metrics, difference_sum, difference_count, difference_max = {}, 0., 0, 0.
            agreement = PolicyAgreement(args.policy_agreement_frames) if args.profile_policy_agreement else None
            categorical_reports = []
            for begin in range(0, config.effective_batch_per_rank, config.physical_microbatch):
                end = begin + config.physical_microbatch
                probes = (CategoricalAgreement(bundles[begin:end], rollouts[begin:end], row_offset=begin)
                          if agreement is not None else None)
                replay_stages = ReplayStages(device) if args.profile_replay_stages else None
                log('phase', phase='replay_microbatch_begin', update=step + 1,
                    trial=replay_trial, physical_microbatch=config.physical_microbatch,
                    microbatch=begin // config.physical_microbatch, rows=end - begin,
                    packed_positions=((end - begin) * config.fixed_sequence_frames if config.fixed_sequence_frames
                                      else sum(len(bundle['student']['pid']) for bundle in bundles[begin:end])),
                    actual_positions=sum(len(bundle['student']['pid']) for bundle in bundles[begin:end]))
                values, logp, sampled = R.replay_update(student, models['audio_teacher'], models['pose_teacher'],
                                bundles[begin:end], latents[begin:end], config, rot, device,
                                world=world, normalization=normalization, policy_probes=probes, stage_profiler=replay_stages,
                                prepared_teachers=prepared_teachers.slice_rows(begin, end))
                if replay_stages is not None:
                    log('replay_stage_profile', update=step + 1, trial=replay_trial,
                        microbatch=begin // config.physical_microbatch, physical_microbatch=config.physical_microbatch,
                        decoder_gradient_sync=config.decoder_gradient_sync, **replay_stages.collect())
                if agreement is not None:
                    agreement.add(bundles[begin:end], rollouts[begin:end], logp, sampled, row_offset=begin)
                    categorical_reports.append(probes.report())
                for key, value in values.items():
                    metrics[key] = metrics.get(key, 0.) + value
                if not args.benchmark_shape_only:
                    # Compare the exact sampled trajectories against their packed replay.
                    behavior = np.concatenate([np.concatenate((rollout.behavior_logprobs['text'][:, None],
                                rollout.behavior_logprobs['audio'], rollout.behavior_logprobs['pose']), 1)
                                for rollout in rollouts[begin:end]])
                    diff = (logp - torch.as_tensor(behavior, device=device)).abs()[sampled]
                    difference_sum += float(diff.sum())
                    difference_count += len(diff)
                    difference_max = max(difference_max, float(diff.max()) if len(diff) else 0.)
                    del behavior, diff
                del values, logp, sampled
                log('phase', phase='replay_microbatch_end', update=step + 1,
                    microbatch=begin // config.physical_microbatch, seconds=time.monotonic() - replay_started)
            torch.cuda.synchronize(device)
            replay_seconds = time.monotonic() - replay_started
            if agreement is not None:
                log('policy_agreement', update=step + 1, trial=replay_trial,
                    physical_microbatch=config.physical_microbatch, **agreement.report(), categorical=categorical_reports)
            if not args.benchmark_shape_only:
                metrics['policy/logprob_mean_abs_difference'] = difference_sum / max(1, difference_count)
                metrics['policy/logprob_max_abs_difference'] = difference_max
            error = None if all(math.isfinite(value) for value in metrics.values()) else f'rank {rank}: nonfinite training metric'
            agree(error, 'finite metric check before optimizer update')
            if replay_benchmark:
                peak_allocated = torch.cuda.max_memory_allocated(device) / 2 ** 30
                peak_reserved = torch.cuda.max_memory_reserved(device) / 2 ** 30
                stats = gradient_statistics(student.parameters(), device)
                error = None if all(math.isfinite(stats[key]) for key in ('sum', 'l2', 'max_abs')) else f'rank {rank}: nonfinite gradient checksum'
                agree(error, 'replay benchmark finite gradients')
                trial = {'trial': replay_trial, 'physical_microbatch': config.physical_microbatch,
                         'decoder_chunk_frames': config.decoder_chunk_frames, 'text_chunk_frames': config.text_chunk_frames,
                         'decoder_gradient_sync': config.decoder_gradient_sync,
                         'checkpoint_depth_decoder': config.checkpoint_depth_decoder,
                         'synthetic_shape_only': args.benchmark_shape_only,
                         'replay_seconds': replay_seconds, 'teacher_seconds_shared': teacher_seconds,
                         'peak_allocated_gib': peak_allocated,
                         'peak_reserved_gib': peak_reserved, 'gradient': stats, 'metrics': dict(metrics)}
                replay_trials.append(trial)
                log('replay_benchmark', update=step + 1, **trial)
        del prepared_teachers
        optimizer_started = time.monotonic()
        log('phase', phase='optimizer_begin', update=step + 1)
        error, norm_value = None, None
        try:
            norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1., error_if_nonfinite=True)
            norm_value = float(norm.full_tensor() if hasattr(norm, 'full_tensor') else norm)
        except Exception as exc:
            error = f'rank {rank}: {type(exc).__name__}: {exc}'
        agree(error, 'finite-gradient/metric check')
        optimizer.step()  # exactly one update after all modalities/codebooks/credits
        torch.cuda.synchronize(device)
        optimizer_seconds = time.monotonic() - optimizer_started
        log('profile', update=step + 1, synthetic_shape_only=args.benchmark_shape_only, preparation_seconds=preparation_seconds,
            frontend_restore_seconds=front_restore_seconds, frontend_offload_seconds=front_offload_seconds,
            rollout_seconds=rollout_seconds, replay_seconds=teacher_seconds + replay_seconds,
            teacher_seconds=teacher_seconds, student_replay_seconds=replay_seconds, optimizer_seconds=optimizer_seconds,
            teacher_phase_memory=teacher_peak,
            peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2 ** 30,
            peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2 ** 30,
            physical_microbatch=config.physical_microbatch, effective_batch_per_rank=config.effective_batch_per_rank,
            rollout_batch_per_rank=actor_batch, actor_instances=0 if args.benchmark_shape_only or args.benchmark_load_rollouts else 2 * actor_batch,
            global_effective_batch=world * config.effective_batch_per_rank,
            replay_segments_per_forward=config.physical_microbatch, replay_benchmarks=replay_trials)
        completed_update = step + 1
        metrics['gradient_norm'] = norm_value
        values = torch.tensor(list(metrics.values()), device=device, dtype=torch.float64)
        maximum_difference = torch.tensor(metrics.get('policy/logprob_max_abs_difference', 0.), device=device, dtype=torch.float64)
        dist.all_reduce(values)
        dist.all_reduce(maximum_difference, op=dist.ReduceOp.MAX)
        values /= world
        metrics = dict(zip(metrics, values.tolist()))
        if not args.benchmark_shape_only:
            metrics['policy/logprob_max_abs_difference'] = float(maximum_difference)
        if not all(math.isfinite(value) for value in metrics.values()):
            raise RuntimeError('Nonfinite training metric')
        metrics.update(update=step + 1, learning_rate=applied_learning_rate, seconds=time.monotonic() - started,
                       synthetic_shape_only=args.benchmark_shape_only)
        log('update', **metrics)
        if rank == 0:
            with (root / 'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(metrics) + '\n')
        flag = torch.tensor(int(stop['requested'] or (root / 'control' / 'STOP_AFTER_SAVE').exists()), device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        save = torch.tensor(int(rank == 0 and checkpoint_due(config, completed_update=step + 1,
                            limit=limit, now=time.monotonic(), deadline=next_save)), device=device)
        dist.all_reduce(save, op=dist.ReduceOp.MAX)
        if not args.no_save and (int(save) or int(flag)):
            # Keep reconstruction metadata in exports, not just an opaque OPD
            # identity: the result must be usable as a later student checkpoint.
            save_cfg = checkpoint_export_config(config, specs['student'], identity=identity,
                            data_config=data.cfg, next_update=step + 1, plan_digest=plan['digest'])
            checkpoint = save_checkpoint(student, {'main': optimizer}, root, step + 1, save_cfg)
            last_checkpoint = checkpoint['resume_path']
            log('checkpoint', path=checkpoint)
            next_save = checkpoint_deadline(config, time.monotonic())
        if int(flag):
            break
        del bundles, latents, rollouts, normalization
    if rank == 0:
        atomic_json(root / 'RESULT.json', {'status': 'completed' if completed_update == len(plan['updates']) else 'bounded_or_stopped',
                    'next_update': completed_update, 'total_updates': len(plan['updates']),
                    'last_checkpoint': last_checkpoint, 'checkpointing_enabled': not args.no_save,
                    'synthetic_shape_only': args.benchmark_shape_only,
                    'identity': identity})
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
