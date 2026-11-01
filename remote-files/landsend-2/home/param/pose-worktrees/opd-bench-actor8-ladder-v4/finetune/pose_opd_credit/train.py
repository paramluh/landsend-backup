"""Two-teacher full-row OPD: current student rollout, analytic KL, finite credit.

Launch with torchrun, an even number of visible GPUs per host, and an explicit
JSON TrainingConfig. --prepare-only validates dataset membership without CUDA.
No training processes are submitted to other machines by this entry point.
"""
from __future__ import annotations

import argparse
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

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def canonical_json(value):
    """The same representation before and after checkpoint JSON round-trips."""
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


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


def make_plan(items, *, world, epochs, seed, rows_per_rank=1):
    """Full-row coverage, independently shuffled epochs, explicit zero-weight tail slots."""
    if (not items or type(world) is not int or world < 1 or type(epochs) is not int or epochs < 1
            or type(seed) is not int or seed < 0 or type(rows_per_rank) is not int or rows_per_rank < 1):
        raise ValueError('Nonempty data and positive world/epochs are required')
    normalized = [tuple(map(int, item)) for item in items]
    if any(len(item) != 3 for item in normalized) or len(set(normalized)) != len(normalized):
        raise ValueError('Data plan requires unique (store, row, orientation) items')
    updates = []
    batch = world * rows_per_rank
    for epoch in range(epochs):
        order = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(len(items))
        rows = [list(normalized[i]) for i in order]
        rows += [None] * ((-len(rows)) % batch)
        updates.extend(rows[i:i + batch] for i in range(0, len(rows), batch))
    digest = hashlib.sha256(json.dumps(updates, separators=(',', ':')).encode()).hexdigest()
    return {'format': 'pose-opd-credit-plan-v1', 'updates': updates, 'digest': digest,
            'rows_per_epoch': len(items), 'epochs': epochs, 'world': world, 'seed': seed,
            'rows_per_rank': rows_per_rank, 'effective_batch': batch}


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
    parser.add_argument('--no-graphs', action='store_true', help='Eager cached actor for diagnostics')
    parser.add_argument('--profile-actor-graphs', action='store_true',
                        help='Time the first 32 captured inference steps with CUDA events')
    parser.add_argument('--qwen32-config', default='/home/param/models/imported-weights/ppm-811-17850-base-bf16/configs/Qwen3-32B')
    parser.add_argument('--qwen4-config', default='/home/param/models/imported-weights/ppm-811-17850-base-bf16/configs/Qwen3-4B')
    parser.add_argument('--voxtral', default='/home/param/models/imported-weights/aloe-ft-vox17127-nopersona/assets/voxtral-realtime')
    args = parser.parse_args(argv)
    if args.max_updates is not None and args.max_updates < 1:
        parser.error('--max-updates must be positive')
    if args.no_save and args.max_updates is None:
        parser.error('--no-save requires a bounded --max-updates')
    if args.prepare_only and (args.resume is not None or args.no_save or args.max_updates is not None):
        parser.error('--prepare-only cannot be combined with resume/profile execution flags')
    if args.benchmark_microbatches is not None:
        if not args.no_save or args.max_updates is None:
            parser.error('--benchmark-microbatches requires bounded --max-updates and --no-save')
        try:
            args.benchmark_microbatches = [int(value) for value in args.benchmark_microbatches.split(',')]
        except ValueError:
            parser.error('--benchmark-microbatches must be comma-separated positive integers')
        if len(args.benchmark_microbatches) != args.max_updates or any(value <= 0 for value in args.benchmark_microbatches):
            parser.error('--benchmark-microbatches needs one positive batch size per --max-updates')
    return args


def validate_benchmark_microbatches(config, schedule):
    if schedule is not None:
        if schedule[0] != config.physical_microbatch:
            raise ValueError('First benchmark microbatch must match configured physical_microbatch')
        if any(config.effective_batch_per_rank % value for value in schedule):
            raise ValueError('Every benchmark microbatch must divide effective_batch_per_rank')


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
        root = canonical_run_dir(args.run_dir)
        root.mkdir(parents=True, exist_ok=True)
        data = ProductionRows(config)
        items = data.items('train')
        plan = make_plan(items, world=world, epochs=config.epochs, seed=config.seed,
                         rows_per_rank=config.effective_batch_per_rank)
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
                    'world': world, 'actor': 'paired-cached-natural-categorical', 'models': specs,
                    'optimizer': {'type': 'AdamW', 'lr': config.learning_rate, 'betas': [.9, .95],
                                  'eps': 1e-8, 'weight_decay': 0., 'max_grad_norm': 1.},
                    'model_configs': [args.qwen32_config, args.qwen4_config], 'voxtral': args.voxtral,
                    'asset_config_sha256': assets})
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
        R.install_decoder_layers(model)
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
    import audio_s4q
    log('phase', phase='frontend_build_begin')
    front = audio_s4q.make_front({'s4_audio': 'flac', 's4_vox_model': args.voxtral, 's4_enc_hidden': 1280}, device)
    front_offloaded = False
    log('phase', phase='frontend_ready')
    rot = M.Rot(cfgs[0], device)
    fast_ppm = load_actor_native(M)
    actor_type = make_fastppm_class(fast_ppm)
    stop = {'requested': False}
    signal.signal(signal.SIGUSR1, lambda *_: stop.update(requested=True))
    limit = min(len(plan['updates']), start + args.max_updates if args.max_updates else len(plan['updates']))
    next_save = time.monotonic() + config.checkpoint_interval_seconds
    completed_update = start
    last_checkpoint = None
    for step in range(start, limit):
        if args.benchmark_microbatches is not None:
            config = replace(config, physical_microbatch=args.benchmark_microbatches[step - start]).validate()
        actor_batch = config.rollout_batch_per_rank or config.physical_microbatch
        started = time.monotonic()
        log('phase', phase='data_audio_begin', update=step + 1,
            physical_microbatch=config.physical_microbatch, effective_batch_per_rank=config.effective_batch_per_rank,
            rollout_batch_per_rank=actor_batch, actor_instances=2 * actor_batch)
        torch.cuda.reset_peak_memory_stats(device)
        optimizer.zero_grad(set_to_none=True)
        error, payload = None, None
        front_restore_seconds, front_offload_seconds = 0., 0.
        try:
            if front_offloaded:
                log('phase', phase='frontend_restore_begin', update=step + 1)
                front_restore_seconds = move_audio_front(front, device)
                front_offloaded = False
                log('phase', phase='frontend_restore_end', update=step + 1, seconds=front_restore_seconds)
            payload = []
            for slot in range(config.effective_batch_per_rank):
                item = plan['updates'][step][rank * config.effective_batch_per_rank + slot]
                seed_row = data.load(items[0] if item is None else item, rank)
                seed_row['dummy'] = item is None
                with torch.no_grad():
                    latents = front.encode([torch.from_numpy(seed_row['wav'])],
                                          [{'row_key': seed_row['row_key'], 'n': seed_row['conversation_frames']}])
                # Both duplex paths consume these exact frozen features from
                # here onward; avoid pickling unused ten-minute waveforms.
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
                engine = actor_type(actor, ldev, cfgs, devices, n_inst=2 * actor_batch,
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
                    if args.profile_actor_graphs and not args.no_graphs and not profile_reported:
                        log('actor_graph_profile', update=step + 1, **engine.stage_profile_result())
                        profile_reported = True
                for begin in range(0, config.effective_batch_per_rank, actor_batch):
                    selected = [(owner, slot, *pair_rows[owner][slot]) for owner in range(2)
                                for slot in range(begin, begin + actor_batch)]
                    seeds = []
                    for instance, (owner, slot, row, lat) in enumerate(selected):
                        engine.set_audio_features(audio_features_from_latents(engine, lat),
                                                  prefix=int(row['P'][0]), instance=instance)
                        seeds.append(int(np.random.SeedSequence([config.seed, step, leader + owner, slot]).generate_state(1)[0]))
                    outputs = rollout_rows(engine, [value[2] for value in selected], seeds=seeds,
                                           graphs=not args.no_graphs,
                                           progress=report_progress)
                    log('phase', phase='actor_microbatch_end', update=step + 1,
                        microbatch=begin // actor_batch,
                        instances=len(selected), seconds=time.monotonic() - rollout_started)
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
        bundles = [replay_from_rollout(row, rollout.text, rollout.audio, rollout.pose)
                   for (row, _), rollout in zip(payload, rollouts)]
        latents = [lat for _, lat in payload]
        del pair_rows, generated, payload
        torch.cuda.synchronize(device)
        rollout_seconds = time.monotonic() - rollout_started
        replay_started = time.monotonic()
        normalization = R.update_normalization(bundles, device)
        metrics, difference_sum, difference_count, difference_max = {}, 0., 0, 0.
        for begin in range(0, config.effective_batch_per_rank, config.physical_microbatch):
            end = begin + config.physical_microbatch
            log('phase', phase='replay_microbatch_begin', update=step + 1,
                microbatch=begin // config.physical_microbatch, rows=end - begin,
                packed_positions=sum(len(bundle['student']['pid']) for bundle in bundles[begin:end]))
            values, logp, sampled = R.replay_update(student, models['audio_teacher'], models['pose_teacher'],
                            bundles[begin:end], latents[begin:end], config, rot, device,
                            world=world, normalization=normalization)
            for key, value in values.items():
                metrics[key] = metrics.get(key, 0.) + value
            # Compare the exact sampled trajectories against their packed replay.
            behavior = np.concatenate([np.concatenate((rollout.behavior_logprobs['text'][:, None],
                        rollout.behavior_logprobs['audio'], rollout.behavior_logprobs['pose']), 1)
                        for rollout in rollouts[begin:end]])
            diff = (logp - torch.as_tensor(behavior, device=device)).abs()[sampled]
            difference_sum += float(diff.sum())
            difference_count += len(diff)
            difference_max = max(difference_max, float(diff.max()) if len(diff) else 0.)
            del values, logp, sampled, behavior, diff
            log('phase', phase='replay_microbatch_end', update=step + 1,
                microbatch=begin // config.physical_microbatch, seconds=time.monotonic() - replay_started)
        torch.cuda.synchronize(device)
        replay_seconds = time.monotonic() - replay_started
        metrics['policy/logprob_mean_abs_difference'] = difference_sum / max(1, difference_count)
        metrics['policy/logprob_max_abs_difference'] = difference_max
        error = None if all(math.isfinite(value) for value in metrics.values()) else f'rank {rank}: nonfinite training metric'
        agree(error, 'finite metric check before optimizer update')
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
        log('profile', update=step + 1, preparation_seconds=preparation_seconds,
            frontend_restore_seconds=front_restore_seconds, frontend_offload_seconds=front_offload_seconds,
            rollout_seconds=rollout_seconds, replay_seconds=replay_seconds, optimizer_seconds=optimizer_seconds,
            peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2 ** 30,
            peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2 ** 30,
            physical_microbatch=config.physical_microbatch, effective_batch_per_rank=config.effective_batch_per_rank,
            rollout_batch_per_rank=actor_batch, actor_instances=2 * actor_batch,
            global_effective_batch=world * config.effective_batch_per_rank,
            replay_segments_per_forward=config.physical_microbatch)
        completed_update = step + 1
        metrics['gradient_norm'] = norm_value
        values = torch.tensor(list(metrics.values()), device=device, dtype=torch.float64)
        maximum_difference = torch.tensor(metrics['policy/logprob_max_abs_difference'], device=device, dtype=torch.float64)
        dist.all_reduce(values)
        dist.all_reduce(maximum_difference, op=dist.ReduceOp.MAX)
        values /= world
        metrics = dict(zip(metrics, values.tolist()))
        metrics['policy/logprob_max_abs_difference'] = float(maximum_difference)
        if not all(math.isfinite(value) for value in metrics.values()):
            raise RuntimeError('Nonfinite training metric')
        metrics.update(update=step + 1, seconds=time.monotonic() - started)
        log('update', **metrics)
        if rank == 0:
            with (root / 'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(metrics) + '\n')
        flag = torch.tensor(int(stop['requested'] or (root / 'control' / 'STOP_AFTER_SAVE').exists()), device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        save = torch.tensor(int(rank == 0 and (time.monotonic() >= next_save or step + 1 == limit)), device=device)
        dist.all_reduce(save, op=dist.ReduceOp.MAX)
        if not args.no_save and (int(save) or int(flag)):
            # Keep reconstruction metadata in exports, not just an opaque OPD
            # identity: the result must be usable as a later student checkpoint.
            save_cfg = dict(specs['student']['config'], opd_identity=identity,
                            keep_latest_optimizer_state=False, vadapter=vadapter,
                            decp_bb_alpha=alpha, data=data.cfg,
                            **{name: specs['student'][name] for name in
                               ('layers', 'tower_layers', 'dec_layers', 'proj_r', 'tower_vocab')},
                            _runtime_state={'next_update': step + 1, 'plan_digest': plan['digest']})
            checkpoint = save_checkpoint(student, {'main': optimizer}, root, step + 1, save_cfg)
            last_checkpoint = checkpoint['resume_path']
            log('checkpoint', path=checkpoint)
            next_save = time.monotonic() + config.checkpoint_interval_seconds
        if int(flag):
            break
        del bundles, latents, rollouts, normalization
    if rank == 0:
        atomic_json(root / 'RESULT.json', {'status': 'completed' if completed_update == len(plan['updates']) else 'bounded_or_stopped',
                    'next_update': completed_update, 'total_updates': len(plan['updates']),
                    'last_checkpoint': last_checkpoint, 'checkpointing_enabled': not args.no_save,
                    'identity': identity})
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
