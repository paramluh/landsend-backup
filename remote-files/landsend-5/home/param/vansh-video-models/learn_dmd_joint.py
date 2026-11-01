"""Shared lifecycle for nine-reference stabilizer + resolver DMD.

Each optimizer step accumulates a full stabilizer batch and a full resolver
batch into the SAME dense model, using the same objective weights for critic
FM and student DMD. The default entry point uses four-target no-GT training;
learn_dmd_full_gt.py selects single-target full-GT training. See their docs.
"""
import argparse
import gc
import json
import math
import os
import random
import shutil
import socket
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch.utils.data import DataLoader

from distillation.bimodal_dmd import noised, resolve_times, stab_times
from distillation.batched_training import (generate_pooled_targets,
    objective_checkpoint, regroup_prepared_rows)
from distillation.bimodal_net import audio_latents, build_net, load_mimi, load_state_dict
from distillation.checkpoints import (complete_checkpoint, file_sha256, load_model_weights,
    load_rank_state, read_checkpoint_metadata, restore_rng, rng_state,
    save_ema, save_rank_states, tensor_bytes, write_json_atomic)
from distillation.joint_data import build_joint_data
from distillation.joint_dmd import (endpoint_fm_loss, generate_resolver, generate_stabilizer,
    score_resolver, score_stabilizer)
from distillation.joint_eval import run_sensitivity
from distillation.joint_metrics import (TIMING_PHASES, add_dmd_diagnostics,
    add_resolver_history_dropout, add_timing_aliases, finalize_diagnostics)
from distillation.joint_runtime import (acknowledge_save_request, claim_save_request,
    checkpoint_compatibility, enforce_checkpoint_retention, sync_checkpoint_files, sync_directory, training_indices,
    full_gt_checkpoint_due, full_gt_evaluation_due, planned_student_steps, validate_reference_bank)
from distillation.joint_tracking import start_online_tracking
from distillation.training_metrics import TrainingMetrics
from distillation.training_tracking import start_training_tracking
from distillation.resolver_data import learning_rate_factor
from distillation.resume_transition import record_resume_transition, validate_resume_compatibility
from learn_dmd_bimodal import GradFlats, Sums, load_config
from utils import saver
from utils.harness import EMA, Zero1FP32MasterAdamW


def root_action(action):
    """Broadcast root-only I/O results/errors before peers enter another collective."""
    box = [None]
    if dist.get_rank() == 0:
        try:
            box[0] = (True, action())
        except Exception as exc:
            box[0] = (False, f'{type(exc).__name__}: {exc}')
    dist.broadcast_object_list(box, src=0, group=saver.gloo)
    success, result = box[0]
    if not success:
        raise RuntimeError(result)
    return result


def root_tracking_action(action):
    """Isolate synchronous SDK/API calls from each rank's training RNG state."""
    def isolated():
        state = rng_state()
        try:
            return action()
        finally:
            restore_rng(state)
    return root_action(isolated)


def dmd_frames(y, teacher, critic, target_velocity, epsilon):
    """Existing endpoint-normalized DMD, before reduction over rows/targets."""
    with torch.no_grad():
        normalizer = (teacher.float() - target_velocity.float()).abs().mean((1, 3, 4))
        gradient = (teacher.float() - critic.float()) / normalizer.clamp_min(epsilon)[:, None, :, None, None]
        target = y.detach().float() - gradient
    return .5 * (y.float() - target).square().mean((1, 3, 4))


def validate_config(config, *, full_gt=False):
    if config['model']['num_refs'] != 9:
        raise ValueError('This experiment requires the nine-reference checkpoint and nine reference slots')
    if config['epochs'] != 1 or config.get('total_steps') is not None:
        raise ValueError('Use epochs=1; total_steps is derived from the complete eligible corpus')
    if config['history_t'] != .2 or config['band_steps'] != [2, 2, 2, 2]:
        raise ValueError('Stabilizer uses eight steps from 1 to .2')
    if full_gt and config.get('training_method') != 'full_gt_single_target':
        raise ValueError('The full-GT entry point requires training_method="full_gt_single_target"')
    if not full_gt and config.get('training_method') == 'full_gt_single_target':
        raise ValueError('Use learn_dmd_full_gt.py for single-target full-GT training')
    if not full_gt and (config['stabilizer_prefix_frames'] != 6 or config['critic_prefix_frames'] != 3):
        raise ValueError('Student uses six detached prefix frames; critic samples use three (seven total)')
    history_dropout = config['stabilizer_history_dropout']
    if (len(history_dropout) > 6 or
            any(not math.isfinite(p) or not 0 <= p <= 1 for p in history_dropout)):
        raise ValueError('Stabilizer dropout requires up to six nearest-prefix-first probabilities in [0,1]')
    if config['resolver_steps'] != 1:
        raise ValueError('Use the one-step GT-conditioned resolver')
    guidance = config['teacher_guidance_scale']
    if (not math.isfinite(guidance) or guidance < 1
            or (not full_gt and guidance != 2.5)):
        raise ValueError('Joint no-GT guidance is 2.5; full-GT guidance must be finite and >= 1')
    allow_guidance_change = config.get('resume_allow_teacher_guidance_change', False)
    if type(allow_guidance_change) is not bool:
        raise ValueError('resume_allow_teacher_guidance_change must be a bool')
    if allow_guidance_change and (not full_gt or not config.get('resume_from') or guidance != 2.5):
        raise ValueError('Guidance resume override requires full-GT resume_from and teacher_guidance_scale=2.5')
    if config['objective_schedule'] != ['stabilizer', 'resolver']:
        raise ValueError('Accumulate one full stabilizer batch followed by one full resolver batch')
    if type(config['global_batch_size']) is not int or config['global_batch_size'] != 160:
        raise ValueError('Each objective requires a global batch of 160 target frames')
    weights = config.get('objective_weights', {})
    if (set(weights) != {'stabilizer', 'resolver'} or
            any(not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value) or value <= 0 for value in weights.values()) or
            (not full_gt and not math.isclose(sum(weights.values()), 1., rel_tol=0., abs_tol=1e-8))):
        raise ValueError('Choose explicit positive objective_weights for stabilizer and resolver before training; '
                         'the no-GT joint recipe additionally requires their sum to be one')
    if config['critic_updates'] < 1 or config['critic_warmup_steps'] < 0:
        raise ValueError('Invalid critic update/warmup count')
    if not 0 < config['ema_decay'] < 1:
        raise ValueError('EMA is required, with decay strictly between zero and one')
    save_interval = config['save_every_n_steps']
    if type(save_interval) is not int or save_interval < 1:
        raise ValueError('save_every_n_steps must be a positive integer number of total optimizer rounds')
    keep_last = config.get('checkpoint_keep_last', 8)
    if type(keep_last) is not int or keep_last < 1:
        raise ValueError('checkpoint_keep_last must be a positive integer')
    if config['eval_weights'] not in ('raw', 'both'):
        raise ValueError('eval_weights must be raw or both')
    if config['compile_mode'] == 'block_cudagraph':
        raise ValueError('Use block or parts compilation for the multi-forward joint autograd graph')
    if not 0 < config['res_t_min'] < config['res_t_max'] <= .2:
        raise ValueError('Resolver scoring must stay inside (0,.2]')
    if config.get('lora') or config.get('resume_weights_only'):
        raise ValueError('This trainer uses dense training and exact optimizer/EMA resume')
    if config.get('reference_checkpoint') not in (None, 'block', 'none'):
        raise ValueError('reference_checkpoint must be block or none when supplied')
    if full_gt:
        eval_interval = config.get('eval_every_n_steps', config['save_every_n_steps'])
        if type(eval_interval) is not int or eval_interval < 1:
            raise ValueError('eval_every_n_steps must be a positive integer')
        limit = config.get('max_student_steps')
        if limit is not None and (type(limit) is not int or limit < 1):
            raise ValueError('max_student_steps must be a positive integer when supplied')
        for key in ('full_gt_trim', 'full_gt_pool_trim'):
            if key in config and type(config[key]) is not bool:
                raise ValueError(f'{key} must be a boolean')
        for key in ('rollout_pool_batch_size_per_gpu',
                    'critic_stabilizer_micro_batch_size', 'critic_resolver_micro_batch_size',
                    'student_stabilizer_micro_batch_size', 'student_resolver_micro_batch_size'):
            value = config.get(key, config['micro_batch_size_per_gpu'])
            if type(value) is not int or value < 1:
                raise ValueError(f'{key} must be a positive integer')
        for role in ('critic', 'student'):
            for kind in ('stabilizer', 'resolver'):
                key = f'{role}_{kind}_checkpoint'
                if config.get(key, config['checkpoint']) not in ('block', 'none'):
                    raise ValueError(f'{key} must be block or none')


def main(experiment='joint'):
    if experiment not in ('joint', 'full_gt'):
        raise ValueError(f'Unknown distillation experiment: {experiment}')
    full_gt = experiment == 'full_gt'
    parser = argparse.ArgumentParser(description=(
        'Nine-reference single-target full-GT stabilizer and resolver DMD' if full_gt else __doc__))
    parser.add_argument('--config', required=True)
    parser.add_argument('--set', action='append', default=[])
    parser.add_argument('--benchmark', type=int, default=0,
                        help='total full joint updates, skipping critic-only warmup, saves and W&B')
    parser.add_argument('--benchmark-skip', type=int, default=2,
                        help='initial benchmark updates excluded from steady-state statistics')
    parser.add_argument('--expected-world-size', type=int,
                        help='fail before loading models if the distributed launch has a different size')
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    config.setdefault('checkpoint_keep_last', 8)
    validate_config(config, full_gt=full_gt)
    benchmark = args.benchmark > 0
    if not benchmark and (config.get('wandb', {}).get('enabled') is not True
                          or config['wandb'].get('mode') != 'online'):
        raise ValueError('Production requires wandb.enabled=true and wandb.mode=online')
    if args.benchmark < 0 or args.benchmark_skip < 0 or (benchmark and args.benchmark_skip >= args.benchmark):
        raise ValueError('Benchmark requires 0 <= benchmark-skip < benchmark')
    if benchmark and config.get('resume_from'):
        raise ValueError('A benchmark must not resume a production training run')
    local_rank = int(os.environ['LOCAL_RANK'])
    local_world = int(os.environ['LOCAL_WORLD_SIZE'])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    dist.init_process_group('nccl', device_id=device, timeout=timedelta(hours=2))
    saver.init_gloo()
    rank, world = dist.get_rank(), dist.get_world_size()
    if args.expected_world_size is not None and world != args.expected_world_size:
        raise ValueError(f'Expected {args.expected_world_size} GPU ranks, got {world}')
    if world % local_world:
        raise ValueError('Use homogeneous node GPU counts')
    torch.manual_seed(config['seed'] + rank)
    np.random.seed(config['seed'] + rank)
    random.seed(config['seed'] + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch._dynamo.config.cache_size_limit = config.get('compile_cache_size_limit', 64)
    torch._dynamo.config.accumulated_cache_size_limit = config.get('compile_accumulated_cache_size_limit', 256)
    torch._dynamo.config.fail_on_recompile_limit_hit = config.get('compile_fail_on_limit', full_gt)
    topology = [None] * world
    dist.all_gather_object(topology, {'rank': rank, 'host': socket.gethostname(),
        'local_rank': local_rank, 'gpu': torch.cuda.get_device_name(device),
        'total_memory_bytes': torch.cuda.get_device_properties(device).total_memory}, group=saver.gloo)

    bank_validation = None
    if config['refs'].get('require_validated_bank', False):
        bank_validation = root_action(lambda: validate_reference_bank(config['refs']['bank']))
        if rank == 0:
            print(json.dumps({'event': 'reference_bank_verified', **bank_validation}), flush=True)
    if full_gt:
        from distillation.full_gt_data import build_full_gt_data, regroup_full_gt_group
        from distillation.full_gt_dmd import (FullGTStabilizer,
            generate_resolver as generate_full_gt_resolver, score_resolver as score_full_gt_resolver)
        data = build_full_gt_data(config, rank, world)
        full_gt_rollout = FullGTStabilizer(config['band_steps'], config['history_t'],
            config['stabilizer_history_dropout'], trim=config.get('full_gt_trim', True))
        # Pool generation can use smaller causal layouts independently of the
        # differentiable student rollout's compilation/activation tradeoff.
        # Context/noise preparation remains shared and occurs exactly once.
        full_gt_pool_rollout = FullGTStabilizer(config['band_steps'], config['history_t'],
            config['stabilizer_history_dropout'],
            trim=config.get('full_gt_pool_trim', config.get('full_gt_trim', True)))
        objective_batch_sizes = {
            role: {kind: config.get(f'{role}_{kind}_micro_batch_size', config['micro_batch_size_per_gpu'])
                   for kind in config['objective_schedule']}
            for role in ('critic', 'student')}
        local_objective_rows = data.objective_microbatches * data.batch_rows
        for role, sizes in objective_batch_sizes.items():
            for kind, size in sizes.items():
                if local_objective_rows % size:
                    raise ValueError(f'{role} {kind} microbatch {size} does not divide '
                                     f'{local_objective_rows} local rows per objective')
        rollout_pool_rows = config.get('rollout_pool_batch_size_per_gpu', config['micro_batch_size_per_gpu'])
        total_pool_rows = local_objective_rows * config['critic_updates']
        if total_pool_rows % min(rollout_pool_rows, total_pool_rows):
            raise ValueError('Rollout pool batch size must divide the combined critic rows per objective')
    else:
        data = build_joint_data(config, rank, world)
    if bank_validation is not None:
        data.signature['reference_bank_validation'] = bank_validation
    signatures = [None] * world
    dist.all_gather_object(signatures, data.signature, group=saver.gloo)
    if any(value != signatures[0] for value in signatures):
        raise ValueError('All ranks must see the identical full corpus manifest and reference configuration')
    epoch_steps = len(data)
    student_steps = planned_student_steps(epoch_steps, config.get('max_student_steps') if full_gt else None)
    targets_per_row = 1 if full_gt else 4
    first = data.group(0)[0] if full_gt else data[0][0]
    latent_hw = tuple(first['latents_low'].shape[-2:])
    del first
    # Check the evaluation horizon before allocating three model replicas.
    if not benchmark:
        eval_targets = config['rollout_eval']['latent_frames'] - 7 if full_gt else config['sensitivity_frames']
        eval_index = config['rollout_eval']['batch_index'] if full_gt else 0
        root_action(lambda: len(data.eval_batch(eval_index, num_targets=eval_targets)['latents_low']))
    teacher_hash = root_action(lambda: file_sha256(config['model']['checkpoint_path']))
    resume = config.get('resume_from')
    # Keep model/data/objective/schedule identity exact; runtime settings such as
    # checkpoint retention may change without altering resumed training state.
    signature = checkpoint_compatibility({
                 'schema': ('full-gt-nine-ref-v1-single-target' if full_gt else 'joint-nine-ref-v2-paired-batches'),
                 'world_size': world, 'data': data.signature,
                 'teacher_sha256': teacher_hash, 'latent_hw': list(latent_hw), 'epoch_steps': epoch_steps,
                 'training': config})
    metadata = root_action(lambda: read_checkpoint_metadata(resume)) if resume else None
    resume_transition = validate_resume_compatibility(metadata['compatibility'], signature,
        allow_teacher_guidance_change=config.get('resume_allow_teacher_guidance_change', False)) if metadata else None

    mimi, template = load_mimi(config['model']['audio'], device)
    state = load_state_dict(config['model']['checkpoint_path'])
    nets = {}
    nets['teacher'] = build_net(state, config['model']['audio'], latent_hw, device, template, num_refs=9)
    nets['teacher'].eval().requires_grad_(False)
    if config['shard_teacher']:
        node_group, _ = dist.new_subgroups(group_size=local_world)
        nets['teacher'].shard_blocks(node_group)
    for role in ('student', 'critic'):
        nets[role] = build_net(state, config['model']['audio'], latent_hw, device, template, num_refs=9)
        nets[role].train().requires_grad_(True)
        if resume:
            load_model_weights(Path(resume), role, nets[role], True)
        for name, p in nets[role].original_named_parameters():
            p.original_name = name
    del state, template
    for role, net in nets.items():
        net.configure(config['compile_blocks'], 'none' if role == 'teacher' else config['checkpoint'],
                      config['compile_mode'])
        net.cache_references = config.get('cache_references', True)
        net.reference_checkpoint = config.get('reference_checkpoint')
    opt = config['optimizer']

    def inner(groups):
        return torch.optim.AdamW(groups, lr=opt['lr'], betas=tuple(opt['betas']),
                                 weight_decay=opt['weight_decay'], eps=opt['eps'], fused=True)

    optimizers = {role: Zero1FP32MasterAdamW([{'params': list(nets[role].parameters())}], inner)
                  for role in ('student', 'critic')}
    gradients = {role: GradFlats(optimizer) for role, optimizer in optimizers.items()}
    warmup = config['critic_warmup_steps']
    total_rounds = warmup + student_steps
    totals = {'student': student_steps, 'critic': total_rounds * config['critic_updates']}
    schedulers = {}
    for role, optimizer in optimizers.items():
        requested = config['lr_warmup_steps'] if role == 'student' else config['critic_lr_warmup_steps']
        # Tiny corpora still have a valid complete-epoch schedule.
        lr_warmup = min(requested, max(0, totals[role] - 2))
        schedulers[role] = torch.optim.lr_scheduler.LambdaLR(optimizer.inner,
            lambda step, total=totals[role], n=lr_warmup:
                learning_rate_factor(step, total, n, config['lr_final_ratio']))
    ema = EMA(config['ema_decay'], optimizers['student'].pairs)
    counters = {'rounds': 0, 'warmup_step': 0, 'epoch_step': 0,
                'student_updates': 0, 'critic_updates': 0, 'student_rows_seen': 0}
    if resume:
        counters = load_rank_state(Path(resume), optimizers, schedulers, ema, True, False)
        restored = [None] * world
        dist.all_gather_object(restored, counters, group=saver.gloo)
        if any(c != metadata['counters'] for c in restored):
            raise ValueError('Rank optimizer counters do not match completed checkpoint')
        if not (0 <= counters['epoch_step'] <= student_steps and 0 <= counters['warmup_step'] <= warmup
                and counters['rounds'] == counters['warmup_step'] + counters['epoch_step']
                and counters['student_updates'] == counters['epoch_step']
                and counters['critic_updates'] == counters['rounds'] * config['critic_updates']):
            raise ValueError('Checkpoint counters violate the saved phase schedule')
    training_rng = rng_state()

    def create_run():
        if resume:
            path = Path(resume).resolve().parent
            stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
            record = path / 'resumes' / stamp
            record.mkdir(parents=True, exist_ok=False)
            source = Path(__file__).resolve().parent
            manifest_path = source / 'source-manifest.json'
            manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
            transition_receipt = record_resume_transition(Path(resume), signature,
                expected_checkpoint_id=metadata['checkpoint_id'],
                allow_teacher_guidance_change=config.get('resume_allow_teacher_guidance_change', False)) \
                if resume_transition is not None else None
            write_json_atomic(record / 'config.json', config)
            write_json_atomic(record / 'runtime.json', {
                'benchmark': benchmark, 'arguments': vars(args), 'topology': topology,
                'torch_version': torch.__version__, 'resume_from': str(Path(resume).resolve()),
                'checkpoint_id': metadata['checkpoint_id'], 'restored_counters': dict(counters),
                'compatibility_transition': transition_receipt,
                'source_directory': str(source),
                'source_commit': manifest.get('remote_commit_verified', manifest.get('base_commit')),
                'source_manifest_sha256': file_sha256(manifest_path) if manifest else None,
                'compile_cache_size_limit': torch._dynamo.config.cache_size_limit,
                'compile_accumulated_cache_size_limit': torch._dynamo.config.accumulated_cache_size_limit})
            sync_directory(record)
            write_json_atomic(path / 'latest-resume.json', {
                'directory': str(record), 'config': str(record / 'config.json'),
                'runtime': str(record / 'runtime.json'), 'resume_from': str(Path(resume).resolve()),
                'config_sha256': file_sha256(record / 'config.json'),
                'runtime_sha256': file_sha256(record / 'runtime.json')})
            sync_directory(path)
        else:
            stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
            path = Path(config['output_dir']).resolve() / f'{"benchmark" if benchmark else "run"}-{stamp}'
            path.mkdir(parents=True, exist_ok=False)
            write_json_atomic(path / 'config.json', config)
            write_json_atomic(path / 'epoch-plan.json', {'signature': signature, 'totals': totals,
                'warmup_rounds': warmup, 'epoch_rounds': epoch_steps,
                'planned_student_steps': student_steps, 'total_rounds': total_rounds})
            write_json_atomic(path / 'runtime.json', {'benchmark': benchmark,
                'benchmark_updates': args.benchmark, 'benchmark_skip': args.benchmark_skip,
                'topology': topology, 'torch_version': torch.__version__,
                'compile_cache_size_limit': torch._dynamo.config.cache_size_limit,
                'compile_accumulated_cache_size_limit': torch._dynamo.config.accumulated_cache_size_limit})
        return str(path)

    run_dir = Path(root_action(create_run))
    if rank == 0:
        print(json.dumps({'event': 'run_created', 'path': str(run_dir), 'benchmark': benchmark,
                          'world_size': world, 'nodes': len({entry['host'] for entry in topology})}), flush=True)
    tracker = None
    if not benchmark:
        def initialize_tracking():
            nonlocal tracker
            start_tracking = start_training_tracking if full_gt else start_online_tracking
            tracker = start_tracking(config, run_dir, counters, epoch_steps=epoch_steps,
                                     world_size=world, resume_checkpoint=resume)
            return tracker.receipt
        tracking_receipt = root_tracking_action(initialize_tracking)
        if rank == 0:
            print(json.dumps({'event': 'wandb_initialized' if full_gt else 'wandb_online_verified',
                              **tracking_receipt}), flush=True)
    sums = Sums()
    training_metrics = TrainingMetrics(device) if full_gt else None
    phase_names = ('prepare', 'critic_pool', 'critic', 'critic_optimizer', 'student', 'student_optimizer',
                   *TIMING_PHASES)
    phase_events = {}
    step_logs = []
    previous_graphs = 0
    previous_compile_seconds = {'dynamo': 0., 'backward': 0.}

    @contextmanager
    def timed(name):
        if benchmark and rank == 0 and counters['rounds'] < 2 and name in (
                'critic_pool', 'critic', 'student', 'student_generation/stabilizer',
                'student_generation/resolver', 'student_backward/stabilizer', 'student_backward/resolver'):
            print(json.dumps({'event': 'benchmark_phase_started', 'phase': name,
                'round': counters['rounds'], 'unique_graphs': torch._dynamo.utils.counters['stats']['unique_graphs'],
                'at': datetime.now(timezone.utc).isoformat()}), flush=True)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            phase_events.setdefault(name, []).append((start, end))

    def prepare(raw, evaluation=False):
        rows = {'latents': raw['latents_low'].to(device, non_blocking=True).float(),
                'pose': raw['pose_latents_low'].to(device, non_blocking=True).float(),
                'ofi': raw['original_frame_indices'].to(device, non_blocking=True),
                'refs': raw['refs'].to(device, non_blocking=True).float(),
                'ref_keep': raw['ref_keep'].to(device, non_blocking=True), 'history_t': .2}
        batch = rows['latents'].shape[0]
        rows['drop'] = raw.get('bad_audio', torch.zeros(batch, dtype=torch.bool)).to(device)
        audio_config = config['model']['audio']
        codebooks = audio_config['num_codebooks'] if evaluation else int(torch.randint(
            audio_config['min_codebooks'], audio_config['num_codebooks'] + 1, ()))
        rows['audio'] = audio_latents(mimi, raw['codes'].to(device, non_blocking=True), codebooks)
        rows['weight'] = raw.get('sample_weight', torch.ones(batch)).to(device).float()
        rows['sample_weight'] = rows['weight']
        return rows

    def generate(rows, kind, critic_pool=False):
        if full_gt:
            phase = 'critic_pool' if critic_pool else 'student'
            context_phase = f'{phase}_context/stabilizer' if kind == 'stabilizer' else 'input_generation/resolver'
            with timed(context_phase):
                noise, context = full_gt_rollout.prepare_context(rows)
                keep = context['keep'] if kind == 'stabilizer' else context['keep'][:, -2:]
                weights = rows['weight'][:, None]
                sums.add(f'data/{kind}_history_slots_dropped',
                         ((~keep).float() * weights).sum(), weights.sum() * keep.shape[1])
            with timed(f'{phase}_forward/{kind}'):
                observer = None
                if not critic_pool and training_metrics is not None:
                    observer = lambda velocity: training_metrics.add_student_rollout(kind, velocity, rows['weight'])
                if kind == 'stabilizer':
                    rollout = full_gt_pool_rollout if critic_pool else full_gt_rollout
                    return rollout.generate(nets['student'], rows, noise=noise, context=context,
                                            observe_velocity=observer)
                return generate_full_gt_resolver(nets['student'], rows, context, noise=noise[:, :, -1:],
                                                 observe_velocity=observer)
        if kind == 'stabilizer':
            prefix = config['critic_prefix_frames'] if critic_pool else config['stabilizer_prefix_frames']
            timer = lambda stage: timed(f'{"critic_pool" if critic_pool else "student"}_{stage}/stabilizer')
            return generate_stabilizer(nets['student'], rows, band_steps=config['band_steps'],
                                       prefix_frames=prefix, history_dropout=config['stabilizer_history_dropout'],
                                       static_layout=config.get('static_stabilizer_layout', True), timed=timer)
        timer = lambda stage: timed('input_generation/resolver' if stage == 'input_generation' else
                                    f'{"critic_pool" if critic_pool else "student"}_forward/resolver')
        return generate_resolver(nets['student'], rows, timed=timer), None

    def score(net, rows, kind, x, times, context, teacher=False):
        if full_gt:
            if kind == 'stabilizer':
                return full_gt_rollout.score_pass(net, rows, x, times, context,
                    guidance_scale=config['teacher_guidance_scale'] if teacher else 1.)
            return score_full_gt_resolver(net, rows, x, times, context)
        if kind == 'stabilizer':
            return score_stabilizer(net, rows, x, times, context,
                guidance_scale=config['teacher_guidance_scale'] if teacher else 1.)
        return score_resolver(net, rows, x, times)

    def corruption(y, kind):
        batch = y.shape[0]
        if full_gt:
            low, high = ((.2 + 1e-6, 1. - 1e-6) if kind == 'stabilizer' else
                         (config['res_t_min'], config['res_t_max']))
            times = low + torch.rand((batch, 1), device=device) * (high - low)
            endpoint = .2 if kind == 'stabilizer' else 0.
        elif kind == 'stabilizer':
            times = stab_times(batch, device, .2 + 1e-6, 1. - 1e-6)
            endpoint = .2
        else:
            times = resolve_times(batch, device, config['res_t_min'], config['res_t_max'])
            endpoint = 0.
        noise = torch.randn_like(y)
        x, target = noised(y, endpoint, times, noise)
        return x, times, target, noise

    def backward(frames, rows, global_rows, kind, metric):
        # GradFlats averages rank gradients. Undo that average, then normalize
        # each objective over its own actual GLOBAL rows. Targets are averaged
        # per row; objective coefficients apply ONCE, independent of
        # microbatch count. Do not divide by two for the two accumulated modes.
        if global_rows <= 0:
            raise ValueError('Every paired update must contain valid rows for both objectives')
        row_losses = frames.mean(1)
        loss = (row_losses * rows['weight']).sum() * world / global_rows * config['objective_weights'][kind]
        loss.backward()
        weight = rows['weight'][:, None].expand_as(frames)
        sums.add(metric, (frames.detach() * weight).sum(), weight.sum())
        for j in range(frames.shape[1]):
            sums.add(f'{metric}/frame_{j}', (frames[:, j].detach() * rows['weight']).sum(), rows['weight'].sum())

    def update_impl(role):
        norm = gradients[role].reduce_clip(config['gradient_clipping'])
        if not torch.isfinite(norm):
            raise FloatingPointError(f'Nonfinite {role} gradient; refusing optimizer update')
        sums.add(f'{role}/grad_norm', norm, 1)
        sums.add(f'{role}/lr_used', optimizers[role].inner.param_groups[0]['lr'], 1)
        if training_metrics is not None:
            training_metrics.add_lr(role, optimizers[role].inner.param_groups[0]['lr'])
        optimizers[role].step()
        schedulers[role].step()
        gradients[role].release()
        counters[f'{role}_updates'] += 1
        if role == 'student':
            ema.update()

    def update(role):
        with timed(f'{role}_optimizer'):
            update_impl(role)

    def prepare_group(raw_group):
        if [raw['objective'] for raw in raw_group] != list(data.objective_schedule):
            raise ValueError('Data objective order changed')
        with timed('prepare'):
            prepared = [prepare(raw) for raw in raw_group]
            for rows, raw in zip(prepared, raw_group):
                if not full_gt and raw['objective'] == 'resolver':
                    add_resolver_history_dropout(sums, rows['ofi'], rows['weight'],
                                                 target_start=rows['latents'].shape[2] - 4)
        return prepared

    def generate_pool(raw_group, prepared):
        with timed('critic_pool'), torch.no_grad():
            return [generate(rows, raw['objective'], critic_pool=True) for raw, rows in zip(raw_group, prepared)]

    def regroup(raw_group, prepared, role):
        sizes = objective_batch_sizes[role]
        return (regroup_full_gt_group(raw_group, sizes),
                regroup_prepared_rows(raw_group, prepared, sizes))

    def checkpoint_for(role, kind):
        return config.get(f'{role}_{kind}_checkpoint', config['checkpoint']) if full_gt else config['checkpoint']

    def critic_update(raw_group, prepared, pool):
        with timed('critic'):
            gradients['critic'].allocate()
            for raw, rows, (y, context) in zip(raw_group, prepared, pool):
                kind = raw['objective']
                with objective_checkpoint(nets['critic'], checkpoint_for('critic', kind)):
                    x, times, target, noise = corruption(y, kind)
                    velocity = score(nets['critic'], rows, kind, x, times, context)
                    if full_gt or kind == 'stabilizer':
                        frames = endpoint_fm_loss(velocity, y, times, noise,
                                                  history_t=.2 if kind == 'stabilizer' else 0.)[1]
                    else:
                        frames = (
                            velocity.float() - target.float()).square().mean((1, 3, 4))
                    if training_metrics is not None:
                        training_metrics.add_fm(kind, frames, times, velocity, target, rows['weight'])
                    backward(frames, rows, int(raw['global_objective_rows'][kind]), kind, f'critic/{kind}')
                # Custom block-checkpoint closures remain reachable through
                # output grad_fn objects even after backward has consumed
                # their saved tensors. Release this objective's outputs
                # before constructing the next graph or the student rollout.
                del velocity, frames, x, times, target, noise, y, context
            update('critic')

    def train_round(raw_group, warming, critic_groups=None):
        kinds = [raw['objective'] for raw in raw_group]
        if kinds != list(data.objective_schedule):
            raise ValueError('Data objective order changed')
        global_rows = int(raw_group[0]['global_valid_rows'])
        if global_rows <= 0:
            raise ValueError('Empty optimizer accumulation group')
        if full_gt:
            if critic_groups is None or len(critic_groups) != config['critic_updates']:
                raise ValueError('Full-GT critic updates require distinct complete source batches')
            # The student is fixed across all critic steps. Generate each
            # objective's complete detached pool in larger batches, then keep
            # four distinct sequential critic updates and their denominators.
            # Prepare BEFORE regrouping to retain the original independent
            # audio-codebook draws for each source microbatch.
            critic_prepared = []
            critic_raw = []
            for critic_group in critic_groups:
                prepared = prepare_group(critic_group)
                grouped_raw, grouped_prepared = regroup(critic_group, prepared, 'critic')
                critic_raw.append(grouped_raw)
                critic_prepared.append(grouped_prepared)
            del prepared, grouped_raw, grouped_prepared
            with timed('critic_pool'):
                pools = generate_pooled_targets(
                    [[(raw['objective'], rows) for raw, rows in zip(group, prepared)]
                     for group, prepared in zip(critic_raw, critic_prepared)],
                    lambda rows, kind: generate(rows, kind, critic_pool=True), rollout_pool_rows)
            for index, (critic_group, prepared, pool) in enumerate(zip(critic_raw, critic_prepared, pools)):
                critic_update(critic_group, prepared, pool)
                # Drop the completed update's views before building its next graph.
                critic_prepared[index] = pools[index] = None
            del pool, prepared, pools, critic_prepared, critic_raw
            if warming:
                return
            prepared = prepare_group(raw_group)
            raw_group, prepared = regroup(raw_group, prepared, 'student')
            kinds = [raw['objective'] for raw in raw_group]
        else:
            # Preserve the no-GT recipe: one detached pool, reused for all
            # critic updates with fresh scoring noise and times.
            prepared = prepare_group(raw_group)
            pool = generate_pool(raw_group, prepared)
            for _ in range(config['critic_updates']):
                critic_update(raw_group, prepared, pool)
            del pool
        if not warming:
            with timed('student'):
                gradients['student'].allocate()
                for raw, rows, kind in zip(raw_group, prepared, kinds):
                    with objective_checkpoint(nets['student'], checkpoint_for('student', kind)):
                        with timed(f'student_generation/{kind}'):
                            y, context = generate(rows, kind)
                        with torch.no_grad(), timed(f'teacher_critic_scoring/{kind}'):
                            x, times, target, noise = corruption(y, kind)
                            teacher = score(nets['teacher'], rows, kind, x, times, context, teacher=True)
                            critic = score(nets['critic'], rows, kind, x, times, context)
                        frames = dmd_frames(y, teacher, critic, target, config['normalizer_epsilon'])
                        with timed(f'diagnostics/{kind}'):
                            add_dmd_diagnostics(sums, kind, y, teacher, critic, target, rows['weight'])
                            if training_metrics is not None:
                                training_metrics.add_dmd(kind, frames, times, teacher, critic,
                                                         target, rows['weight'], config['normalizer_epsilon'])
                        del teacher, critic, x, times, target, noise
                        with timed(f'student_backward/{kind}'):
                            backward(frames, rows, int(raw['global_objective_rows'][kind]), kind, f'student/{kind}')
                    # In a tuple assignment Python evaluates the next rollout
                    # before replacing y, so an explicit release avoids overlap
                    # with the preceding objective's completed autograd graph.
                    del y, context, frames
                update('student')
            counters['student_rows_seen'] += global_rows

    def evaluate(directory, identity=None):
        def evaluate_root(label):
            saved_rng = rng_state()
            try:
                if full_gt:
                    from distillation.training_evaluation import evaluate_checkpoint
                    options = config['rollout_eval']
                    raw = data.eval_batch(options['batch_index'], num_targets=options['latent_frames'] - 7)
                    raw = {key: value[:1] if isinstance(value, torch.Tensor) and value.ndim else value
                           for key, value in raw.items()}
                    clip = data.source.clips[int(raw['epoch_window_id'][0])]
                    raw['source_paths'] = {'video_latents': str(clip.video_path),
                                           'pose_latents': str(clip.pose_path), 'mimi_codes': str(clip.mimi_path)}
                    rows = prepare(raw, evaluation=True)
                    with torch.no_grad():
                        audio = mimi.decode(raw['codes'][:, :, :2 * options['latent_frames']].to(device)).audio_values
                        audio = audio[0, 0].float().cpu().numpy()
                    return evaluate_checkpoint(nets['student'], rows, raw, audio, config, directory,
                                               counters['rounds'], label, run_dir, tracker.receipt['url'],
                                               identity=identity)
                rows = prepare(data.eval_batch(0, num_targets=config['sensitivity_frames']), evaluation=True)
                result = run_sensitivity(nets['student'], rows, config)
                write_json_atomic(directory / f'sensitivity-{label}.json', result)
                return {'weights': label, 'file': f'sensitivity-{label}.json'}
            finally:
                restore_rng(saved_rng)
        reports = [root_action(lambda: evaluate_root('raw'))]
        if config['eval_weights'] == 'both':
            # All ranks participate in the ZeRO EMA swap, evaluation itself is
            # rank zero only and never enters frozen-teacher collectives.
            with ema.swapped_in(optimizers['student'], torch.bfloat16):
                reports.append(root_action(lambda: evaluate_root('ema')))
        return reports

    def evaluate_current_state():
        # This receipt identifies a transient in-memory model state, never a
        # resumable checkpoint. No weights, optimizer shards, or EMA are saved.
        saved_rng = rng_state()
        started = time.perf_counter()
        try:
            def reserve_evaluation():
                identity = {'evaluation_id': uuid.uuid4().hex, 'evaluation_kind': 'in_memory',
                    'checkpoint_id': None, 'student_updates': counters['student_updates'],
                    'rounds': counters['rounds'], 'counters': dict(counters),
                    'last_durable_checkpoint': str(last_checkpoint or resume) if last_checkpoint or resume else None,
                    'source_directory': str(Path(__file__).resolve().parent), 'compatibility': signature}
                directory = run_dir / 'evaluation-runs' / (
                    f'eval-student-{counters["student_updates"]:06d}-{identity["evaluation_id"]}')
                directory.mkdir(parents=True, exist_ok=False)
                write_json_atomic(directory / 'evaluation.identity.json', identity)
                return str(directory), identity
            directory, identity = root_action(reserve_evaluation)
            directory = Path(directory)
            probe_started = time.perf_counter()
            reports = evaluate(directory, identity=identity)
            probe_seconds = time.perf_counter() - probe_started
            completion = {**identity, 'reports': reports, 'evaluation_seconds': probe_seconds}
            root_action(lambda: write_json_atomic(directory / 'evaluation.complete.json', completion))
            tracking_started = time.perf_counter()
            root_tracking_action(lambda: tracker.evaluation(directory, counters))
            completion.update(tracking_seconds=time.perf_counter() - tracking_started,
                              total_seconds=time.perf_counter() - started)
            root_action(lambda: write_json_atomic(directory / 'evaluation.complete.json', completion))
            if rank == 0:
                print(json.dumps({'event': 'in_memory_evaluation_complete', 'path': str(directory),
                    'evaluation_id': identity['evaluation_id'], 'counters': dict(counters),
                    **{key: completion[key] for key in ('evaluation_seconds', 'tracking_seconds', 'total_seconds')}}),
                    flush=True)
        finally:
            restore_rng(saved_rng)

    def save_checkpoint():
        owned = sum(sum(m.numel() * m.element_size() for _, m in o.pairs) + tensor_bytes(o.inner.state)
                    for o in optimizers.values()) + sum(s.numel() * s.element_size() for _, _, s in ema.triples)
        owned = torch.tensor(owned, device=device, dtype=torch.int64)
        dist.all_reduce(owned)

        def reserve():
            weight_bytes = sum(p.numel() * p.element_size() for role in ('student', 'critic') for p in nets[role].parameters())
            ema_bytes = sum(p.numel() * 2 for p in nets['student'].parameters())
            required = int((int(owned) + weight_bytes + ema_bytes) * 1.15) + 1024**3
            if shutil.disk_usage(run_dir).free < required:
                raise RuntimeError(f'Need {required / 2**30:.1f} GiB free on rank-zero output filesystem before pruning old state')
            # A failed save leaves a distinct uncommitted directory. A replay
            # from the preceding checkpoint never overwrites or collides with it.
            pending = run_dir / f'.step-{counters["rounds"]:06d}-{uuid.uuid4().hex[:12]}.partial'
            pending.mkdir(exist_ok=False)
            return str(pending)

        directory = Path(root_action(reserve))

        def save_raw():
            for role in ('student', 'critic'):
                tensors = {name: p.detach().cpu().contiguous() for name, p in nets[role].original_named_parameters()}
                save_file(tensors, str(directory / f'{role}.safetensors'))

        root_action(save_raw)
        save_ema(directory, ema, True)
        root_action(lambda: sync_checkpoint_files(directory))
        # Each rank flushes its own shard on the shared output filesystem.
        save_rank_states(directory, optimizers, schedulers, ema, counters, False)
        root_action(lambda: complete_checkpoint(directory, {
            'world_size': world, 'compatibility': signature, 'counters': dict(counters),
            'optimizer_state_retained': True, 'critic_warmup_complete': counters['warmup_step'] >= warmup,
            'raw_weights': True, 'ema_decay': config['ema_decay'], 'lora': None}))

        def promote():
            final = directory.with_name(directory.name.removeprefix('.').removesuffix('.partial'))
            directory.rename(final)
            sync_directory(final)
            write_json_atomic(run_dir / 'latest.json', {'checkpoint': str(final), 'counters': dict(counters)})
            sync_directory(run_dir)
            return str(final)

        directory = Path(root_action(promote))
        # Model/optimizer checkpoint is durable even if evaluation is interrupted.
        # Enforce retention before evaluation: slow or failed probes must not
        # retain another optimizer generation or excess completed weight sets.
        retention = root_action(lambda: enforce_checkpoint_retention(
            directory, keep_last=config['checkpoint_keep_last']))
        if rank == 0:
            print(json.dumps({'event': 'checkpoint_complete', 'path': str(directory),
                              'counters': dict(counters), 'retention': retention}), flush=True)
        evaluation_started = time.perf_counter()
        reports = evaluate(directory)
        evaluation_seconds = time.perf_counter() - evaluation_started

        def finish():
            meta = json.loads((directory / 'metadata.json').read_text())
            # Keep the model-state completion manifest immutable. Evaluation
            # is a separate receipt pinned to that checkpoint generation.
            write_json_atomic(directory / 'evaluation.complete.json', {
                'checkpoint_id': meta['checkpoint_id'], 'reports': reports,
                'evaluation_seconds': evaluation_seconds})
            write_json_atomic(run_dir / 'latest.json', {'checkpoint': str(directory), 'counters': dict(counters)})
            return str(directory)

        saved = root_action(finish)
        tracking_started = time.perf_counter()
        root_tracking_action(lambda: tracker.evaluation(directory, counters))
        evaluation_timing = {'evaluation_seconds': evaluation_seconds,
            'tracking_seconds': time.perf_counter() - tracking_started,
            'total_seconds': time.perf_counter() - evaluation_started}
        def save_evaluation_timing():
            receipt = directory / 'evaluation.complete.json'
            write_json_atomic(receipt, {**json.loads(receipt.read_text()), **evaluation_timing})
        root_action(save_evaluation_timing)
        if rank == 0:
            print(json.dumps({'event': 'checkpoint_and_evaluation_complete', 'path': saved,
                              **evaluation_timing}), flush=True)
        return saved

    def record(warming, started, input_wait, group, start_memory, gc_seconds):
        nonlocal previous_graphs
        torch.cuda.synchronize()
        compiler = torch._dynamo.utils.counters
        unique_graphs = compiler['stats']['unique_graphs']
        compiler_timers = torch._dynamo.utils.compilation_time_metrics
        compile_seconds = {'dynamo': sum(compiler_timers.get('_compile.compile_inner', ())),
                           'backward': sum(compiler_timers.get('backward._backward_impl', ()))}
        observed = {'seconds/wall': time.monotonic() - started, 'seconds/input_wait': input_wait,
                    'seconds/gc': gc_seconds,
                    **{f'seconds/{name}': sum(start.elapsed_time(end) for start, end in
                       phase_events.get(name, ())) / 1000 for name in phase_names},
                    'memory/peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                    'memory/peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
                    'memory/allocated_start_gib': start_memory[0],
                    'memory/reserved_start_gib': start_memory[1],
                    'memory/allocated_end_gib': torch.cuda.memory_allocated() / 2**30,
                    'memory/reserved_end_gib': torch.cuda.memory_reserved() / 2**30,
                    'compile/unique_graphs': unique_graphs,
                    'compile/new_graphs': unique_graphs - previous_graphs,
                    'compile/graph_breaks': sum(compiler['graph_break'].values()),
                    'compile/frames_total': compiler['frames']['total'],
                    'compile/frames_ok': compiler['frames']['ok']}
        for name, value in compile_seconds.items():
            observed[f'compile/{name}_seconds_total'] = value
            observed[f'compile/new_{name}_seconds'] = value - previous_compile_seconds[name]
            previous_compile_seconds[name] = value
        observed = add_timing_aliases(observed)
        previous_graphs = unique_graphs
        values = torch.tensor(list(observed.values()), device=device, dtype=torch.float64)
        dist.all_reduce(values, op=dist.ReduceOp.MAX)
        observed = dict(zip(observed, values.cpu().tolist()))
        metrics = finalize_diagnostics(sums.report(), learning_rates={
            role: optimizer.inner.param_groups[0]['lr'] for role, optimizer in optimizers.items()},
            warming=warming)
        for role in ('critic', 'student'):
            if all(f'{role}/{kind}' in metrics for kind in config['objective_schedule']):
                metrics[f'{role}/loss'] = sum(config['objective_weights'][kind] * metrics[f'{role}/{kind}']
                                            for kind in config['objective_schedule'])
        objective_rows = {k: int(v) for k, v in group[0]['global_objective_rows'].items()}
        log = {**counters, 'phase': 'critic_warmup' if warming else 'joint', 'benchmark': benchmark,
               **observed,
               'epoch_steps': epoch_steps, 'epoch_fraction': counters['epoch_step'] / epoch_steps,
               'planned_student_steps': student_steps,
               'training_fraction': counters['student_updates'] / student_steps,
               'global_valid_rows': int(group[0]['global_valid_rows']),
               'global_objective_rows': objective_rows,
               'global_objective_targets': {k: targets_per_row * v for k, v in objective_rows.items()},
               'objective_weights': config['objective_weights'],
               **metrics}
        if training_metrics is not None:
            online, accumulators = training_metrics.report(config['objective_weights'])
            log.update(training_metrics=online, metric_sums=accumulators)
            training_metrics.reset()
        if rank == 0:
            print(json.dumps(log), flush=True)
            with (run_dir / 'metrics.jsonl').open('a') as handle:
                handle.write(json.dumps(log) + '\n')
        if not benchmark:
            root_tracking_action(lambda: tracker.training(log))
        if benchmark:
            step_logs.append(log)

    def loader(indices):
        kwargs = dict(batch_size=None, sampler=indices, num_workers=config['num_dataloader_workers'],
                      pin_memory=True, generator=torch.Generator().manual_seed(config['seed']))
        if config['num_dataloader_workers']:
            kwargs.update(prefetch_factor=config['prefetch_factor'])
        return DataLoader(data, **kwargs)

    if resume:
        retention = root_action(lambda: enforce_checkpoint_retention(
            Path(resume), keep_last=config['checkpoint_keep_last']))
        if rank == 0:
            print(json.dumps({'event': 'checkpoint_retention_on_resume', 'path': str(resume),
                              'retention': retention}), flush=True)
    if resume and not root_action(lambda: (Path(resume) / 'evaluation.complete.json').exists()):
        reports = evaluate(Path(resume))
        root_action(lambda: write_json_atomic(Path(resume) / 'evaluation.complete.json', {
            'checkpoint_id': metadata['checkpoint_id'], 'reports': reports}))
        recovered_tracking = root_tracking_action(lambda: tracker.evaluation(Path(resume), counters, startup=True))
        if rank == 0:
            print(json.dumps({'event': 'checkpoint_evaluation_recovered', 'path': str(resume),
                              'counters': dict(counters), 'checkpoint_id': metadata['checkpoint_id'],
                              'tracking': recovered_tracking}), flush=True)
    # Tracking startup, remote verification and resumed evaluation must not
    # perturb the restored training RNG stream.
    restore_rng(training_rng)
    # Freeze long-lived model/compiler objects and collect only at common step
    # boundaries. Mid-backward Python GC pauses otherwise stall every GPU at
    # the next gradient collective (see the measured five-node DMD runs).
    gc.collect()
    gc.freeze()
    gc.disable()
    dist.barrier()
    # Critic-only warmup never advances the production epoch cursor. Benchmark
    # warmup instead exercises the full joint round, including student backward.
    warm_indices, joint_indices = training_indices(epoch_steps, counters, warmup, args.benchmark,
        max_student_steps=config.get('max_student_steps') if full_gt else None)
    stopped = False
    last_checkpoint = None
    for warming, indices in ((True, warm_indices), (False, joint_indices)):
        if not len(indices):
            continue
        batches = iter(loader(indices))
        for _ in range(len(indices)):
            phase_events.clear()
            torch.cuda.reset_peak_memory_stats()
            start_memory = (torch.cuda.memory_allocated() / 2**30,
                            torch.cuda.memory_reserved() / 2**30)
            started = time.monotonic()
            batch = next(batches)
            group = batch['student'] if full_gt else batch
            input_wait = time.monotonic() - started
            with timed('outer'):
                train_round(group, warming, batch['critic'] if full_gt else None)
            counters['warmup_step' if warming else 'epoch_step'] += 1
            counters['rounds'] += 1
            # With automatic GC disabled, collect at the same boundary on every
            # rank. In particular, never freeze before collecting: unreachable
            # tensor/autograd cycles would otherwise become permanently frozen.
            collected_at = time.monotonic()
            gc.collect()
            phase_step = counters['warmup_step' if warming else 'epoch_step']
            if phase_step <= 4:
                gc.freeze()
            gc_seconds = time.monotonic() - collected_at
            record(warming, started, input_wait, group, start_memory, gc_seconds)
            if benchmark:
                continue
            request = root_action(lambda: claim_save_request(run_dir))
            due = (counters['rounds'] % config['save_every_n_steps'] == 0 or
                   (not warming and counters['epoch_step'] == epoch_steps))
            if full_gt:
                due = full_gt_checkpoint_due(counters, config['save_every_n_steps'], student_steps, warming=warming)
            if due or request['save']:
                last_checkpoint = save_checkpoint()
                root_action(lambda: acknowledge_save_request(run_dir, request))
            elif full_gt and full_gt_evaluation_due(counters,
                    config.get('eval_every_n_steps', config['save_every_n_steps']), warming=warming):
                evaluate_current_state()
            if request['quit']:
                stopped = True
                break
        del batches
        if stopped:
            break
    if benchmark:
        local_summary = {'rank': rank, 'host': socket.gethostname(),
            'unique_graphs': torch._dynamo.utils.counters['stats']['unique_graphs'],
            'graph_breaks': sum(torch._dynamo.utils.counters['graph_break'].values()),
            'memory_allocated_gib': torch.cuda.memory_allocated() / 2**30,
            'memory_reserved_gib': torch.cuda.memory_reserved() / 2**30}
        ranks = [None] * world
        dist.all_gather_object(ranks, local_summary, group=saver.gloo)

        def finish_benchmark():
            measured = step_logs[args.benchmark_skip:]
            timings = {}
            for key in measured[0]:
                if key.startswith('seconds/'):
                    values = [entry[key] for entry in measured]
                    timings[key] = {'median': float(np.median(values)), 'mean': float(np.mean(values)),
                                    'min': min(values), 'max': max(values), 'p90': float(np.percentile(values, 90))}
            result = {'benchmark': True, 'complete': True, 'world_size': world,
                'nodes': len({entry['host'] for entry in topology}), 'updates': args.benchmark,
                'skipped_updates': args.benchmark_skip, 'measured_updates': len(measured),
                'steps': [entry['seconds/wall'] for entry in measured], 'timings': timings,
                'median': timings['seconds/wall']['median'],
                'target_frames_per_second_per_objective': config['global_batch_size'] / timings['seconds/wall']['mean'],
                'peak_allocated_gib': max(entry['memory/peak_allocated_gib'] for entry in measured),
                'peak_reserved_gib': max(entry['memory/peak_reserved_gib'] for entry in measured),
                'measured_new_graphs': sum(entry['compile/new_graphs'] for entry in measured),
                'ranks': ranks, 'path': str(run_dir), 'counters': counters,
                'completed_at_utc': datetime.now(timezone.utc).isoformat()}
            write_json_atomic(run_dir / 'benchmark.complete.json', result)
            print(json.dumps(result), flush=True)
        root_action(finish_benchmark)
    else:
        complete = counters['warmup_step'] == warmup and counters['epoch_step'] == student_steps
        root_action(lambda: write_json_atomic(run_dir / ('training.complete.json' if complete else 'training.paused.json'), {
            'complete': complete, 'counters': counters, 'epoch_steps': epoch_steps,
            'planned_student_steps': student_steps,
            'reason': ('student_limit_complete' if student_steps < epoch_steps else 'epoch_complete') if complete else 'save_q',
            'checkpoint': last_checkpoint,
            'completed_at_utc': datetime.now(timezone.utc).isoformat()}))
    if not benchmark:
        root_tracking_action(lambda: tracker.finish())
    gc.enable()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
