"""Bounded GPU check: original single partner versus the fanout scheduler.

Both executions reuse identical weights, rows, per-row sampling seeds and actor
cache shapes. This checks numerical/causal equivalence, not training-update fit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .teacher_partner_benchmark import (
    _audit_result, _build_engine_pair, _build_model, _release_engines,
    _storage_identity, _sync, _write_json, memory_report,
)


def compare_results(reference, candidate):
    """Compare all generated lanes, behavior probabilities and heard inputs."""
    import torch
    checks = []
    def compare(name, left, right):
        if isinstance(left, torch.Tensor):
            left = left.detach().float().cpu().numpy()
            right = right.detach().float().cpu().numpy()
        if isinstance(left, dict):
            if left.keys() != right.keys():
                checks.append({'name': name, 'equal': False, 'reason': 'different keys'})
                return
            for key in left:
                compare(f'{name}/{key}', left[key], right[key])
            return
        if isinstance(left, (list, tuple)):
            if len(left) != len(right):
                checks.append({'name': name, 'equal': False, 'reason': 'different lengths'})
                return
            for index, (a, b) in enumerate(zip(left, right, strict=True)):
                compare(f'{name}/{index}', a, b)
            return
        a, b = np.asarray(left), np.asarray(right)
        same = a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
        result = {'name': name, 'equal': bool(same), 'shape': list(a.shape)}
        if not same and a.shape == b.shape:
            matches = (a == b) | (np.isnan(a) & np.isnan(b))
            result['different_values'] = int((~matches).sum())
            result['first_difference'] = np.argwhere(~matches)[0].tolist()
            finite = np.isfinite(a) & np.isfinite(b)
            result['maximum_finite_absolute_difference'] = float(
                np.max(np.abs(a[finite].astype(np.float64) - b[finite].astype(np.float64)), initial=0.))
        checks.append(result)
    for index, (left, right) in enumerate(zip(reference['rollouts'], candidate['rollouts'], strict=True)):
        for key in ('text', 'audio', 'pose', 'sampled_masks', 'behavior_logprobs'):
            compare(f'row{index}/{key}', getattr(left, key), getattr(right, key))
    for key in ('waveforms', 'latents', 'partner_rollouts'):
        compare(key, reference[key], candidate[key])
    return {'exact': all(check['equal'] for check in checks), 'checks': len(checks),
            'differences': [check for check in checks if not check['equal']]}


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--instances', type=int, default=16)
    parser.add_argument('--frames', type=int, default=375)
    parser.add_argument('--capacity-frames', type=int, default=1500)
    parser.add_argument('--fast-state', action='store_true',
                        help='Also validate optional natural partner-state finalization')
    parser.add_argument('--items-json', type=Path,
                        help='Previously admitted benchmark item list, avoiding another full dataset scan')
    parser.add_argument('--qwen32-config', default='/home/param/models/imported-weights/ppm-811-17850-base-bf16/configs/Qwen3-32B')
    parser.add_argument('--qwen4-config', default='/home/param/models/imported-weights/ppm-811-17850-base-bf16/configs/Qwen3-4B')
    parser.add_argument('--voxtral', default='/home/param/models/imported-weights/aloe-ft-vox17127-nopersona/assets/voxtral-realtime')
    parser.add_argument('--mimi', default='/home/param/models/mimi')
    args = parser.parse_args(argv)
    if not 1 <= args.frames <= args.capacity_frames or args.instances < 1:
        parser.error('Positive batch/length and sufficient capacity required')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'result.json').exists():
        raise ValueError('Use a fresh output directory')
    def log(event, **fields):
        record = {'event': event, 'time': time.time(), **fields}
        line = json.dumps(record, default=str)
        print(line, flush=True)
        with (args.output / 'events.jsonl').open('a') as stream:
            stream.write(line + '\n')
    from .config import TrainingConfig
    from .data import ProductionRows, prefix_seed
    from .train import checkpoint_spec, load_actor_native, rollout_seed, source_digest
    from .rollout import make_fastppm_class
    from .partner_rollout import rollout_teacher_partner
    from .partner_fanout import rollout_teacher_partners
    from . import runtime as R
    import torch
    cfg = TrainingConfig.from_dict(json.loads(args.config.read_text()))
    _write_json(args.output / 'identity.json', {'config': cfg.to_dict(), 'arguments': vars(args),
        'source_sha256': source_digest(), 'benchmark_only': True, 'GPU_training_fit_validated': False})
    data = ProductionRows(cfg)
    if args.items_json is None:
        items = data.items()[:args.instances]
    else:
        selected = json.loads(args.items_json.read_text())
        items = (selected['selected_items'] if isinstance(selected, dict) else selected)[:args.instances]
    if len(items) != args.instances:
        raise ValueError('Not enough admitted items for the requested batch')
    rows = [prefix_seed(data.load(item, rank=0, update=0, slot=slot), args.frames)
            for slot, item in enumerate(items)]
    seeds = [rollout_seed(cfg.seed, 0, 0, slot) for slot in range(len(rows))]
    log('data_ready', instances=len(rows), frames=args.frames, selected_items=items,
        prefills=[int(row['prefill_frames']) for row in rows])
    torch.set_num_threads(1)
    torch.manual_seed(cfg.seed)
    M = R.load_native()
    M.QWEN32, M.QWEN4B_CFG = args.qwen32_config, args.qwen4_config
    native = load_actor_native(M)
    specs = {'student': checkpoint_spec(cfg.student, 'student'),
        'partner': checkpoint_spec(cfg.resolved_partner_checkpoint, 'audio_teacher',
                                   gain_policy=cfg.audio_teacher_gain_policy)}
    devices = tuple(torch.device(f'cuda:{index}') for index in range(4))
    prefix = cfg.fixed_sequence_frames - cfg.conversation_prefix_frames - cfg.speech_delay
    loaded, engines = {}, {}
    for role, pair in (('student', devices[:2]), ('partner', devices[2:])):
        has_pose = role == 'student'
        log('model_build_begin', role=role)
        model, placement, cfgs = _build_model(torch, M, specs[role], pair,
            backbone_cut=cfg.actor_backbone_cut if has_pose else cfg.partner_actor_backbone_cut,
            require_pose=has_pose, log=log)
        loaded[role] = {'model': model, 'layer_devices': placement, 'cfgs': cfgs,
            'devices': pair, 'has_pose': has_pose, 'prefix_capacity': prefix,
            'actor_type': make_fastppm_class(native, require_pose=has_pose),
            'initial_hook_ids': set(model.inp._forward_hooks), 'storage_identity': _storage_identity(model)}
    from voxtral_front import VoxtralFront
    fronts = {role: VoxtralFront(args.voxtral, device=value['devices'][0]) for role, value in loaded.items()}
    try:
        engines, setup = _build_engine_pair(torch, cfg, loaded, instances=args.instances,
            capacity=args.capacity_frames, graphs=True, log=log)
        log('reference_begin')
        reference = rollout_teacher_partner(engines['student'], engines['partner'], rows, seeds,
            fronts['student'], fronts['partner'], args.mimi, graphs=True, teacher_replay='block',
            delivery_delay_samples=cfg.partner_audio_delay_samples)
        log('reference_complete', timings=reference['timings'], audit=_audit_result(reference, rows))
        if args.fast_state:
            from .partner_faststate import disable_partner_text_history
            log('fast_state_enabled', **disable_partner_text_history(engines['partner']))
        log('fanout_begin')
        candidate = rollout_teacher_partners(engines['student'], [engines['partner']], rows, seeds,
            fronts['student'], [fronts['partner']], args.mimi, graphs=True, teacher_replay='block',
            capacity_frames=args.capacity_frames, profile_calls=0,
            delivery_delay_samples=cfg.partner_audio_delay_samples)
        log('fanout_complete', timings=candidate['timings'], audit=_audit_result(candidate, rows))
        comparison = compare_results(reference, candidate)
        _sync(torch, devices)
        result = {'comparison': comparison, 'setup': setup, 'memory': memory_report(torch, devices),
            'reference_timings': reference['timings'], 'fanout_timings': candidate['timings'],
            'student_rows': len(rows), 'frames': args.frames, 'cache_capacity_frames': args.capacity_frames}
        _write_json(args.output / 'result.json', result)
        log('comparison_complete', **comparison)
        if not comparison['exact']:
            raise RuntimeError('Single-partner fanout differs from original execution; inspect result.json')
    finally:
        _release_engines(torch, engines, loaded, devices)


if __name__ == '__main__':
    main()
