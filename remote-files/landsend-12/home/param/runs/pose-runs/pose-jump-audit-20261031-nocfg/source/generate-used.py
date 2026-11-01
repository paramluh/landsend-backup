#!/usr/bin/env python3
"""Offline, resumable pose jump audit using the deployed, unchanged FastPPM engine.

Example (one of eight disjoint GPU-pair workers; run on the allocated nodes):
  CUDA_VISIBLE_DEVICES=0,1 MIMI_DIR=/home/param/models/mimi \
  /home/param/pose-repo/.venv/bin/python tools/pose_jump_audit/generate.py \
    --root /home/param/runs/pose-runs/pose-jump-audit-20261031 \
    --reference-request /home/param/runs/pose-runs/pose-jump-audit-20261031/reference-request.json \
    --shard-index 0 --shard-count 8 --batch-size 16

Each shard owns whole fixed-size batches (batch_index % shard_count). A batch
always keeps its seed and engine row assignments, including partial resumes.
The site worker's prompt/forcing preparation is reused but its audio/video
publication is never entered. Only tokens, words and provenance are saved.
Exactly 60s includes 66 prefill token frames (5.28s), with the original UI's
prefill_s=5.12 retained: clip_s=(750-66)/12.5=54.72 generated seconds.

No inference benchmark, video job, web queue, or live-serving state is created.
SIGTERM/SIGINT and ROOT/STOP drain the current batch before stopping.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import copy
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import socket
import sys
import threading
import time
import traceback

ENGINE_SHA256 = '882ed238c83d5e6dab961a6bcb6399b9b392e12b1e4beed4b3ddde98f1084c8b'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}-{threading.get_ident()}')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--reference-request', type=Path, required=True)
    parser.add_argument('--site', type=Path, default=Path('/home/param/websites/pose-playground'))
    parser.add_argument('--engine', type=Path, default=Path('/home/param/websites/pose-monoplex-engine'))
    parser.add_argument('--people', type=Path, default=Path('/home/param/datasets/pose-demo/prompts'))
    parser.add_argument('--checkpoint', type=Path, default=Path('/home/param/models/imported-weights/ppm-aloe-monoplex-pose-raw-bandfix-355/weights/export-355'))
    parser.add_argument('--total', type=int, default=1000)
    parser.add_argument('--duration', type=float, default=60.0, help='Total output seconds, including prefill')
    parser.add_argument('--batch-size', type=int, choices=(1, 4, 8, 16, 32, 64), default=16)
    parser.add_argument('--pose-cfg-off', action='store_true', help='Explicit user-requested experiment delta: disable pose CFG and omit unconditional engine rows')
    parser.add_argument('--shard-index', type=int, required=True)
    parser.add_argument('--shard-count', type=int, required=True)
    parser.add_argument('--seed-base', type=int, default=8018011)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--plan-only', action='store_true', help='Validate and emit the plan without importing or initializing CUDA')
    parser.add_argument('--prepare-only', action='store_true', help='Validate the exact production prompt/forcing preparation without loading model weights or allocating CUDA tensors')
    args = parser.parse_args()
    if not (0 <= args.shard_index < args.shard_count) or args.total < 1:
        parser.error('Require 0 <= shard-index < shard-count and total >= 1')
    if args.duration != 60:
        parser.error('This reference audit is fixed at exactly 60 seconds')
    args.root = args.root.resolve()
    args.root.mkdir(parents=True, exist_ok=True)
    shard_root = args.root / 'shards' / f'{args.shard_index:02d}'
    shard_root.mkdir(parents=True, exist_ok=True)
    lock = (shard_root/'worker.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f'Shard {args.shard_index} already has a live process holding its lock')
    source = json.loads(args.reference_request.read_text())
    if source.get('model') != 'monoplex' or source.get('prefill', {}).get('src') != 'clip':
        raise ValueError('This audit requires the original monoplex actor-clip reference request')
    if source.get('params', {}).get('pose_only') or source.get('prefill', {}).get('person') != 'heldout_02':
        raise ValueError('Unexpected reference conditioning; expected v1 actor heldout_02 with generated speech')
    frames = round(args.duration * 12.5)
    prefill_frames = 2*round(float(source['prefill']['s'])/.16)+2
    clip_s = (frames-prefill_frames)/12.5
    assigned = list(range(args.shard_index, math.ceil(args.total/args.batch_size), args.shard_count))
    engine_hash = file_digest(args.engine/'ppm/fast_ppm.py')
    if engine_hash != ENGINE_SHA256:
        raise ValueError(f'Engine changed from the verified deployed HF355 implementation: {engine_hash}')
    fingerprint_fields = {
        'reference_request': source,
        'checkpoint': str(args.checkpoint.resolve()),
        'engine_sha256': engine_hash,
        'worker_source_sha256': file_digest(args.site/'worker.py'),
        'total': args.total, 'total_seconds': args.duration, 'frames': frames,
        'batch_size': args.batch_size, 'seed_base': args.seed_base,
        'prefill_frames': prefill_frames, 'clip_s': clip_s,
    }
    if args.pose_cfg_off:
        fingerprint_fields.update(parameter_delta={'pose_cfg_on': {'reference': source['params']['pose_cfg_on'], 'experiment': 0}},
                                  cfg_rows=False, static_max_s=args.duration, static_prefix='exact prepared P')
    fingerprint = digest(fingerprint_fields)
    plan = dict(fingerprint_fields, fingerprint=fingerprint, shard_index=args.shard_index,
                shard_count=args.shard_count, batch_indices=assigned,
                sample_indices=[i for b in assigned for i in range(b*args.batch_size, min(args.total, (b+1)*args.batch_size))],
                seed_mode='batch_seed_and_row', audio_decoding=False, video_rendering=False,
                run_directory='runs/rollout-{sample_index:06d}')
    previous_plan = shard_root/'plan.json'
    if previous_plan.is_file() and json.loads(previous_plan.read_text()).get('fingerprint') != fingerprint:
        raise ValueError('Existing shard plan differs: use a separate root for a changed experiment')
    write_json(previous_plan, plan)
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return
    os.environ.setdefault('GENWEB_MONO_ENGINE', str(args.engine))
    os.environ.setdefault('GENWEB_PEOPLE', str(args.people))
    os.environ.setdefault('GENWEB_ACTORS', str(args.people))
    os.environ.setdefault('MIMI_DIR', '/home/param/models/mimi')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('PPM_CD', '2')
    sys.path.insert(0, str(args.site))
    state = {'state': 'initializing', 'host': socket.gethostname(), 'pid': os.getpid(),
             'shard_index': args.shard_index, 'shard_count': args.shard_count,
             'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
             'started': time.time(), 'fingerprint': fingerprint, 'completed_samples': 0,
             'assigned_samples': len(plan['sample_indices'])}
    halt = threading.Event()
    final = threading.Event()
    original_hook = sys.excepthook
    def report_unhandled(kind, exc, tb):
        final.set()
        state.update(state='failed', error=f'{kind.__name__}: {exc}',
                     traceback=''.join(traceback.format_exception(kind, exc, tb)), finished=time.time())
        write_json(shard_root/'status.json', dict(state, heartbeat=time.time()))
        original_hook(kind, exc, tb)
    sys.excepthook = report_unhandled
    signal.signal(signal.SIGTERM, lambda *_: halt.set())
    signal.signal(signal.SIGINT, lambda *_: halt.set())
    def heartbeat():
        while not final.wait(5):
            write_json(shard_root/'status.json', dict(state, heartbeat=time.time()))
    threading.Thread(target=heartbeat, daemon=True).start()
    write_json(shard_root/'status.json', dict(state, heartbeat=time.time()))
    spec = importlib.util.spec_from_file_location('audit_site_worker', args.site/'worker.py')
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    worker.WID = f'audit-{socket.gethostname()}-{args.shard_index}'
    worker.CK = str(args.checkpoint)
    np, torch, FP = worker.np, worker.torch, worker.FP
    torch.set_num_threads(args.threads)
    # Reuse the production preparation generator without entering its publishing
    # continuation (which would decode audio and enqueue full-video rendering).
    wk = worker.Worker.__new__(worker.Worker)
    wk.pp = [{'id': source['prefill']['person'], 'label': source['prefill']['person']}]
    request = copy.deepcopy(source)
    request.update(run_id=f'audit-preparation-{args.shard_index}', seed=args.seed_base,
                   clip_s=clip_s, stream=False, no_live=True, gt=False)
    if args.pose_cfg_off:
        request['params']['pose_cfg_on'] = 0
    request.pop('batch', None)
    prep_dir = shard_root/'preparation'
    prep_dir.mkdir(exist_ok=True)
    parts = wk.run_parts(str(prep_dir), request)
    ctx = next(parts)
    parts.close()
    if ctx['n'] != frames or ctx['N'] != prefill_frames:
        raise ValueError(f'Unexpected prepared timeline n={ctx["n"]}, N={ctx["N"]}')
    # Original v1 parameters must survive the production validation unchanged.
    for name, value in source['params'].items():
        expected = 0 if args.pose_cfg_off and name == 'pose_cfg_on' else value
        if name in FP.PARAMS and ctx['params'][name] != expected:
            raise ValueError(f'Original parameter was changed: {name}')
    prepared = {'P': ctx['P'], 'frames': ctx['n'], 'prefill_frames': ctx['N'],
                'params': ctx['params'], 'params_sha256': digest(ctx['params']),
                'prompt_sha256': hashlib.sha256(b''.join(np.ascontiguousarray(ctx['x'][k]).tobytes() for k in ('ids', 'depth', 'present'))).hexdigest(),
                'forcing': {k: None if ctx[k] is None else {'shape': list(ctx[k].shape), 'sha256': hashlib.sha256(ctx[k].tobytes()).hexdigest()}
                            for k in ('pp', 'pa', 'lane_ids')}}
    write_json(shard_root/'prepared.json', prepared)
    if args.prepare_only:
        final.set()
        state.update(state='prepared', finished=time.time())
        write_json(shard_root/'status.json', dict(state, heartbeat=time.time()))
        print(json.dumps(prepared, indent=2), flush=True)
        return
    if torch.cuda.device_count() != 2:
        raise ValueError('Expose exactly two exclusively assigned GPUs per worker')
    def run_dir(index):
        return args.root/'runs'/f'rollout-{index:06d}'
    def complete(index):
        directory = run_dir(index)
        try:
            summary = json.loads((directory/'summary.json').read_text())
            if summary['fingerprint'] != fingerprint or summary['sample_index'] != index:
                return False
            if json.loads((directory/'status.json').read_text())['state'] != 'done':
                return False
            with np.load(directory/'tokens.npz', allow_pickle=False) as saved:
                return all(saved[k].shape == (frames, 32) and saved[k].min() >= 0 and saved[k].max() < FP.CBS for k in ('pose', 'mimi')) and saved['lane'].shape == (frames,)
        except (OSError, ValueError, KeyError):
            return False
    already = {i for i in plan['sample_indices'] if complete(i)}
    state.update(completed_samples=len(already))
    if len(already) == len(plan['sample_indices']):
        state.update(state='done', finished=time.time(), heartbeat=time.time())
        final.set()
        write_json(shard_root/'status.json', state)
        print(json.dumps(state), flush=True)
        return
    devs = [torch.device('cuda:0'), torch.device('cuda:1')]
    torch.cuda.set_device(devs[-1])
    with torch.inference_mode():
        state.update(state='loading_model')
        m, ldev, cfgs, binfo = FP.build_model(worker.CK, devs)
        if not binfo['pose'] or not binfo['decp_bb'] or binfo['step'] != 355:
            raise ValueError(f'Expected exact pose mono355 checkpoint: {binfo}')
        wk.step = binfo['step']
        wk.max_s = args.duration if args.pose_cfg_off else 64.0
        if args.pose_cfg_off:
            wk.silence = None
        else:
            wk.mimi = worker.IO.Mimi(devs[-1])
            silence_frames = 2048 + round(wk.max_s*12.5)+8
            wk.silence = wk.mimi.encode(np.zeros(silence_frames*worker.SPF, np.float32))[:, :silence_frames]
        wk.model_args = (m, ldev, cfgs, devs)
        wk.engines = OrderedDict()
        wk.eng = None
        prefix = ctx['P'] if args.pose_cfg_off else max(512, math.ceil(ctx['P']/512)*512)
        state.update(state='capturing', engine_shape=[args.batch_size, wk.max_s, prefix])
        if args.pose_cfg_off:
            # Existing FastPPM's native non-CFG mode, with one constant shape.
            # No unconditional rows or Mimi silence codec are needed at all.
            length = prefix + round(wk.max_s*FP.FPS) + 8 + worker.CD + 8
            c32, c4, _ = cfgs
            memory_estimate = []
            for device in devs:
                trunk = sum(d == device for d in ldev)*args.batch_size*c32.num_key_value_heads*c32.head_dim
                towers = args.batch_size*(len(m.tower_p.layers) if device == devs[0] else len(m.tower_s.layers))*c4.num_key_value_heads*c4.head_dim
                need = 4*length*(trunk+towers)+(6 << 30)
                free, total = torch.cuda.mem_get_info(device)
                memory_estimate.append({'device': str(device), 'estimated_graph_and_workspace_bytes': need,
                                        'free_before_capture': free, 'total': total})
                if free < need:
                    raise RuntimeError(f'Insufficient free memory for fixed non-CFG batch: {memory_estimate[-1]}')
            write_json(shard_root/'memory-plan.json', {'engine_shape': state['engine_shape'], 'cfg_rows': False, 'devices': memory_estimate})
            eng = FP.FastPPM(m, ldev, cfgs, devs, n_inst=args.batch_size, cfg_rows=False,
                             max_s=wk.max_s, p_max=prefix, mimi_stream=False)
            eng.capture()
        else:
            eng, _ = wk.get_engine(args.batch_size, wk.max_s, prefix)
        write_json(shard_root/'engine.json', dict(binfo, engine_sha256=engine_hash,
                    engine_shape=[args.batch_size, wk.max_s, prefix],
                    cfg_rows=not args.pose_cfg_off,
                    params_sha256=prepared['params_sha256'], memory_allocated=[torch.cuda.memory_allocated(d) for d in devs]))
        for batch_index in assigned:
            indices = list(range(batch_index*args.batch_size, min(args.total, (batch_index+1)*args.batch_size)))
            if all(i in already for i in indices):
                continue
            if halt.is_set() or (args.root/'STOP').exists():
                state.update(state='stopped', finished=time.time())
                break
            seed = args.seed_base+batch_index
            batch_id = f'audit-batch-{batch_index:06d}'
            state.update(state='generating', batch_index=batch_index, batch_seed=seed,
                         sample_indices=indices, frames_done=0, frames_total=frames+worker.CD)
            executions = []
            for row, index in enumerate(indices):
                execution = {'kind': 'batched_cuda_graph', 'batch_id': batch_id,
                             'batch_index': batch_index, 'batch_seed': seed, 'engine_row': row,
                             'sample_index': index, 'sample_count': len(indices), 'engine_rows': args.batch_size,
                             'engine_shape': [args.batch_size, wk.max_s, prefix],
                             'cfg_rows': not args.pose_cfg_off,
                             'seed_mode': 'batch_seed_and_row', 'engine_sha256': engine_hash}
                executions.append(execution)
                if index in already:
                    continue
                directory = run_dir(index)
                directory.mkdir(parents=True, exist_ok=True)
                req = copy.deepcopy(source)
                if args.pose_cfg_off:
                    req['params']['pose_cfg_on'] = 0
                    if isinstance(req.get('ui'), dict):
                        req['ui']['pose_cfg_on'] = False
                req.update(run_id=directory.name, created=time.time(), seed=seed, clip_s=clip_s,
                           stream=False, no_live=True, gt=False, client='pose-jump-audit',
                           audit={'fingerprint': fingerprint, 'reference_run': source['run_id'],
                                  'total_seconds': args.duration, 'sample_index': index,
                                  'parameter_delta': fingerprint_fields.get('parameter_delta', {})},
                           batch={'id': batch_id, 'k': row, 'n': len(indices)})
                if isinstance(req.get('ui'), dict):
                    req['ui'].update(seed=seed, n_samples=len(indices), clip_s=clip_s)
                write_json(directory/'request.json', req)
                write_json(directory/'status.json', {'state': 'running', 'execution': execution, 'started': time.time()})
            instances = [dict(ctx['x'], a_mode='empty') for _ in range(args.batch_size)]
            t0 = time.time()
            try:
                eng.start(instances, seed, ctx['params'], f0=0)
                for f in range(frames+worker.CD):
                    g, force = f-worker.CD, {}
                    if ctx['pp'] is not None and 0 <= g < ctx['N']:
                        force['pose'] = np.repeat((ctx['pp'][g]+worker.OFF)[None], args.batch_size, axis=0)
                    if ctx['pa'] is not None and 0 <= g < ctx['NA']:
                        force['codes'] = np.repeat((ctx['pa'][g]+worker.OFF)[None], args.batch_size, axis=0)
                    if ctx['lane_ids'] is not None and f < ctx['NA']+worker.CD:
                        force['lane'] = np.full(args.batch_size, ctx['lane_ids'][ctx['P']+f], np.int64)
                    eng.step(f, force=force or None, cold_pad=ctx['lane_ids'] is None)
                    if f % 25 == 0 or f == frames+worker.CD-1:
                        state.update(frames_done=f+1, batch_elapsed_seconds=time.time()-t0)
                for row, index in enumerate(indices):
                    if index in already:
                        continue
                    directory = run_dir(index)
                    P, n = ctx['P'], frames
                    saved = eng.readback(row, P+n+worker.CD)
                    codes = saved['dep_b'][P+worker.CD:P+worker.CD+n]-worker.OFF
                    poses = saved['pdep'][P+worker.CD:P+worker.CD+n]-worker.OFF
                    if not all(x.shape == (n, 32) and x.min() >= 0 and x.max() < FP.CBS for x in (codes, poses)):
                        raise ValueError(f'Invalid output tokens for sample {index}')
                    if ctx['pp'] is not None and not np.array_equal(poses[:ctx['N']], ctx['pp']):
                        raise ValueError(f'Pose prefill mismatch for sample {index}')
                    if ctx['pa'] is not None and not np.array_equal(codes[:ctx['NA']], ctx['pa']):
                        raise ValueError(f'Audio prefill mismatch for sample {index}')
                    with (directory/'tokens.npz.tmp').open('wb') as fp:
                        np.savez_compressed(fp, pose=poses.astype(np.int16), mimi=codes.astype(np.int16), lane=saved['lane_b'][P:P+n])
                    (directory/'tokens.npz.tmp').replace(directory/'tokens.npz')
                    words = worker.IO.lane_words(saved['lane_b'][:P+n], P, P, P)
                    wl = [{'t0': float(w['t']), 't1': round(min(float(words[k+1]['t']) if k+1<len(words) else float(w['t'])+.4, float(w['t'])+1.5), 2),
                           'w': w['text'], 'ch': 'B', 'prefill': float(w['t']) < ctx['N']/12.5} for k, w in enumerate(words)]
                    write_json(directory/'words.json', wl)
                    summary = {'run_id': directory.name, 'fingerprint': fingerprint, 'sample_index': index,
                               'reference_run': source['run_id'], 'model': 'monoplex export-355',
                               'parameter_delta': fingerprint_fields.get('parameter_delta', {}),
                               'person': source['prefill']['person'], 'params': ctx['params'],
                               'params_sha256': prepared['params_sha256'], 'seed': seed,
                               'frames': n, 'duration': n/12.5, 'P': P, 'prefill_frames': ctx['N'],
                               'pose_prefill_frames': ctx['N'], 'forced_audio_frames': ctx['NA'],
                               'prefill_s': source['prefill']['s'], 'prefill_source': 'clip',
                               'frame0_global_state': None, 'clip_s': clip_s,
                               'execution': executions[row], 'cfg_guided_frames': int(eng.pcfg_guided[row].item()),
                               'tokens_sha256': file_digest(directory/'tokens.npz'),
                               'batch_wall_seconds': time.time()-t0, 'audio_decoded': False, 'video_rendered': False}
                    write_json(directory/'summary.json', summary)
                    write_json(directory/'status.json', {'state': 'done', 'execution': executions[row],
                               'frames_done': frames, 'duration': n/12.5, 'finished': time.time()})
                    already.add(index)
                state.update(completed_samples=len(already), last_completed_batch=batch_index)
                with (shard_root/'events.jsonl').open('a') as fp:
                    fp.write(json.dumps({'event': 'batch_done', 'batch_index': batch_index, 'seed': seed,
                                         'sample_indices': indices, 'time': time.time(), 'completed_samples': len(already)})+'\n')
                print(json.dumps(dict(state, heartbeat=time.time())), flush=True)
            except Exception as exc:
                state.update(state='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
                for index in indices:
                    if index not in already:
                        write_json(run_dir(index)/'status.json', {'state': 'error', 'error': state['error'], 'time': time.time()})
                raise
        else:
            state.update(state='done', finished=time.time())
    final.set()
    write_json(shard_root/'status.json', dict(state, heartbeat=time.time()))
    print(json.dumps(state), flush=True)


if __name__ == '__main__':
    main()
