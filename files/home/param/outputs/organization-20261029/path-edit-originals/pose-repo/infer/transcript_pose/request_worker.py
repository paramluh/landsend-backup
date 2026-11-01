#!/usr/bin/env python3
"""Prepared-clip executors on landsend-1 through 4, plus the legacy serial loop.

`execute-job` processes one dispatcher assignment. Both GPUs are locked during
PPM generation; the second is released when PPM exits, and the first remains
locked through pose/video rendering. `check` is CPU-only. The legacy `run` loop
retains its landsend-1 GPU6/7 defaults. No executor may run on node0 or node12.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import fcntl
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
import traceback
import uuid

from request_schema import CLIP_IDS, RECIPES, load_request, media_frames

REPO = Path(__file__).resolve().parents[2]
SOURCE = Path('/home/param/pose-runs/controlled-comparison-20261028')
QUEUE = Path('/home/param/pose-runs/inference-requests-20261028')
EXECUTIONS = Path('/home/param/pose-runs/inference-request-executions-20261028')
NODES = tuple(f'landsend-{i}' for i in range(1, 5))
MODELS = Path('/home/param/models')
PPM_PY = REPO / '.venv/bin/python'
VIDEO_PY = Path('/home/param/vansh-video-models/.venv/bin/python')
GPU_LOCKS = Path('/home/param/.cache/heldout-comparison-gpu-locks')
GPUS = (6, 7)
_PREFILL_LOCK = threading.RLock()


def stamp():
    return datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')


def read(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Expected regular file: {path}')
    return json.loads(path.read_text())


def write(path, data):
    """Durable atomic replacement: readers see either old or complete new JSON."""
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    try:
        with temporary.open('x') as stream:
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def state(job, status, stage, **extra):
    path = job / 'state.json'
    try:
        previous = read(path) if path.exists() else {}
        require(isinstance(previous, dict), 'State must be an object')
    except (ValueError, OSError):
        previous = {}
    now = stamp()
    previous.update(id=job.name, status=status, stage=stage, updated_at=now)
    previous.setdefault('created_at', now)
    previous.update(extra)
    if status != 'complete':
        for key in ('video', 'metadata', 'pose_video', 'pose_sha256', 'pose_overlay_video', 'pose_overlay_sha256'):
            previous.pop(key, None)
    write(path, previous)


def jobs(queue):
    """Only API-published 32-character lowercase UUID hex; .new-* is invisible."""
    result = []
    for path in (queue / 'jobs').iterdir():
        try:
            valid = uuid.UUID(path.name).hex == path.name
        except ValueError:
            valid = False
        if valid and path.is_dir() and not path.is_symlink():
            request = path / 'request.json'
            if request.is_file() and not request.is_symlink():
                result.append(path)
    return sorted(result, key=lambda p: ((p / 'request.json').stat().st_mtime_ns, p.name))


def request(job):
    path = job / 'request.json'
    require(not path.is_symlink() and path.stat().st_size <= 8192, 'Invalid or oversized request file')
    return load_request(path.read_text())


def job_status(job):
    try:
        current = read(job / 'state.json')['status']
        require(current in ('queued', 'running', 'complete', 'failed'), 'Invalid request state')
        return current
    except (ValueError, KeyError, TypeError, OSError):
        state(job, 'failed', 'failed', completed_at=stamp(), error='Missing or corrupt durable request state')
        return 'failed'


def recover_interrupted(queue):
    # The exclusive worker lock proves that no previous worker still owns a job.
    for job in jobs(queue):
        if job_status(job) == 'running':
            state(job, 'failed', 'failed', completed_at=stamp(),
                  error='Worker stopped during this request. Partial outputs were retained; submit a new request to retry.')


def comparison_ready(source_root=None):
    path = (Path(source_root) if source_root is not None else SOURCE) / 'CONTROLLED_COMPARISON_COMPLETE.json'
    if not path.is_file():
        return False
    receipt = read(path)
    teams = receipt.get('teams', [])
    expected = {(recipe, clip) for recipe in RECIPES for clip in CLIP_IDS}
    actual = [(team.get('recipe'), clip) for team in teams for clip in team.get('clips', [])]
    return (receipt.get('status') == 'complete' and len(teams) == 16 and len(actual) == 24
            and set(actual) == expected and all(t.get('status') == 'complete' for t in teams))


def source_clip(clip_id, source_root=None):
    source_root = Path(source_root) if source_root is not None else SOURCE
    selection = read(source_root / 'selection.json')['clips']
    require(len(selection) == 8 and {c['id'] for c in selection} == set(CLIP_IDS), 'Invalid prepared clip selection')
    clip = next(c for c in selection if c['id'] == clip_id)
    require(clip['split'] == 'eval' and clip['frames'] == 752 and clip['voice_clip_frames'] == 100,
            'Prepared clip must retain its held-out source and original eight-second voice')
    assets = read(source_root / clip_id / 'video_assets.json')
    require(assets['status'] == 'complete' and assets['reference_count'] == 9, 'Appearance assets are not ready')
    require(assets['actor_id'] == clip['actor_id'] and assets['clip_key'] == clip['clip_key'], 'Appearance identity mismatch')
    for path in [clip['voice_clip'], clip['lane'], assets['frame0_pose'], assets['sink_latent'],
                 assets['reference_bank']]:
        require(Path(path).resolve().is_relative_to((source_root / clip_id).resolve()) and Path(path).exists(),
                'Prepared source asset missing or outside its clip directory')
    for prop in range(1, 10):
        require((Path(assets['reference_bank']) / assets['actor_id'] / f'property_{prop}/latents.safetensors').is_file(),
                f'Missing reference property {prop}')
    require(clip['voice_provenance']['nonoverlapping_generation'] is True, 'Voice reference overlaps generation')
    return clip, assets


def parse_gpu_pair(value):
    try:
        pair = tuple(int(x) for x in value.split(','))
    except (ValueError, AttributeError):
        raise ValueError('GPU pair must be two distinct comma-separated IDs in [0,7]') from None
    require(len(pair) == 2 and pair[0] != pair[1] and all(0 <= gpu <= 7 for gpu in pair),
            'GPU pair must be two distinct comma-separated IDs in [0,7]')
    return pair


def execution_guard(node, gpus):
    require(node in NODES and socket.gethostname().split('.')[0] == node,
            'Executor must run on its assigned host, restricted to landsend-1 through landsend-4')
    require(os.getuid() == 1003 and os.geteuid() == 1003, 'GPU worker must run as param (UID 1003), never root')
    require(tuple(gpus) == parse_gpu_pair(','.join(map(str, gpus))), 'Invalid GPU assignment')


def environment(video=False, gpus=GPUS):
    gpus = parse_gpu_pair(','.join(map(str, gpus)))
    env = os.environ.copy()
    # Video uses only the primary GPU. Reuse the already-warm comparison cache
    # regardless of which secondary GPU was available for the PPM stage.
    cache = (f'/home/param/.cache/heldout-video/gpu-{gpus[0]}' if video
             else f'/home/param/.cache/request-worker/gpu-{gpus[0]}-{gpus[1]}')
    env.update(CUDA_VISIBLE_DEVICES=str(gpus[0]) if video else ','.join(map(str, gpus)), OMP_NUM_THREADS='2' if video else '8',
               HF_HUB_OFFLINE='1', PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false',
               MIMI_DIR=str(MODELS / 'mimi'), CUDA_HOME='/usr/local/cuda',
               PPM_FA3_ROOT=str(REPO / '.cache/kernels/vllm-flash-attn3'),
               FA3_DIR=str(REPO / '.cache/kernels/vllm-flash-attn3'),
               POSE_TOKENIZER_DIR=str(MODELS / 'pose-tokenizer-32cb-200k-noshift'),
               ELECTRON_ENCODER=str(MODELS / 'electron-vae/encoder-v2-pose.safetensors'),
               CPATH=f'{REPO}/.venv/include/python3.12:{REPO}/.venv/include',
               TORCHINDUCTOR_CACHE_DIR=f'{cache}/inductor', TRITON_CACHE_DIR=f'{cache}/triton')
    env['PATH'] = str((VIDEO_PY if video else PPM_PY).parent) + os.pathsep + env.get('PATH', '')
    return env


class StopRequested(Exception):
    pass


def stop_signal(signum, frame):
    raise StopRequested(f'Worker received signal {signum}')


def execute(command, log, *, env=None, timeout=7200):
    """No shell; on interruption terminate only this worker's own child group."""
    with log.open('a') as output:
        output.write(f'\n{stamp()} {json.dumps(list(map(str, command)))}\n')
        output.flush()
        proc = subprocess.Popen(list(map(str, command)), cwd=REPO, env=env,
                                stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = proc.wait(timeout=timeout)
            if code:
                raise RuntimeError(f'{Path(command[0]).name} exited {code}; see {log.name}')
        except BaseException:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            raise


class GPUReservation:
    def __init__(self, handles):
        self.handles = handles

    def release(self, gpu):
        handle = self.handles.pop(gpu, None)
        if handle is not None:
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()


@contextmanager
def gpu_pair(gpus=GPUS):
    """Acquire the same per-device locks used by the comparison renderer."""
    GPU_LOCKS.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        handles = {}
        for gpu in gpus:
            lock = stack.enter_context((GPU_LOCKS / f'gpu-{gpu}.lock').open('a'))
            handles[gpu] = lock
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                yield False
                return
        for gpu in gpus:
            common = ['nvidia-smi', '-i', str(gpu)]
            pids = subprocess.check_output([*common, '--query-compute-apps=pid', '--format=csv,noheader'],
                                           text=True, timeout=10).strip()
            used = int(subprocess.check_output([*common, '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                                text=True, timeout=10).strip())
            if pids or used >= 1024:
                yield False
                return
        yield GPUReservation(handles)


def generation_command(job, req, clip):
    n = media_frames(req['duration_seconds'])
    batch = [{'clip_index': clip['clip_index'], 'start_frame': clip['start_frame'], 'frames': n,
              'voice_clip': clip['voice_clip'], 'seed': req['seed'], 'out': str(job / 'ppm')}]
    write(job / 'batch.json', batch)
    command = [PPM_PY, '-u', REPO / 'infer/transcript_pose/infer_batch.py',
               '--batch-json', job / 'batch.json', '--recipe', req['recipe'], '--devs', '0,1',
               '--ckpt', MODELS / 'transcript-pose' / req['recipe'] / 'export-200',
               '--text-mode', req['text_mode'], '--prefill-frames', str(req.get('prefill_frames', 0))]
    for key in ('audio_temp', 'pose_temp', 'audio_topk', 'pose_topk', 'text_temp', 'text_topk'):
        command += ['--' + key.replace('_', '-'), str(req[key])]
    return command


@lru_cache(maxsize=1)
def text_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(MODELS / 'ppm-811-17850-base-bf16/configs/Qwen3-32B'),
                                         local_files_only=True)


@lru_cache(maxsize=1)
def prefill_source_data():
    from inputs import training_data
    return training_data().CompactXabi('/home/param/datasets/xabi-v2pure-cache', tokenizer=text_tokenizer())


def source_prefill(clip, count):
    """Read local codec IDs directly from the source cache, independent of delay packing."""
    import numpy as np

    if not count:
        return {}
    with _PREFILL_LOCK:
        data = prefill_source_data()
        source = data.clips[clip['clip_index']]
        require(source.cache_key == clip['cache_key'] and source.cache_index == clip['cache_index'],
                'Prefill source recording changed')
        _, offsets, audio, pose = data._shard(source.shard)
        start = int(offsets[source.cache_index]) + clip['start_frame']
        end = start + count
        require(end <= int(offsets[source.cache_index + 1]), 'Prefill exceeds source recording')
        arrays = dict(audio_codes=np.asarray(audio[start:end].T, dtype=np.int64).copy(),
                      pose_codes=np.asarray(pose[start:end].T, dtype=np.int64).copy(),
                      text_lane=np.asarray(np.load(clip['lane'], allow_pickle=False)[:count], dtype=np.int64))
    require(arrays['audio_codes'].shape == arrays['pose_codes'].shape == (32, count), 'Prefill code shape mismatch')
    require(all(np.all((a >= 0) & (a < 2048)) for k, a in arrays.items() if k != 'text_lane'),
            'Prefill contains invalid source codec IDs')
    return arrays


def array_hashes(arrays):
    import numpy as np
    return {k: hashlib.sha256(np.ascontiguousarray(v, dtype=np.int64).tobytes()).hexdigest()
            for k, v in arrays.items()}


def validate_inputs(ppm, req, clip, meta):
    """Verify the actual saved tensors, including absence of timed-text leakage."""
    import numpy as np

    n, forced = media_frames(req['duration_seconds']), req['text_mode'] == 'forced'
    count = req.get('prefill_frames', 0)
    source = source_prefill(clip, count)
    hashes = array_hashes(source)
    require(meta['prefill_frames'] == count and meta['f0'] == count, 'Prefill generation boundary mismatch')
    if count:
        require(meta.get('ground_truth_prefill_media') is True
                and meta.get('prefill_modalities') == ['audio', 'pose', 'text']
                and meta.get('prefill_duration_seconds') == count / 12.5
                and meta.get('prefill_policy') == 'exact synchronized audio/pose/text prefix'
                and meta.get('prefill_sha256') == hashes,
                'Ground-truth prefill provenance differs from the original source cache')
    lane = np.load(clip['lane'], allow_pickle=False)[:n]
    words = read(Path(clip['words']))
    source_text = ' '.join(w['text'] for w in words if
                          (w['end'] > 0 and w['start'] < n / 12.5) or
                          (w['end'] == w['start'] and 0 <= w['start'] < n / 12.5))
    require(meta['transcript'] == source_text, 'Source transcript differs from the prepared clip window')
    supplied = source_text if forced or req['recipe'] == 'compact-all' else None
    if 'supplied_transcript' in meta:
        require(meta['supplied_transcript'] == supplied, 'Transcript conditioning metadata mismatch')
    with np.load(ppm / 'inputs.npz', allow_pickle=False) as inputs:
        P, conditioning_P = int(meta['P']), int(meta['conditioning_P'])
        require(conditioning_P == clip['conditioning_P'] == 109 and P >= conditioning_P,
                'Voice/system conditioning length changed')
        prefix_hashes = {key: hashlib.sha256(np.ascontiguousarray(inputs[key][:, :conditioning_P]).tobytes()).hexdigest()
                         for key in ('ids', 'depth', 'present')}
        require(prefix_hashes == clip['voice_provenance']['system_prefix_array_sha256'], 'Voice/system prefix changed')
        require(meta['transcript_span'] == [conditioning_P, P], 'Upfront transcript span mismatch')
        prefix = inputs['ids'][1, conditioning_P:P]
        if req['recipe'] == 'xabi-original':
            require(P == conditioning_P and meta['transcript_end_token'] is None,
                    'Original Xabi must have no upfront transcript prefix')
        else:
            require(len(prefix) >= 2 and prefix[0] == 151680 and prefix[-1] == 151681
                    and meta['transcript_end_token'] == 151681, 'Missing transcript prefix boundaries')
            if req['recipe'] == 'aligned-media':
                require(np.array_equal(prefix[1:-1], lane), 'Upfront aligned transcript changed')
            else:
                expected = text_tokenizer().encode(source_text, add_special_tokens=False, split_special_tokens=True)
                require(np.array_equal(prefix[1:-1], expected), 'Upfront ordinary transcript changed')
        require(np.all(inputs['depth'][0, P:] == -1)
                and np.all(inputs['depth'][1, P:P+2] == -1)
                and np.all(inputs['depth'][1, P+2+count:] == -1)
                and np.all(inputs['pdepth'][:P+2] == -1)
                and np.all(inputs['pdepth'][P+2+count:] == -1),
                'Ground-truth media was supplied outside the explicit prefill')
        if count:
            offsets = np.arange(32, dtype=np.int64) * 2048
            require(np.array_equal((inputs['depth'][1, P+2:P+2+count] - offsets).T, source['audio_codes'])
                    and np.array_equal((inputs['pdepth'][P+2:P+2+count] - offsets).T, source['pose_codes'])
                    and np.array_equal(inputs['ids'][1, P:P+count], source['text_lane']),
                    'Packed prefill audio, pose or text differs from the original source')
        if forced:
            require('text_force' in inputs and np.array_equal(inputs['text_force'][P:P+n], lane)
                    and np.all(inputs['text_force'][P+n:] == 151670), 'Forced text or delay tails differ')
            require(np.array_equal(inputs['ids'][1], inputs['text_force']), 'Backbone text differs from forced lane')
            lane_hash = hashlib.sha256(np.ascontiguousarray(inputs['text_force'][P:], dtype=np.int64).tobytes()).hexdigest()
        else:
            require('text_force' not in inputs and np.all(inputs['ids'][1, P+count:] == 151670),
                    'Native generation received ground-truth framewise text after prefill')
            lane_hash = None
        require(meta['framewise_lane_sha256'] == lane_hash, 'Supplied framewise text checksum mismatch')
    return dict(framewise_lane_sha256=lane_hash, conditioning_prefix_sha256=prefix_hashes,
                source_transcript=source_text, supplied_transcript=supplied,
                prefill_frames=count, prefill_duration_seconds=count / 12.5,
                prefill_sha256=hashes, ground_truth_prefill_media=bool(count))


def validate_ppm(job, req, clip):
    import numpy as np
    import soundfile as sf
    from safetensors.numpy import load_file

    ppm, n = job / 'ppm', media_frames(req['duration_seconds'])
    meta = read(ppm / 'meta.json')
    forced = req['text_mode'] == 'forced'
    expected = dict(recipe=req['recipe'], clip_index=clip['clip_index'], frames=n, seed=req['seed'],
                    batch_size=1, force_text=forced, prefill_frames=req.get('prefill_frames', 0),
                    ground_truth_generation_media=False, voice_clip_frames=100)
    require(all(meta.get(k) == v for k, v in expected.items()), 'PPM identity or conditioning mismatch')
    require(meta.get('text_mode', 'forced') == req['text_mode'], 'Text mode metadata mismatch')
    require(meta['checkpoint']['ck'] == str(MODELS / 'transcript-pose' / req['recipe'] / 'export-200')
            and meta['checkpoint']['pose'] is True and meta['checkpoint']['decp_bb'] is True
            and meta['checkpoint']['decp_bb_alpha'] == 1.0, 'PPM checkpoint or architecture mismatch')
    for modality, prefix in [('audio', 'cb0'), ('audio', 'light'), ('pose', 'pose'), ('pose', 'pose_light'), ('text', 'text')]:
        require(meta['sampling'][prefix + '_temp'] == req[modality + '_temp'] and
                meta['sampling'][prefix + '_topk'] == req[modality + '_topk'], 'Sampling settings mismatch')
    voice = np.load(clip['voice_clip'], allow_pickle=False)
    voice_hash = hashlib.sha256(np.ascontiguousarray(voice, dtype=np.int64).tobytes()).hexdigest()
    require(voice.shape == (32, 100) and voice_hash == clip['voice_provenance']['codes_sha256'], 'Original voice changed')
    require(meta['voice_codes_sha256'] == voice_hash and
            np.array_equal(voice, np.load(ppm / 'voice_clip_mimi.npy', allow_pickle=False)), 'Supplied voice differs')
    lane = np.load(clip['lane'], allow_pickle=False)[:n]
    with np.load(ppm / 'tokens.npz', allow_pickle=False) as data:
        generated = data['lane']
        require(generated.shape == (n,) and np.issubdtype(generated.dtype, np.integer)
                and np.all((generated >= 0) & ((generated < 151643) | np.isin(generated, [151645, 151670, 151671]))),
                'Invalid output text lane')
        if forced:
            require(np.array_equal(generated, lane), 'Generated text differs from forced original framewise lane')
        elif not req.get('prefill_frames', 0):
            require(generated[0] == 151670, 'Native generation must retain the initial PAD bootstrap')
        source = source_prefill(clip, req.get('prefill_frames', 0))
        for key, codes in source.items():
            actual = generated[:len(codes)] if key == 'text_lane' else data[key][:, :codes.shape[1]]
            require(np.array_equal(actual, codes), f'Output {key} changed the ground-truth prefill')
        generated_hash = hashlib.sha256(np.ascontiguousarray(generated, dtype=np.int64).tobytes()).hexdigest()
        if not forced or 'generated_lane_sha256' in meta:
            require(meta.get('generated_lane_sha256') == generated_hash, 'Generated text checksum mismatch')
        for name, filename in [('audio_codes', 'mimi.safetensors'), ('pose_codes', 'pose_tokens.safetensors')]:
            codes = data[name]
            require(codes.shape == (32, n) and np.issubdtype(codes.dtype, np.integer)
                    and codes.min() >= 0 and codes.max() < 2048, 'Invalid generated media codes')
            require(np.array_equal(codes, load_file(str(ppm / filename))['codes']), 'Exported media codes differ')
    input_validation = validate_inputs(ppm, req, clip, meta)
    audio, sr = sf.read(ppm / 'audio.wav')
    require(sr == 24000 and len(audio) == n * 1920 and np.isfinite(audio).all(), 'Invalid decoded audio')
    return {**input_validation, 'voice_codes_sha256': voice_hash, 'generated_lane_sha256': generated_hash,
            'sampling': meta['sampling'], 'checkpoint': meta['checkpoint'],
            'text_mode': req['text_mode'], 'text_sampling_active': not forced,
            'generated_transcript': read(ppm / 'words.json')['text'], 'audio_peak': float(np.max(np.abs(audio))),
            'audio_clipped_samples': int(np.count_nonzero(np.abs(audio) >= 1)),
            'generation_seconds': meta['generation_seconds']}


def validate_latents(path, shape=None):
    import torch
    from safetensors.torch import load_file

    tensors = load_file(str(path), device='cpu')
    require(bool(tensors) and all(torch.isfinite(t).all().item() for t in tensors.values()), 'Nonfinite latent tensor')
    if shape:
        require(tuple(tensors['latents'].shape) == shape, f'Latent shape mismatch: {path.name}')


def validate_video_receipt(receipt, n, seed):
    checkpoint = MODELS / 'full-gt-nine-ref-endpoint-unguided-20261020/step-002911/student.safetensors'
    expected = dict(status='complete', decoded_frames=2*n-3, checkpoint=str(checkpoint), shared_weights=True,
                    reference_keep=[True]*9, stabilizer_total_steps=8, stabilizer_band_steps=[2]*4,
                    resolver_steps=1, renoising=False, runtime_audio_guidance_scale=1.0, seed=seed)
    require(all(receipt.get(k) == v for k, v in expected.items()), 'Video model, schedule, references or output contract mismatch')
    return {key: receipt[key] for key in ('checkpoint', 'shared_weights', 'reference_keep', 'stabilizer_band_steps',
                                         'stabilizer_total_steps', 'resolver_steps', 'renoising', 'runtime_audio_guidance_scale')}


def run_job(job, req, clip, assets, *, node='landsend-1', gpus=GPUS, source_root=None,
            release_secondary=None, release_primary=None):
    execution_guard(node, gpus)
    source_root = Path(source_root) if source_root is not None else SOURCE
    started = time.monotonic()
    n, seconds = media_frames(req['duration_seconds']), req['duration_seconds']
    state(job, 'running', 'generating', started_at=stamp(), request=req, assigned_node=node,
          gpu_pair=list(gpus), active_gpus=list(gpus), executor_pid=os.getpid())
    write(job / 'validated_request.json', req)
    execute(generation_command(job, req, clip), job / 'ppm.log', env=environment(gpus=gpus))
    if release_secondary is not None:
        release_secondary()
        state(job, 'running', 'validating-ppm', active_gpus=[gpus[0]])
    ppm_validation = validate_ppm(job, req, clip)
    state(job, 'running', 'rendering-pose')
    execute([VIDEO_PY, '-u', REPO / 'infer/transcript_pose/render.py', job / 'ppm',
             '--frame0', assets['frame0_pose'], '--skeleton-video'],
            job / 'pose.log', env=environment(video=True, gpus=gpus))
    validate_latents(job / 'ppm/pose_latents_electron_v3.safetensors', (n // 2 + 1, 48, 30, 16))
    state(job, 'running', 'rendering-video')
    execute([VIDEO_PY, '-u', REPO / 'infer/transcript_pose/render_video.py', job / 'ppm',
             '--sink', assets['sink_latent'], '--reference-bank', assets['reference_bank'],
             '--actor-id', assets['actor_id'], '--clip-key', assets['clip_key'],
             '--seed', str(req['video_seed']), '--out', job / 'diffusion'],
            job / 'video.log', env=environment(video=True, gpus=gpus))
    if release_primary is not None:
        release_primary()
        state(job, 'running', 'validating', active_gpus=[])
    else:
        state(job, 'running', 'validating')
    receipt = read(job / 'diffusion/generation/status.json')
    video_validation = validate_video_receipt(receipt, n, req['video_seed'])
    require(bool(receipt['latent_files']), 'Missing generated video latents')
    for item in receipt['latent_files']:
        path = Path(item['path']).resolve()
        require(path.is_relative_to((job / 'diffusion').resolve()) and sha(path) == item['sha256'],
                'Generated latent file identity mismatch')
        validate_latents(path)
    frames = seconds * 25
    temporary, final = job / 'video.partial.mp4', job / 'video.mp4'
    execute(['ffmpeg', '-v', 'error', '-y', '-i', job / 'diffusion/generation/video.mp4',
             '-map', '0:v:0', '-map', '0:a:0', '-vf', f'trim=end_frame={frames},setpts=PTS-STARTPTS',
             '-af', f'atrim=end={seconds},asetpts=PTS-STARTPTS', '-frames:v', str(frames), '-t', str(seconds),
             '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18', '-pix_fmt', 'yuv420p',
             '-threads', '2', '-c:a', 'aac', '-ar', '24000', '-movflags', '+faststart', temporary], job / 'validation.log')
    streams = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json',
                                                  str(temporary)], timeout=60))['streams']
    video = next(s for s in streams if s['codec_type'] == 'video')
    audio = next(s for s in streams if s['codec_type'] == 'audio')
    require(int(video['nb_frames']) == frames and abs(float(video['duration']) - seconds) < .001
            and video['r_frame_rate'] == '25/1', 'Final video duration or frame count mismatch')
    require(int(audio['sample_rate']) == 24000 and abs(float(audio['duration']) - seconds) < .05, 'Final audio duration mismatch')
    execute(['ffmpeg', '-v', 'error', '-xerror', '-i', temporary, '-f', 'null', '-'], job / 'validation.log')
    with temporary.open('rb') as stream:
        os.fsync(stream.fileno())
    temporary.replace(final)
    state(job, 'running', 'rendering-pose-views', active_gpus=[])
    execute([VIDEO_PY, '-u', REPO / 'infer/transcript_pose/pose_views.py', '--video', final,
             '--pose-dir', job / 'ppm', '--out', job, '--frame0', assets['frame0_pose']],
            job / 'pose-views.log', env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2'))
    marker = None
    if req.get('prefill_frames', 0):
        from prefill_marker import apply_prefill_marker
        marker = apply_prefill_marker(job, boundary_seconds=req['prefill_frames'] / 12.5, threads=2)
    from pose_views import validate_pose_views
    pose_receipt = validate_pose_views(job, video=final)
    reference = read(source_root / req['clip_id'] / 'reference-selection.json')
    refs = [{**{k: r[k] for k in ('property', 'source_timestamp_s', 'source_sha256')},
             'inside_evaluation_window': clip['source_start_seconds'] <= r['source_timestamp_s'] < clip['source_start_seconds'] + n / 12.5}
            for r in reference['rows']]
    metadata = dict(request=req, frames=frames, duration_seconds=seconds, media_frames=n,
                    text_mode=req['text_mode'], force_text=req['text_mode'] == 'forced',
                    ground_truth_generation_media=False, batch_size=1,
                    ground_truth_prefill_media=bool(req.get('prefill_frames', 0)),
                    prefill_frames=req.get('prefill_frames', 0),
                    prefill_duration_seconds=req.get('prefill_frames', 0) / 12.5,
                    prefill_marker=marker,
                    clip_index=clip['clip_index'], recording_id=clip['recording_id'], source_start_seconds=clip['source_start_seconds'],
                    voice_reference_seconds=8, voice_reference_nonoverlapping_generation=True,
                    ppm=ppm_validation, pose=read(job / 'ppm/render.json'),
                    video_model='Shared raw step-2911 nine-reference student',
                    video_schedule=video_validation,
                    video_input=read(job / 'diffusion/input.json'),
                    reference_evaluation_window_overlap_count=sum(r['inside_evaluation_window'] for r in refs),
                    reference_images=refs,
                    strict_bank_byte_identity=False,
                    evaluation_scope='Held-out PPM recording; image-conditioned video demonstration, not leakage-free video evaluation',
                    postprocessing=f'Trim {2*n-3} generated RGB frames to {frames}; trim audio to {seconds}s; no padding or looping',
                    full_cpu_decode_passed=True, sha256=sha(final), completed_at=stamp(),
                    pose_video='pose.mp4', pose_sha256=sha(job / 'pose.mp4'),
                    pose_overlay_video='overlay.mp4', pose_overlay_sha256=sha(job / 'overlay.mp4'),
                    pose_views=pose_receipt,
                    wall_seconds=time.monotonic() - started, node=node, gpus=list(gpus),
                    gpu_stages={'ppm': list(gpus), 'pose': [gpus[0]], 'video': [gpus[0]]})
    write(job / 'metadata.json', metadata)
    state(job, 'complete', 'complete', completed_at=stamp(), video='video.mp4', metadata='metadata.json', active_gpus=[],
          pose_video='pose.mp4', pose_sha256=metadata['pose_sha256'],
          pose_overlay_video='overlay.mp4', pose_overlay_sha256=metadata['pose_overlay_sha256'])


def process_start_ticks(pid):
    """Linux process identity, robust to PID reuse and spaces in comm."""
    return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19])


def execute_job(job, node, gpus, source_root=None):
    """One assigned request; exit75 is capacity unavailable, never a failed sample."""
    execution_guard(node, gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    source_root = Path(source_root) if source_root is not None else SOURCE
    require(not job.is_symlink() and job.is_dir() and job.parent.name == 'jobs'
            and uuid.UUID(job.name).hex == job.name, 'Expected a canonical32hex job directory')
    with (job / '.executor.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('An executor already owns this job; no duplicate launch', flush=True)
            return 75
        assignment = read(job / 'assignment.json')
        require(assignment.get('node') == node and assignment.get('gpus') == list(gpus),
                'Dispatcher assignment does not match executor host/GPU arguments')
        claim = assignment.get('claim_token')
        require(isinstance(claim, str) and uuid.UUID(claim).hex == claim, 'Invalid dispatcher claim token')
        if job_status(job) == 'complete':
            old = read(job / 'executor.json')
            require(old.get('claim_token') == claim and old.get('status') == 'complete'
                    and sha(job / 'video.mp4') == read(job / 'metadata.json')['sha256'],
                    'Completed execution identity/checksum mismatch')
            return 0
        require(job_status(job) in ('queued', 'running'), 'Terminal failed execution requires a new request')
        record = dict(pid=os.getpid(), start_ticks=process_start_ticks(os.getpid()), claim_token=claim,
                      node=node, gpus=list(gpus), active_gpus=list(gpus), phase='acquiring',
                      status='running', started_at=stamp())

        def update(status, **extra):
            record.update(status=status, updated_at=stamp(), **extra)
            write(job / 'executor.json', record)

        update('running')
        try:
            req = request(job)
            clip, assets = source_clip(req['clip_id'], source_root)
            if not comparison_ready(source_root):
                state(job, 'queued', 'waiting-for-comparison', request=req, active_gpus=[])
                update('busy', exit_code=75, active_gpus=[], completed_at=stamp())
                return 75
            with gpu_pair(gpus) as reservation:
                if not reservation:
                    state(job, 'queued', 'waiting-for-gpus', request=req, active_gpus=[])
                    update('busy', exit_code=75, active_gpus=[], completed_at=stamp())
                    return 75
                update('running', active_gpus=list(gpus), phase='generating')

                def release_secondary():
                    reservation.release(gpus[1])
                    update('running', active_gpus=[gpus[0]], phase='pose-video')

                def release_primary():
                    reservation.release(gpus[0])
                    update('running', active_gpus=[], phase='validating')

                run_job(job, req, clip, assets, node=node, gpus=gpus, source_root=source_root,
                        release_secondary=release_secondary, release_primary=release_primary)
            update('complete', exit_code=0, active_gpus=[], completed_at=stamp())
            return 0
        except Exception as error:
            (job / 'error.log').write_text(traceback.format_exc())
            state(job, 'failed', 'failed', error=str(error), completed_at=stamp(), active_gpus=[])
            update('failed', exit_code=1, active_gpus=[], error=str(error), completed_at=stamp())
            return 1


def serve(queue, *, once=False, poll=10):
    require(socket.gethostname().split('.')[0] == 'landsend-1', 'GPU worker may run only on landsend-1')
    require(os.getuid() == 1003 and os.geteuid() == 1003, 'GPU worker must run as param (UID 1003), never root')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # The worker and all validation stay on CPU.
    (queue / 'jobs').mkdir(parents=True, exist_ok=True)
    with (queue / '.worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        recover_interrupted(queue)
        while not (queue / 'STOP').exists():
            pending = [job for job in jobs(queue) if job_status(job) == 'queued']
            if not pending:
                if once:
                    return
                time.sleep(poll)
                continue
            job = pending[0]
            try:
                req = request(job)
                clip, assets = source_clip(req['clip_id'])
                if not comparison_ready():
                    state(job, 'queued', 'waiting-for-comparison', request=req)
                    if once:
                        return
                    time.sleep(poll)
                    continue
                with gpu_pair() as idle:
                    if not idle:
                        state(job, 'queued', 'waiting-for-gpus', request=req)
                        if once:
                            return
                        time.sleep(poll)
                        continue
                    if (queue / 'STOP').exists():
                        return
                    run_job(job, req, clip, assets)
            except Exception as error:
                (job / 'error.log').write_text(traceback.format_exc())
                state(job, 'failed', 'failed', error=str(error), completed_at=stamp())
                if isinstance(error, StopRequested):
                    raise
            if once:
                return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('check', 'run', 'execute-job'))
    parser.add_argument('--queue-root', type=Path, default=QUEUE)
    parser.add_argument('--source-root', type=Path, default=SOURCE)
    parser.add_argument('--job-dir', type=Path, help='Private dispatcher execution directory containing request/state/assignment JSON')
    parser.add_argument('--node', choices=NODES, help='Required for execute-job; must match the actual host')
    parser.add_argument('--gpu-pair', help='Required for execute-job; two distinct IDs0–7, e.g.1,3')
    parser.add_argument('--once', action='store_true', help='Process at most one queued request; never wait for occupied GPUs')
    parser.add_argument('--poll-seconds', type=int, default=10, choices=range(1, 61), metavar='1..60')
    args = parser.parse_args()
    if args.command == 'execute-job' and (args.node is None or args.gpu_pair is None or args.job_dir is None):
        parser.error('execute-job requires --node, --gpu-pair and --job-dir')
    if args.command == 'check':
        clips = [source_clip(clip_id, args.source_root)[0]['id'] for clip_id in CLIP_IDS]
        print(json.dumps({'host': socket.gethostname(), 'allowed_hosts': NODES, 'gpu_ids': list(range(8)),
                          'comparison_complete': comparison_ready(args.source_root), 'source_clips': clips,
                          'queue': str(args.queue_root.resolve()), 'gpu_work_started': False}, indent=2))
        return
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, stop_signal)
    try:
        if args.command == 'execute-job':
            raise SystemExit(execute_job(args.job_dir.absolute(), args.node, parse_gpu_pair(args.gpu_pair),
                                         args.source_root.resolve()))
        serve(args.queue_root.resolve(), once=args.once, poll=args.poll_seconds)
    except StopRequested as error:
        print(str(error), flush=True)
    except BlockingIOError:
        raise SystemExit('Another request worker holds the queue lock')


if __name__ == '__main__':
    main()
