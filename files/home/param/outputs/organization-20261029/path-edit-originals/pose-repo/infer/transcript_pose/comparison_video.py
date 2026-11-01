"""Durable CPU coordinator and GPU worker for the 8 x 3 held-out video comparison."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback

DEFAULT_ROOT = Path('/home/param/pose-runs/heldout-comparison-20261028')
POSE_REPO = Path('/home/param/pose-repo')
VIDEO_REPO = Path('/home/param/vansh-video-models')
VIDEO_PY = VIDEO_REPO / '.venv/bin/python'
PUBLISH = Path('/home/param/pose-demo-website/comparison/publish.py')
SSH = ['ssh', '-i', '/home/param/.ssh/id_ed25519_cluster', '-o', 'BatchMode=yes']
RECIPES = ['aligned-media', 'compact-all', 'xabi-original']


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def stamp():
    return datetime.now(timezone.utc).isoformat(timespec='microseconds')


def execute(command, log=None, **kwargs):
    if log is None:
        return subprocess.run(command, check=True, **kwargs)
    with Path(log).open('a') as handle:
        return subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True, **kwargs)


def remote(node, command, **kwargs):
    return execute([*SSH, node, shlex.join([str(x) for x in command])], **kwargs)


def sync_to(node, paths):
    for value in paths:
        path = Path(value).resolve()
        if not path.is_relative_to('/home/param'):
            raise ValueError(f'Only our local assets may be copied: {path}')
        remote(node, ['mkdir', '-p', path.parent])
        execute(['rsync', '-aL', '-e', shlex.join(SSH), str(path), f'{node}:{path.parent}/'])


def pose_views(directory):
    """Create synchronized CPU views from the already decoded production raster."""
    from pose_views import build_pose_views

    directory = Path(directory)
    metadata = read(directory / 'ppm/meta.json') if (directory / 'ppm/meta.json').is_file() else {}
    prefill = int(metadata.get('prefill_frames', 0))
    # Recover an interrupted marker pass from its immutable clean backups
    # before asking the pose-view validator to inspect presentation outputs.
    if prefill and (directory / 'unmarked/pose-views.json').is_file():
        from prefill_marker import apply_prefill_marker
        apply_prefill_marker(directory, boundary_seconds=prefill / 12.5)
    build_pose_views(directory / 'video.mp4', directory / 'pose', out=directory)
    if prefill:
        from prefill_marker import apply_prefill_marker
        apply_prefill_marker(directory, boundary_seconds=prefill / 12.5)
    return ['--pose-video', directory / 'pose.mp4', '--pose-overlay-video', directory / 'overlay.mp4']


def jobs(root):
    selection = read(root / 'selection.json')
    if len(selection['clips']) != 8:
        raise ValueError('This rollout expects exactly eight selected clips')
    for model_index, recipe in enumerate(RECIPES):
        for ordinal, clip in enumerate(selection['clips']):
            directory = root / 'outputs' / clip['gallery_id'] / recipe
            yield {'root': str(root), 'clip': clip, 'recipe': recipe,
                   'node': f'landsend-{model_index + 2}', 'gpu': ordinal,
                   'directory': str(directory), 'ppm': str(directory / 'ppm'),
                   'assets': str(root / clip['id'] / 'video_assets.json')}


def public_state(job, status, message=''):
    root, directory = Path(job['root']), Path(job['directory'])
    write(directory / 'video_state.json', {'status': status, 'message': message,
                                         'updated_at': stamp(), 'node': job['node'], 'gpu': job['gpu']})
    state = 'failed' if status == 'failed' else 'generating'
    execute([sys.executable, str(PUBLISH), '--data-root', str(root / 'gallery'), 'status',
             '--clip-id', job['clip']['id'], '--model', job['recipe'], '--status', state, '--message', message])


def wait_for(root, condition):
    while not condition():
        if (root / 'STOP_VIDEO').exists():
            raise RuntimeError('STOP_VIDEO exists; no further stages will launch')
        time.sleep(10)
    if (root / 'STOP_VIDEO').exists():
        raise RuntimeError('STOP_VIDEO exists; no further stages will launch')


def complete(path):
    try:
        return read(path).get('status') == 'complete'
    except (OSError, ValueError):
        return False


@contextmanager
def idle_gpu(gpu):
    """Reserve our slot and block before launch while any GPU process is present."""
    locks = Path('/home/param/.cache/heldout-comparison-gpu-locks')
    locks.mkdir(parents=True, exist_ok=True)
    with (locks / f'gpu-{gpu}.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        while True:
            result = subprocess.run(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid',
                                     '--format=csv,noheader,nounits'], check=True, text=True, capture_output=True)
            memory = subprocess.run(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.used',
                                     '--format=csv,noheader,nounits'], check=True, text=True, capture_output=True)
            if not result.stdout.strip() and int(memory.stdout.strip()) < 1024:
                break
            print(f'GPU {gpu} occupied; waiting without launching', flush=True)
            time.sleep(10)
        yield


def worker(job, stage):
    import torch
    from safetensors.torch import load_file

    torch.set_num_threads(2)
    directory, ppm = Path(job['directory']), Path(job['ppm'])
    pose = directory / 'pose'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(job['gpu']),
               POSE_TOKENIZER_DIR='/home/param/models/pose-tokenizer-32cb-200k-noshift',
               ELECTRON_ENCODER='/home/param/models/electron-vae/encoder-v2-pose.safetensors',
               CUDA_HOME='/usr/local/cuda', OMP_NUM_THREADS='2',
               CPATH='/home/param/pose-repo/.venv/include/python3.12:/home/param/pose-repo/.venv/include',
               TORCHINDUCTOR_CACHE_DIR=f'/home/param/.cache/heldout-video/gpu-{job["gpu"]}/inductor',
               TRITON_CACHE_DIR=f'/home/param/.cache/heldout-video/gpu-{job["gpu"]}/triton')
    if stage == 'pose':
        if (pose / 'POSE_COMPLETE.json').is_file():
            return
        pose.mkdir(parents=True, exist_ok=True)
        for name in ('pose_tokens.safetensors', 'mimi.safetensors', 'audio.wav'):
            target = pose / name
            if not target.exists():
                target.symlink_to(ppm / name)
        with idle_gpu(job['gpu']):
            execute([str(VIDEO_PY), str(POSE_REPO / 'infer/transcript_pose/render.py'), str(pose),
                     '--frame0', job['clip']['frame0_pose'], '--skeleton-video'],
                    log=directory / 'pose.log', env=env)
        data = load_file(str(pose / 'pose_latents_electron_v3.safetensors'), device='cpu')
        assert tuple(data['latents'].shape) == (377, 48, 30, 16)
        assert int(data['valid_frames']) == 1504 and torch.isfinite(data['latents']).all()
        write(pose / 'POSE_COMPLETE.json', {'status': 'complete', 'frames': 1504,
                                           'latent_frames': 377, 'completed_at': stamp()})
        return
    if (directory / 'VIDEO_COMPLETE.json').is_file():
        return
    assets = read(job['assets'])
    if assets.get('status') != 'complete':
        raise ValueError('Reference assets are not complete')
    diffusion = directory / 'diffusion'
    if not complete(diffusion / 'generation/status.json'):
        if diffusion.exists():
            # Keep failed/interrupted attempts for review; never overwrite them.
            diffusion.rename(directory / f'diffusion-attempt-{time.time_ns()}')
        with idle_gpu(job['gpu']):
            execute([str(VIDEO_PY), str(POSE_REPO / 'infer/transcript_pose/render_video.py'), str(pose),
                     '--sink', assets['sink_latent'], '--reference-bank', assets['reference_bank'],
                     '--actor-id', assets['actor_id'], '--clip-key', assets['clip_key'],
                     '--seed', '2026', '--out', str(diffusion)], log=directory / 'video.log', env=env)
    receipt = read(diffusion / 'generation/status.json')
    assert receipt['status'] == 'complete' and receipt['decoded_frames'] == 1501
    for item in receipt['latent_files']:
        path = Path(item['path'])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
        tensors = load_file(str(path), device='cpu')
        assert all(torch.isfinite(tensor).all() for tensor in tensors.values())
    final = directory / 'video.mp4'
    temporary = directory / 'video.partial.mp4'
    execute(['ffmpeg', '-v', 'error', '-y', '-i', str(diffusion / 'generation/video.mp4'),
             '-map', '0:v:0', '-map', '0:a:0', '-vf', 'trim=end_frame=1500,setpts=PTS-STARTPTS',
             '-af', 'atrim=end=60,asetpts=PTS-STARTPTS', '-frames:v', '1500', '-t', '60',
             '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18', '-pix_fmt', 'yuv420p',
             '-threads', '2', '-c:a', 'aac', '-ar', '24000', '-movflags', '+faststart', str(temporary)])
    info = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(temporary)]))
    video = next(s for s in info['streams'] if s['codec_type'] == 'video')
    audio = next(s for s in info['streams'] if s['codec_type'] == 'audio')
    assert int(video['nb_frames']) == 1500 and abs(float(video['duration']) - 60) < .001
    assert abs(float(audio['duration']) - 60) < .05
    execute(['ffmpeg', '-v', 'error', '-i', str(temporary), '-f', 'null', '-'])
    temporary.replace(final)
    write(directory / 'VIDEO_COMPLETE.json', {'status': 'complete', 'completed_at': stamp(),
          'sha256': hashlib.sha256(final.read_bytes()).hexdigest(), 'frames': 1500, 'seconds': 60.0,
          'source_generated_frames': 1501, 'postprocessing': 'Crop final one generated RGB frame; trim audio to 60s; no padding or looping',
          'node': job['node'], 'gpu': job['gpu'], 'video_model': 'Shared raw step-2911 nine-reference student',
          'video_source': 'param/bimodal@26cedf69db7257658de7f6e53bae4972dac0f20b',
          'reference_assets': assets, 'diffusion_receipt': receipt})


def run_job(job):
    root, directory = Path(job['root']), Path(job['directory'])
    directory.mkdir(parents=True, exist_ok=True)
    config = directory / 'video_job.json'
    write(config, job)
    try:
        if not (directory / 'VIDEO_COMPLETE.json').is_file():
            public_state(job, 'waiting-ppm', 'Waiting for audio and pose generation')
            wait_for(root, lambda: (Path(job['ppm']) / 'PPM_COMPLETE.json').is_file())
            if job['node'] == 'landsend-2' and job['gpu'] == 0:
                # Root reserves this device for bank encoding/identity checks.
                wait_for(root, lambda: (root / 'RELEASE_NODE2_GPU0').is_file())
            public_state(job, 'rendering-pose', 'Rendering generated pose')
            sync_to(job['node'], [config, Path(job['ppm']), job['clip']['frame0_pose']])
            remote(job['node'], [VIDEO_PY, Path(__file__).resolve(), 'worker', '--job', config, '--stage', 'pose'],
                   log=directory / 'remote-pose.log')
            public_state(job, 'waiting-references', 'Preparing appearance reference images')
            wait_for(root, lambda: complete(job['assets']))
            assets = read(job['assets'])
            copies = [job['assets'], assets['sink_latent'], assets['reference_bank']]
            bank_provenance = Path(assets['reference_bank']).parent / 'reference-bank-provenance.json'
            if bank_provenance.is_file():
                copies.append(bank_provenance)
            sync_to(job['node'], copies)
            public_state(job, 'rendering-video', 'Generating video')
            remote(job['node'], [VIDEO_PY, Path(__file__).resolve(), 'worker', '--job', config, '--stage', 'video'],
                   log=directory / 'remote-video.log')
            execute(['rsync', '-a', '-e', shlex.join(SSH), '--exclude=ppm', f'{job["node"]}:{directory}/', str(directory) + '/'])
        completed = read(directory / 'VIDEO_COMPLETE.json')
        if completed['status'] != 'complete':
            raise ValueError('Video completion receipt invalid')
        final = directory / 'video.mp4'
        if hashlib.sha256(final.read_bytes()).hexdigest() != completed['sha256']:
            raise ValueError('Copied video checksum mismatch')
        metadata = {'recipe': job['recipe'], 'dataset_split': 'Production Xabi eval',
                    'voice_reference_frames': job['clip']['voice_clip_frames'], 'duration_seconds': 60,
                    'video_model': completed['video_model'], 'seed': read(Path(job['ppm']) / 'PPM_COMPLETE.json').get('seed'),
                    'reference_conditioning': 'Nine freshly encoded reference images from this recording and speaker; identical across the three models',
                    'postprocessing': completed['postprocessing']}
        reference_selection = read(root / job['clip']['id'] / 'reference-selection.json')
        metadata['references_from_evaluation_window'] = sum(bool(r['inside_evaluation_window']) for r in reference_selection['rows'])
        metadata['reference_images'] = [{'property': r['property'], 'source_timestamp_seconds': r['source_timestamp_s'],
                                         'inside_evaluation_window': r['inside_evaluation_window'],
                                         'source_sha256': r['source_sha256']} for r in reference_selection['rows']]
        metadata['evaluation_scope'] = 'Held-out PPM recording; reference-image-conditioned video demonstration, not leakage-free video evaluation'
        add_public_provenance(metadata, completed['reference_assets'])
        public_metadata = directory / 'public-metadata.json'
        write(public_metadata, metadata)
        execute([sys.executable, str(PUBLISH), '--data-root', str(root / 'gallery'), 'publish',
                 '--clip-id', job['clip']['id'], '--model', job['recipe'], '--video', str(final),
                 '--metadata', str(public_metadata), *map(str, pose_views(directory))])
        write(directory / 'video_state.json', {'status': 'complete', 'updated_at': stamp(), 'published': True})
        return {'clip': job['clip']['id'], 'recipe': job['recipe'], 'status': 'complete'}
    except Exception as error:
        (directory / 'video-error.txt').write_text(traceback.format_exc())
        public_state(job, 'failed', str(error))
        return {'clip': job['clip']['id'], 'recipe': job['recipe'], 'status': 'failed', 'error': str(error)}


def add_public_provenance(metadata, assets):
    provenance = assets.get('provenance', {})
    if isinstance(provenance, str):
        provenance = read(provenance)
    allowed = ('exact_source_track_matches', 'actor_mapping_source', 'source_pngs_verified_sha256_and_decoded_pixels',
               'property_crops', 'reference_quality', 'limitation', 'visual_correspondence_status')
    metadata['reference_provenance'] = {key: provenance[key] for key in allowed if key in provenance}
    metadata['strict_bank_byte_identity'] = bool(assets.get('strict_bank_byte_identity', False))
    metadata['reference_evaluation_window_overlap_count'] = assets['reference_evaluation_window_overlap_count']
    metadata['visual_reference_source'] = 'Nine same-track HF reference PNGs, freshly encoded with pinned Wan2.2 VAE; canonical property-specific crops'


def refresh_metadata(root):
    """Repair public metadata independently of active GPU workers/coordinator."""
    for job in jobs(root):
        directory = Path(job['directory'])
        public_path = directory / 'public-metadata.json'
        if not public_path.is_file() or not (directory / 'VIDEO_COMPLETE.json').is_file():
            continue
        metadata = read(public_path)
        add_public_provenance(metadata, read(job['assets']))
        write(public_path, metadata)
        manifest_path = root / 'gallery/manifest.json'
        with (root / 'gallery/.publish.lock').open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            manifest = read(manifest_path)
            clip = next(c for c in manifest['clips'] if c['id'] == job['clip']['id'])
            output = clip['outputs'].get(job['recipe'])
            if output and output.get('status') == 'complete' and output.get('metadata') != metadata:
                output['metadata'] = metadata
                manifest['updated_at'] = stamp()
                write(manifest_path, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for mode in ('plan', 'run', 'refresh-metadata'):
        p = sub.add_parser(mode)
        p.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    w = sub.add_parser('worker')
    w.add_argument('--job', required=True, type=Path)
    w.add_argument('--stage', required=True, choices=['pose', 'video'])
    args = parser.parse_args()
    if args.command == 'worker':
        worker(read(args.job), args.stage)
        return
    planned = list(jobs(args.root.resolve()))
    if args.command == 'refresh-metadata':
        refresh_metadata(args.root.resolve())
        return
    if args.command == 'plan':
        print(json.dumps([{'clip': j['clip']['id'], 'recipe': j['recipe'], 'node': j['node'], 'gpu': j['gpu'],
                           'ppm_ready': (Path(j['ppm']) / 'PPM_COMPLETE.json').exists(),
                           'references_ready': complete(j['assets'])} for j in planned], indent=2))
        return
    with ThreadPoolExecutor(max_workers=24) as pool:
        futures = [pool.submit(run_job, j) for j in planned]
        outcomes = [future.result() for future in as_completed(futures)]
    write(args.root / 'video-batch-status.json', {'completed_at': stamp(), 'outcomes': outcomes})
    if any(o['status'] != 'complete' for o in outcomes):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
