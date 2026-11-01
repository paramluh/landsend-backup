"""Run the controlled comparison on 16 GPU pairs on landsend-1 through 4.

Eight B=2 teams and eight B=1 teams produce 24 PPM samples. Each team releases
its PPM process before rendering its sample(s) on individual GPUs in that pair.
The controller itself is CPU-only and may run on landsend-0.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback

import numpy as np
import soundfile as sf

import comparison_video as video

ROOT = Path('/home/param/pose-runs/controlled-comparison-20261028')
REPO = Path('/home/param/pose-repo')


def teams(root):
    clips = video.read(root / 'selection.json')['clips']
    if len(clips) != 8 or any(c['frames'] != 752 or c['voice_clip_frames'] != 100 for c in clips):
        raise ValueError('Expected eight clips, 752 output frames and 100 common voice frames each')
    for node, recipe in [('landsend-1', 'aligned-media'), ('landsend-2', 'compact-all')]:
        for index in range(4):
            yield dict(node=node, recipe=recipe, devs=[2 * index, 2 * index + 1], clips=clips[2 * index:2 * index + 2])
    for node, offset in [('landsend-3', 0), ('landsend-4', 4)]:
        for index in range(4):
            yield dict(node=node, recipe='xabi-original', devs=[2 * index, 2 * index + 1], clips=[clips[offset + index]])


def output_dir(root, clip, recipe):
    return root / 'outputs' / clip['gallery_id'] / recipe


def check_ppm(path, clip, recipe, seed, batch_size):
    meta = video.read(path / 'meta.json')
    if (meta['recipe'] != recipe or meta['clip_index'] != clip['clip_index'] or
            meta['frames'] != 752 or meta['seed'] != seed or meta['batch_size'] != batch_size):
        raise ValueError('PPM identity/batch metadata mismatch')
    assert meta['force_text'] is True and meta['prefill_frames'] == 0
    assert meta['ground_truth_generation_media'] is False and meta['voice_clip_frames'] == 100
    sampling = meta['sampling']
    assert all(sampling[key] == 1 for key in ('cb0_temp', 'light_temp', 'pose_temp', 'pose_light_temp'))
    assert all(sampling[key] == 0 for key in ('cb0_topk', 'light_topk', 'pose_topk', 'pose_light_topk'))
    voice = np.load(clip['voice_clip'], allow_pickle=False)
    voice_hash = hashlib.sha256(np.ascontiguousarray(voice, dtype=np.int64).tobytes()).hexdigest()
    assert meta['voice_codes_sha256'] == voice_hash
    expected_lane = np.load(clip['lane'], allow_pickle=False)
    with np.load(path / 'tokens.npz', allow_pickle=False) as data:
        assert np.array_equal(data['lane'], expected_lane), 'Output text lane differs from common supplied text'
        for name in ('audio_codes', 'pose_codes'):
            codes = data[name]
            assert codes.shape == (32, 752) and codes.min() >= 0 and codes.max() < 2048
    audio, sample_rate = sf.read(path / 'audio.wav')
    assert sample_rate == 24000 and len(audio) == 752 * 1920 and np.isfinite(audio).all()
    receipt = {'status': 'complete', 'recipe': recipe, 'clip_index': clip['clip_index'], 'seed': seed,
               'frames': 752, 'batch_size': batch_size, 'force_text': True, 'voice_codes_sha256': voice_hash,
               'framewise_lane_sha256': meta['framewise_lane_sha256'], 'ground_truth_generation_media': False,
               'sampling': sampling,
               'completed_at': video.stamp(), 'exit_code': 0}
    video.write(path / 'PPM_COMPLETE.json', receipt)
    return receipt


def publish(job):
    directory = Path(job['directory'])
    receipt = video.read(directory / 'VIDEO_COMPLETE.json')
    final = directory / 'video.mp4'
    assert receipt['status'] == 'complete' and receipt['frames'] == 1500 and receipt['seconds'] == 60
    assert hashlib.sha256(final.read_bytes()).hexdigest() == receipt['sha256']
    ppm = video.read(directory / 'ppm/PPM_COMPLETE.json')
    clip = job['clip']
    metadata = {'recipe': job['recipe'], 'dataset_split': 'Production Xabi eval', 'duration_seconds': 60,
                'voice_reference_seconds': 8, 'voice_codes_sha256': ppm['voice_codes_sha256'],
                'framewise_lane_sha256': ppm['framewise_lane_sha256'], 'force_text': True,
                'ground_truth_generation_media': False, 'seed': ppm['seed'], 'batch_size': ppm['batch_size'],
                'conditioning_control': 'Identical original-speaker reference codes and identical framewise text for all three models',
                'voice_reference_nonoverlapping_generation': True,
                'sampling': {'audio_temperature': ppm['sampling']['cb0_temp'], 'audio_top_k': ppm['sampling']['cb0_topk'],
                             'pose_temperature': ppm['sampling']['pose_temp'], 'pose_top_k': ppm['sampling']['pose_topk'],
                             'residual_codebooks_use_same_settings': True, 'top_k_zero_means': 'Unfiltered'},
                'video_model': receipt['video_model'], 'postprocessing': receipt['postprocessing'],
                'evaluation_scope': 'Held-out PPM recording; image-conditioned video demonstration, not leakage-free video evaluation'}
    video.add_public_provenance(metadata, video.read(job['assets']))
    selection = video.read(Path(job['root']) / clip['id'] / 'reference-selection.json')
    metadata['reference_images'] = [{'property': row['property'], 'source_timestamp_seconds': row['source_timestamp_s'],
                                    'inside_evaluation_window': row['inside_evaluation_window'],
                                    'source_sha256': row['source_sha256']} for row in selection['rows']]
    path = directory / 'public-metadata.json'
    video.write(path, metadata)
    video.execute([sys.executable, str(video.PUBLISH), '--data-root', str(Path(job['root']) / 'gallery'), 'publish',
                   '--clip-id', clip['id'], '--model', job['recipe'], '--video', str(final), '--metadata', str(path),
                   *map(str, video.pose_views(directory))])
    video.write(directory / 'video_state.json', {'status': 'complete', 'published': True, 'updated_at': video.stamp()})


def run_video(job):
    node, directory = job['node'], Path(job['directory'])
    if (directory / 'VIDEO_COMPLETE.json').exists():
        publish(job)
        return
    config = directory / 'video_job.json'
    video.write(config, job)
    video.sync_to(node, [config, directory / 'ppm/PPM_COMPLETE.json'])
    video.public_state(job, 'rendering-pose', 'Rendering generated pose')
    video.remote(node, [video.VIDEO_PY, REPO / 'infer/transcript_pose/comparison_video.py',
                        'worker', '--job', config, '--stage', 'pose'], log=directory / 'remote-pose.log')
    if not video.complete(job['assets']):
        raise ValueError('Matching reference assets have not passed validation')
    video.public_state(job, 'rendering-video', 'Generating video')
    video.remote(node, [video.VIDEO_PY, REPO / 'infer/transcript_pose/comparison_video.py',
                        'worker', '--job', config, '--stage', 'video'], log=directory / 'remote-video.log')
    video.execute(['rsync', '-a', '-e', shlex.join(video.SSH), '--exclude=ppm', f'{node}:{directory}/', str(directory) + '/'])
    publish(job)


def guard_pair(team):
    # Execute a real conditional check; never print utilization then launch anyway.
    code = '''import subprocess,sys
for gpu in sys.argv[1:]:
 p=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 m=int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
 if p or m>=1024:raise SystemExit('GPU '+gpu+' occupied; no PPM launch')
'''
    video.remote(team['node'], ['python3', '-c', code, *team['devs']])


def run_team(root, team):
    recipe, node = team['recipe'], team['node']
    if node not in ('landsend-1', 'landsend-2', 'landsend-3', 'landsend-4'):
        raise ValueError('GPU work is forbidden on this node')
    name = f'{node}-{team["devs"][0]}-{recipe}'
    logs = root / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    jobs, batch = [], []
    for row, clip in enumerate(team['clips']):
        directory = output_dir(root, clip, recipe)
        directory.mkdir(parents=True, exist_ok=True)
        seed = 20261028 + int(clip['gallery_id']) - 1
        jobs.append({'root': str(root), 'clip': clip, 'recipe': recipe, 'node': node, 'gpu': team['devs'][row],
                     'directory': str(directory), 'ppm': str(directory / 'ppm'),
                     'assets': str(root / clip['id'] / 'video_assets.json')})
        batch.append({'clip_index': clip['clip_index'], 'start_frame': clip['start_frame'], 'frames': 752,
                      'voice_clip': clip['voice_clip'], 'seed': seed, 'out': str(directory / 'ppm')})
    try:
        ready = all((Path(job['ppm']) / 'PPM_COMPLETE.json').exists() for job in jobs)
        if not ready:
            if (root / 'STOP').exists():
                raise RuntimeError('STOP exists; no team launch')
            for job in jobs:
                video.public_state(job, 'generating-ppm', 'Generating audio and pose with the common voice and text')
            config = root / 'teams' / f'{name}.json'
            video.write(config, batch)
            # Data stays on its PPM team node for rendering after the model exits.
            video.sync_to(node, [config, *[root / c['id'] for c in team['clips']]])
            guard_pair(team)
            command = 'source ' + shlex.quote(str(REPO / 'finetune/transcript_pose/environment/activate.sh'))
            command += ' && cd ' + shlex.quote(str(REPO))
            command += ' && OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1 MIMI_DIR=/home/param/models/mimi'
            command += ' FA3_DIR=/home/param/pose-repo/.cache/kernels/vllm-flash-attn3'
            command += ' CUDA_VISIBLE_DEVICES=' + ','.join(map(str, team['devs'])) + ' ' + shlex.join([
                str(REPO / '.venv/bin/python'), '-u', 'infer/transcript_pose/infer_batch.py', '--recipe', recipe,
                '--batch-json', str(config), '--devs', '0,1',
                '--audio-temp', '1', '--pose-temp', '1', '--audio-topk', '0', '--pose-topk', '0'])
            video.remote(node, ['bash', '-lc', command], log=logs / f'{name}.ppm.log')
            for job, item in zip(jobs, batch):
                dest = Path(job['ppm'])
                dest.mkdir(parents=True, exist_ok=True)
                video.execute(['rsync', '-a', '-e', shlex.join(video.SSH), f'{node}:{dest}/', str(dest) + '/'])
                check_ppm(dest, job['clip'], recipe, item['seed'], len(jobs))
        else:
            for job, item in zip(jobs, batch):
                check_ppm(Path(job['ppm']), job['clip'], recipe, item['seed'], len(jobs))
        # PPM subprocess has exited: B2 pairs become two independent B1 video jobs.
        if (root / 'STOP').exists():
            raise RuntimeError('STOP exists; no video stage launch')
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            for future in as_completed([pool.submit(run_video, job) for job in jobs]):
                future.result()
        result = {'team': name, 'status': 'complete', 'clips': [c['id'] for c in team['clips']],
                  'recipe': recipe, 'batch_size': len(jobs), 'completed_at': video.stamp()}
    except Exception as error:
        (logs / f'{name}.error.txt').write_text(traceback.format_exc())
        for job in jobs:
            if not (Path(job['directory']) / 'VIDEO_COMPLETE.json').exists():
                video.public_state(job, 'failed', str(error))
        result = {'team': name, 'status': 'failed', 'error': str(error), 'completed_at': video.stamp()}
    video.write(root / 'teams' / f'{name}.status.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'run'])
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    planned = list(teams(root))
    assert len(planned) == 16 and sum(len(t['clips']) for t in planned) == 24
    assert len({(t['node'], g) for t in planned for g in t['devs']}) == 32
    if args.command == 'plan':
        print(json.dumps([dict(node=t['node'], recipe=t['recipe'], devs=t['devs'], batch_size=len(t['clips']),
                               clips=[c['id'] for c in t['clips']]) for t in planned], indent=2))
        return
    # This process must never accidentally select a GPU on its controller host.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    with ThreadPoolExecutor(max_workers=16) as pool:
        results = [f.result() for f in as_completed([pool.submit(run_team, root, team) for team in planned])]
    video.write(root / 'CONTROLLED_COMPARISON_COMPLETE.json', {'status': 'complete' if all(r['status'] == 'complete' for r in results) else 'failed',
              'completed_at': video.stamp(), 'teams': results})
    if any(r['status'] != 'complete' for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
