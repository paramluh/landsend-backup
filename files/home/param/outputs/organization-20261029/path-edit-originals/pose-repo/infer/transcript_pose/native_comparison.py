#!/usr/bin/env python3
"""Collect, validate, render and publish the sixteen prelaunched native jobs.

The run's execution-plan.json records one B1 PPM job per GPU pair on nodes1–4.
Its CPU launcher writes teams/<clip>-<recipe>.ppm-exit.json after each PPM
process exits. This coordinator immediately starts that job's video stages on
the first GPU of its released pair. It never starts or interrupts PPM jobs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib
import os
from pathlib import Path
import shlex
import time
import traceback

import comparison_video as video
import request_worker as worker
from request_schema import normalize_request

ROOT = Path('/home/param/pose-runs/native-comparison-20261028')
REPO = Path(__file__).resolve().parents[2]


def state(job, stage, message=''):
    video.write(Path(job['directory']) / 'video_state.json',
                dict(status=stage, message=message, updated_at=video.stamp(), node=job['node'], gpu=job['gpu']))
    # Publication can be deployed while existing GPU work is in progress.
    if stage != 'complete' and (Path(job['root']) / 'gallery/manifest.json').exists() and (REPO / 'infer/transcript_pose/publish_native.py').exists():
        try:
            importlib.import_module('publish_native').publish_native_state(job, stage, message)
        except Exception as exc:
            print(f"Gallery status update pending: {exc}", flush=True)


def wait_for_ppm(root, job):
    path = root / 'teams' / f"{job['clip']['id']}-{job['recipe']}.ppm-exit.json"
    while True:
        if (root / 'STOP').exists():
            raise RuntimeError('STOP exists; no subsequent stage will launch')
        try:
            receipt = video.read(path)
        except (OSError, ValueError):
            time.sleep(5)
            continue
        if receipt['exit_code'] != 0:
            raise RuntimeError(f"PPM exited {receipt['exit_code']}; inspect {job['clip']['id']}-{job['recipe']}.ppm.log")
        return receipt


def validate_ppm(job):
    directory, clip = Path(job['directory']), job['clip']
    req = normalize_request(dict(recipe=job['recipe'], clip_id=clip['id'], text_mode='native',
                                 duration_seconds=60, seed=20261028 + int(clip['gallery_id']) - 1,
                                 video_seed=2026, text_temp=1, text_topk=0,
                                 audio_temp=1, audio_topk=0, pose_temp=1, pose_topk=0))
    validation = worker.validate_ppm(directory, req, clip)
    meta = video.read(directory / 'ppm/meta.json')
    receipt = dict(validation)
    receipt.update(status='complete', completed_at=video.stamp(), exit_code=0,
                   recipe=job['recipe'], clip_index=clip['clip_index'], frames=752,
                   text_mode='native', force_text=False, batch_size=1,
                   seed=req['seed'], request=req)
    receipt['generated_lane_sha256'] = meta['generated_lane_sha256']
    receipt['ground_truth_generation_media'] = False
    video.write(directory / 'ppm/PPM_COMPLETE.json', receipt)
    return receipt


def finish(job):
    root, directory = Path(job['root']), Path(job['directory'])
    name = f"{job['clip']['id']}-{job['recipe']}"
    video.write(root / 'teams' / f'{name}.status.json', dict(status='running', job=name, updated_at=video.stamp()))
    try:
        if job['node'] not in [f'landsend-{i}' for i in range(1, 5)]:
            raise ValueError('GPU stages are restricted to landsend-1 through4')
        if job['recipe'] not in ('compact-all', 'xabi-original'):
            raise ValueError('Native comparison supports compact and Xabi checkpoints')
        wait_for_ppm(root, job)
        ppm = Path(job['ppm'])
        ppm.mkdir(parents=True, exist_ok=True)
        video.execute(['rsync', '-a', '-e', shlex.join(video.SSH), f"{job['node']}:{ppm}/", str(ppm) + '/'])
        validate_ppm(job)
        print(f'{name}: PPM complete and validated', flush=True)
        config = root / 'teams' / f'{name}.json'
        video.write(config, job)
        video.sync_to(job['node'], [config, ppm / 'PPM_COMPLETE.json'])
        if not (directory / 'VIDEO_COMPLETE.json').exists():
            for stage, label in [('pose', 'rendering-pose'), ('video', 'rendering-video')]:
                if (root / 'STOP').exists():
                    raise RuntimeError('STOP exists; no subsequent stage will launch')
                state(job, label, 'Rendering generated pose' if stage == 'pose' else 'Generating native video')
                video.remote(job['node'], [video.VIDEO_PY, REPO / 'infer/transcript_pose/comparison_video.py',
                                          'worker', '--job', config, '--stage', stage],
                             log=directory / f'remote-{stage}.log')
            video.execute(['rsync', '-a', '-e', shlex.join(video.SSH), '--exclude=ppm',
                           f"{job['node']}:{directory}/", str(directory) + '/'])
        completed = video.read(directory / 'VIDEO_COMPLETE.json')
        worker.validate_video_receipt(completed['diffusion_receipt'], 752, 2026)
        if completed['frames'] != 1500 or completed['seconds'] != 60:
            raise ValueError('Final video must contain1500frames/60seconds')
        # This validates the artifact hash again and publishes atomically to the
        # local and node1 galleries without replacing another job's manifest.
        importlib.import_module('publish_native').publish_native(job)
        result = dict(status='complete', job=name, recipe=job['recipe'], clip_id=job['clip']['id'],
                      node=job['node'], gpu=job['gpu'], completed_at=video.stamp())
    except Exception as exc:
        (root / 'logs' / f'{name}.pipeline-error.txt').write_text(traceback.format_exc())
        state(job, 'failed', str(exc))
        result = dict(status='failed', job=name, error=str(exc), completed_at=video.stamp())
    video.write(root / 'teams' / f'{name}.status.json', result)
    print(result, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--retry-failed', action='store_true', help='Retry only failed collection/render/publication jobs; never rerun PPM')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Coordinator/validation are CPU-only.
    root = args.root.resolve()
    planned = video.read(root / 'execution-plan.json')
    assert len(planned) == 16 and len({(j['recipe'], j['clip']['id']) for j in planned}) == 16
    jobs = planned
    if args.retry_failed:
        jobs = [j for j in planned if video.complete(root / 'teams' / f"{j['clip']['id']}-{j['recipe']}.status.json") is False
                and (root / 'teams' / f"{j['clip']['id']}-{j['recipe']}.status.json").exists()
                and video.read(root / 'teams' / f"{j['clip']['id']}-{j['recipe']}.status.json").get('status') == 'failed']
    # Transformers resolves this class lazily; concurrent first imports can race
    # inside its lazy module loader. Resolve it once before parallel validation.
    from transformers import AutoTokenizer  # noqa: F401
    with ThreadPoolExecutor(max_workers=16) as pool:
        results = [future.result() for future in as_completed([pool.submit(finish, job) for job in jobs])]
    combined = []
    for job in planned:
        path = root / 'teams' / f"{job['clip']['id']}-{job['recipe']}.status.json"
        combined.append(video.read(path) if path.exists() else dict(status='running'))
    passed = all(r['status'] == 'complete' for r in combined)
    video.write(root / 'NATIVE_COMPARISON_COMPLETE.json',
                dict(status='complete' if passed else 'in-progress', completed_at=video.stamp(), jobs=combined))
    if any(r['status'] == 'failed' for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
