"""Durable 24-cell native-format comparison with full-model silent-audio guidance.

The CPU controller runs sixteen teams on landsend-1..4. Each detached team
generates on a GPU pair, then its released GPUs independently render videos.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import tarfile
import time
import traceback

import numpy as np
import soundfile as sf

import comparison_video as video
from controlled_comparison import teams

ROOT = Path('/home/param/pose-runs/audio-guided-native-comparison-20261028')
PREFILL_ROOT = Path('/home/param/pose-runs/audio-guided-prefill2s-comparison-20261028')
ALLOWED_ROOTS = (ROOT, PREFILL_ROOT)
SOURCE = Path('/home/param/pose-runs/controlled-comparison-20261028')
REPO = Path('/home/param/pose-repo')
NODES = tuple(f'landsend-{i}' for i in range(1, 5))
SAMPLE = dict(text_temp=1, text_topk=0, audio_temp=1, pose_temp=1,
              audio_topk=30, pose_topk=30, pose_light_topk=50, pose_audio_guidance=3)


def prefill_frames(root):
    if root not in ALLOWED_ROOTS:
        raise ValueError(f'Unexpected comparison root: {root}')
    return 25 if root == PREFILL_ROOT else 0


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def pid_record(pid):
    start = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    return dict(pid=pid, start_ticks=start, host=socket.gethostname().split('.')[0])


def alive(record):
    try:
        fields = Path(f"/proc/{record['pid']}/stat").read_text().rsplit(')', 1)[1].split()
        return fields[19] == record['start_ticks'] and fields[0] != 'Z'
    except (OSError, KeyError):
        return False


def team_name(team):
    return f"{team['node']}-{team['devs'][0]}-{team['recipe']}"


def build_jobs(root, team):
    result = []
    for row, clip in enumerate(team['clips']):
        directory = root / 'outputs' / clip['gallery_id'] / team['recipe']
        result.append(dict(root=str(root), run_root=str(root), clip=clip, clip_id=clip['id'],
                           recipe=team['recipe'], node=team['node'], gpu=team['devs'][row],
                           devs=team['devs'], batch_size=len(team['clips']), batch_row=row,
                           text_mode='forced' if team['recipe'] == 'aligned-media' else 'native',
                           seed=20261028 + int(clip['gallery_id']) - 1, video_seed=2026,
                           prefill_frames=prefill_frames(root),
                           directory=str(directory), ppm=str(directory / 'ppm'),
                           assets=clip['video_assets'], selection=str(root / 'selection.json')))
    return result


def prepare(root):
    count = prefill_frames(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / 'logs').mkdir(parents=True, exist_ok=True)
    selection = video.read(SOURCE / 'selection.json')
    selection.update(comparison='Native formats; full-model silent-audio pose guidance',
                     source_selection=str(SOURCE / 'selection.json'),
                     source_selection_sha256=digest(SOURCE / 'selection.json'),
                     conditioning_control='Same voice and appearance; aligned forced text, compact sampled text with transcript, Xabi sampled text without transcript',
                     sampling=SAMPLE, pose_guidance='4 * audio-conditioned logits - 3 * silence-conditioned logits, all32 pose codebooks',
                     prefill_frames=count, prefill_duration_seconds=count / 12.5,
                     prefill_modalities=['audio', 'pose', 'text'] if count else [],
                     continuation_seconds=60 - count / 12.5)
    if count:
        from request_worker import source_prefill, array_hashes
        for clip in selection['clips']:
            values = source_prefill(clip, count)
            target = root / 'conditioning' / f"{clip['id']}-prefill.npz"
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, **values)
            clip.update(prefill_source=str(target), prefill_frames=count, prefill_sha256=array_hashes(values))
    saved = root / 'selection.json'
    if saved.exists() and video.read(saved) != selection:
        raise ValueError('Existing selection differs; do not overwrite a run')
    video.write(saved, selection)
    planned = list(teams(root))
    assert len(planned) == 16 and sum(len(t['clips']) for t in planned) == 24
    assert len({(t['node'], g) for t in planned for g in t['devs']}) == 32
    jobs = []
    for team in planned:
        name = team_name(team)
        team.update(root=str(root), prefill_frames=count, jobs=build_jobs(root, team))
        for job in team['jobs']:
            Path(job['directory']).mkdir(parents=True, exist_ok=True)
            for key in ('voice_clip', 'words', 'lane', 'frame0_pose', 'sink_latent', 'reference_bank', 'video_assets'):
                if not Path(job['clip'][key]).exists():
                    raise FileNotFoundError(job['clip'][key])
        batch = [dict(clip_index=j['clip']['clip_index'], start_frame=j['clip']['start_frame'], frames=752,
                      voice_clip=j['clip']['voice_clip'], seed=j['seed'], out=j['ppm'], prefill_frames=count) for j in team['jobs']]
        team['batch_json'] = str(root / 'teams' / f'{name}.batch.json')
        video.write(team['batch_json'], batch)
        video.write(root / 'teams' / f'{name}.json', team)
        jobs.extend(team['jobs'])
    video.write(root / 'execution-plan.json', planned)
    video.write(root / 'jobs.json', jobs)
    print(json.dumps(dict(root=str(root), teams=len(planned), samples=len(jobs), sampling=SAMPLE)), flush=True)
    return planned


def sync_sources(root):
    planned = video.read(root / 'execution-plan.json')
    source_paths = sorted([* (REPO / 'infer/transcript_pose').glob('*.py'),
                           * (REPO / 'infer/engine/ppm').rglob('*.py'),
                           REPO / 'finetune/transcript_pose/environment/activate.sh', video.PUBLISH,
                           video.PUBLISH.parent / 'request_server.py', video.PUBLISH.parent / 'studio.html',
                           video.PUBLISH.parent / 'index.html', video.PUBLISH.parent / 'pose-player.js',
                           video.PUBLISH.parent / 'pose-player.css'])
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    snapshot = root / 'source' / stamp
    snapshot.mkdir(parents=True, exist_ok=False)
    hashes = {str(p): digest(p) for p in source_paths}
    with tarfile.open(snapshot / 'source.tar.gz', 'w:gz') as archive:
        for path in source_paths:
            archive.add(path, arcname=str(path.relative_to('/home/param')))
    video.write(snapshot / 'sha256.json', hashes)
    def sync_node(node):
        own = [team for team in planned if team['node'] == node]
        clips = {c['id']: c for team in own for c in team['clips']}
        # Source and original inputs only: no model weights, environments or credentials.
        video.remote(node, ['mkdir', '-p', root / 'teams', root / 'logs'])
        for folder in ('infer/transcript_pose', 'infer/engine/ppm'):
            video.execute(['rsync', '-a', '--exclude=__pycache__', '--include=*/', '--include=*.py', '--exclude=*',
                           '-e', shlex.join(video.SSH), str(REPO / folder) + '/', f'{node}:{REPO / folder}/'])
        video.sync_to(node, [REPO / 'finetune/transcript_pose/environment/activate.sh', root / 'selection.json',
                             root / 'execution-plan.json', root / 'jobs.json', root / 'teams',
                             *[Path(c['input_assets']) for c in clips.values()]])
        if (root / 'conditioning').exists():
            video.sync_to(node, [root / 'conditioning'])
        probe = "import pathlib,json,sys; p=pathlib.Path(sys.argv[1]); m=json.loads((p/'meta.json').read_text()); assert (p/'COMPLETE').is_file() and (p/'rep.pt').is_file() and all((p/f'rank{i}.pt').is_file() for i in range(m['world'])); print(p.name, m['step'], m['world'])"
        for recipe in {team['recipe'] for team in own}:
            video.remote(node, ['python3', '-c', probe, f'/home/param/models/transcript-pose/{recipe}/export-200'])
        return dict(node=node, recipes=sorted({t['recipe'] for t in own}), clips=sorted(clips))
    with ThreadPoolExecutor(max_workers=4) as pool:
        checks = list(pool.map(sync_node, NODES))
    video.write(root / 'source-sync.json', dict(status='complete', completed_at=video.stamp(),
                snapshot=str(snapshot), source_sha256=hashes, nodes=checks))


def check_ppm(job):
    from safetensors.numpy import load_file
    from request_worker import validate_inputs, source_prefill
    path, clip = Path(job['ppm']), job['clip']
    meta = video.read(path / 'meta.json')
    forced = job['text_mode'] == 'forced'
    count = job.get('prefill_frames', 0)
    expected = dict(recipe=job['recipe'], clip_index=clip['clip_index'], frames=752, seed=job['seed'],
                    batch_size=job['batch_size'], batch_row=job['batch_row'], text_mode=job['text_mode'],
                    force_text=forced, prefill_frames=count, ground_truth_generation_media=False, voice_clip_frames=100)
    assert all(meta.get(k) == v for k, v in expected.items()), 'PPM identity/conditioning mismatch'
    assert meta['checkpoint']['ck'] == f"/home/param/models/transcript-pose/{job['recipe']}/export-200"
    assert meta['checkpoint']['pose'] and meta['checkpoint']['decp_bb'] and meta['checkpoint']['decp_bb_alpha'] == 1
    want = dict(text_temp=1, cb0_temp=1, light_temp=1, pose_temp=1, pose_light_temp=1, text_topk=0,
                cb0_topk=30, light_topk=30, pose_topk=30, pose_light_topk=50, pose_cfg_on=1,
                pose_cfg_s=4, pose_cfg_cb0=1, pose_cfg_light=1, pose_cfg_speaking_only=0)
    assert all(meta['sampling'].get(k) == v for k, v in want.items()), 'Sampling/guidance settings differ'
    guidance = meta['audio_guidance']
    assert guidance['enabled'] and guidance['strength'] == 3 and guidance['engine_scale'] == 4
    assert guidance['scope'] == 'all pose codebooks' and guidance['full_model_silence_pass'] and guidance['shared_text_and_pose']
    silence = np.load(path / 'guidance_silence_mimi.npy', allow_pickle=False)
    assert silence.shape == (32, guidance['silence_frames']) and silence.min() >= 0 and silence.max() < 2048
    assert hashlib.sha256(np.ascontiguousarray(silence, dtype=np.int64).tobytes()).hexdigest() == guidance['silence_codes_sha256']
    audit = guidance['audit']
    assert audit['enabled'] and audit['independent_cache_rows'] and audit['d33_uses_separate_backbone_hidden']
    assert audit['conditional_weight'] == 4 and audit['silence_weight'] == -3
    assert audit['codebooks'] == list(range(32)) and audit['speaking_only'] is False
    assert audit['guided_frames'] == 752-count and audit['conditional_row'] != audit['silence_row']
    assert audit['silence_codes_sha256'] == guidance['silence_codes_sha256']
    if count:
        assert audit['prefix_audio_frames_replaced_by_channel'] == [0, 100+max(0, count-2)]
        assert audit['forced_audio_frames_replaced_by_channel'] == [0, min(2, count)]
        assert audit['total_supplied_audio_frames_replaced_by_channel'] == [0, 100+count]
        assert audit['forced_pose_frames'] == min(2, count)
    voice = np.load(clip['voice_clip'], allow_pickle=False)
    voice_hash = hashlib.sha256(np.ascontiguousarray(voice, dtype=np.int64).tobytes()).hexdigest()
    assert voice.shape == (32, 100) and voice_hash == meta['voice_codes_sha256'] == clip['voice_provenance']['codes_sha256']
    assert np.array_equal(voice, np.load(path / 'voice_clip_mimi.npy', allow_pickle=False))
    with np.load(path / 'tokens.npz', allow_pickle=False) as tokens:
        lane = tokens['lane']
        assert lane.shape == (752,) and np.issubdtype(lane.dtype, np.integer)
        assert np.all((lane >= 0) & ((lane < 151643) | np.isin(lane, [151645, 151670, 151671])))
        if forced:
            assert np.array_equal(lane, np.load(clip['lane'], allow_pickle=False))
        elif not count:
            assert lane[0] == 151670
        for key, source in source_prefill(clip, count).items():
            actual = lane[:count] if key == 'text_lane' else tokens[key][:, :count]
            assert np.array_equal(actual, source), f'Output {key} changed source prefill'
        assert hashlib.sha256(np.ascontiguousarray(lane, dtype=np.int64).tobytes()).hexdigest() == meta['generated_lane_sha256']
        for name, filename in [('audio_codes', 'mimi.safetensors'), ('pose_codes', 'pose_tokens.safetensors')]:
            codes = tokens[name]
            assert codes.shape == (32, 752) and np.issubdtype(codes.dtype, np.integer) and codes.min() >= 0 and codes.max() < 2048
            assert np.array_equal(codes, load_file(str(path / filename))['codes'])
    request = dict(recipe=job['recipe'], text_mode=job['text_mode'], duration_seconds=60, prefill_frames=count)
    inputs = validate_inputs(path, request, clip, meta)
    audio, sr = sf.read(path / 'audio.wav')
    assert sr == 24000 and len(audio) == 752 * 1920 and np.isfinite(audio).all()
    receipt = dict(status='complete', completed_at=video.stamp(), exit_code=0, **{**expected, **inputs},
                   voice_codes_sha256=voice_hash, sampling=meta['sampling'], audio_guidance=guidance,
                   generated_lane_sha256=meta['generated_lane_sha256'], checkpoint=meta['checkpoint'],
                   generated_transcript=video.read(path / 'words.json')['text'])
    video.write(path / 'PPM_COMPLETE.json', receipt)
    return receipt


def update(job, stage, message=''):
    from publish_guided import publish_guided_state
    publish_guided_state(job, stage, message)


def ppm_worker(team_file):
    team = video.read(team_file)
    root, name = Path(team['root']), team_name(team)
    assert socket.gethostname().split('.')[0] == team['node'] and team['node'] in NODES
    locks = Path('/home/param/.cache/heldout-comparison-gpu-locks')
    locks.mkdir(parents=True, exist_ok=True)
    handles = []
    code = 1
    try:
        for gpu in team['devs']:
            handle = (locks / f'gpu-{gpu}.lock').open('a')
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            handles.append(handle)
            procs = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
            if procs:
                raise RuntimeError(f'GPU{gpu} occupied; no PPM launch')
        if (root / 'STOP').exists():
            raise RuntimeError('STOP exists; no PPM launch')
        video.write(root / 'teams' / f'{name}.ppm-process.json', pid_record(os.getpid()))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, team['devs'])), OMP_NUM_THREADS='8', MKL_NUM_THREADS='8',
                   HF_HUB_OFFLINE='1', MIMI_DIR='/home/param/models/mimi', PYTHONUNBUFFERED='1',
                   FA3_DIR=str(REPO / '.cache/kernels/vllm-flash-attn3'), PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
                   PPM_FA3_ROOT=str(REPO / '.cache/kernels/vllm-flash-attn3'), CUDA_HOME='/usr/local/cuda',
                   CPATH=f'{REPO}/.venv/include/python3.12:{REPO}/.venv/include',
                   TRITON_CACHE_DIR=str(REPO / '.cache/triton'))
        command = [str(REPO / '.venv/bin/python'), '-u', str(REPO / 'infer/transcript_pose/infer_batch.py'),
                   '--recipe', team['recipe'], '--batch-json', team['batch_json'], '--devs', '0,1',
                   '--text-mode', 'forced' if team['recipe'] == 'aligned-media' else 'native',
                   '--prefill-frames', str(team.get('prefill_frames', 0))]
        for key, value in SAMPLE.items():
            command += ['--' + key.replace('_', '-'), str(value)]
        video.write(root / 'teams' / f'{name}.command.json', dict(command=command, gpus=team['devs']))
        code = subprocess.call(command, cwd=REPO, env=env)
    except Exception:
        traceback.print_exc()
    finally:
        for handle in reversed(handles):
            handle.close()
        video.write(root / 'teams' / f'{name}.ppm-exit.json', dict(exit_code=code, status='complete' if code == 0 else 'failed',
                    node=team['node'], completed_at=video.stamp()))
    return code


def start_ppm(team_file):
    team = video.read(team_file)
    root, name = Path(team['root']), team_name(team)
    receipt = root / 'teams' / f'{name}.ppm-exit.json'
    process_file = root / 'teams' / f'{name}.launcher.json'
    if receipt.exists():
        return
    if process_file.exists() and alive(video.read(process_file)):
        return
    with (root / 'logs' / f'{name}.ppm.log').open('a') as log:
        process = subprocess.Popen([str(REPO / '.venv/bin/python'), str(Path(__file__).resolve()),
                                    'ppm-worker', '--team-file', str(team_file)],
                                   stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                   start_new_session=True, cwd=REPO)
    video.write(process_file, pid_record(process.pid))


def run_video(job):
    from publish_guided import publish_guided
    directory, node = Path(job['directory']), job['node']
    config = directory / 'video_job.json'
    video.write(config, job)
    video.sync_to(node, [config, directory / 'ppm/PPM_COMPLETE.json'])
    if not (directory / 'VIDEO_COMPLETE.json').exists():
        for stage, label in [('pose', 'rendering-pose'), ('video', 'rendering-video')]:
            if (Path(job['root']) / 'STOP').exists():
                raise RuntimeError('STOP exists; no video launch')
            update(job, label, 'Rendering generated conditioning pose' if stage == 'pose' else 'Generating video')
            video.remote(node, [video.VIDEO_PY, REPO / 'infer/transcript_pose/comparison_video.py',
                                'worker', '--job', config, '--stage', stage], log=directory / f'remote-{stage}.log')
        video.execute(['rsync', '-a', '-e', shlex.join(video.SSH), '--exclude=ppm', f'{node}:{directory}/', str(directory) + '/'])
    # Publication creates/validates pose views and verifies the video receipt.
    publish_guided(job)


def run_team(root, team):
    name = team_name(team)
    jobs = team['jobs']
    try:
        if not all((Path(job['ppm']) / 'PPM_COMPLETE.json').exists() for job in jobs):
            if (root / 'STOP').exists():
                raise RuntimeError('STOP exists; no PPM launch')
            for job in jobs:
                update(job, 'generating-ppm', 'Generating audio and pose with full-model silence guidance')
            team_file = root / 'teams' / f'{name}.json'
            video.remote(team['node'], [REPO / '.venv/bin/python', Path(__file__).resolve(), 'start-ppm', '--team-file', team_file])
            receipt = root / 'teams' / f'{name}.ppm-exit.json'
            probe = "import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); print(p.read_text() if p.exists() else '{}')"
            began = time.monotonic()
            while True:
                result = video.remote(team['node'], ['python3', '-c', probe, receipt], capture_output=True, text=True)
                value = json.loads(result.stdout)
                if value:
                    video.write(receipt, value)
                    break
                if time.monotonic() - began > 7200:
                    raise TimeoutError('PPM team has not completed in two hours; inspect its detached worker')
                time.sleep(10)
            if value['exit_code'] != 0:
                raise RuntimeError(f"PPM exited{value['exit_code']}; see remote logs/{name}.ppm.log")
            for job in jobs:
                dest = Path(job['ppm'])
                dest.mkdir(parents=True, exist_ok=True)
                video.execute(['rsync', '-a', '-e', shlex.join(video.SSH), f"{team['node']}:{dest}/", str(dest) + '/'])
        for job in jobs:
            check_ppm(job)
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            for future in as_completed([pool.submit(run_video, job) for job in jobs]):
                future.result()
        result = dict(team=name, status='complete', clips=[j['clip_id'] for j in jobs], completed_at=video.stamp())
    except Exception as exc:
        (root / 'logs' / f'{name}.error.txt').write_text(traceback.format_exc())
        for job in jobs:
            try:
                if not video.complete(Path(job['directory']) / 'video_state.json'):
                    update(job, 'failed', str(exc))
            except Exception:
                traceback.print_exc()
        result = dict(team=name, status='failed', error=str(exc), completed_at=video.stamp())
    finally:
        subprocess.run(['rsync', '-a', '-e', shlex.join(video.SSH), f"{team['node']}:{root}/logs/{name}.ppm.log", str(root / 'logs') + '/'], check=False)
    video.write(root / 'teams' / f'{name}.status.json', result)
    print(json.dumps(result), flush=True)
    return result


def run(root):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    (root / 'logs').mkdir(parents=True, exist_ok=True)
    with (root / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        video.write(root / 'controller-process.json', pid_record(os.getpid()))
        assert video.read(root / 'source-sync.json')['status'] == 'complete'
        # Resolve Transformers lazy import once, before concurrent validation.
        from request_worker import text_tokenizer, prefill_source_data
        text_tokenizer()
        if prefill_frames(root):
            prefill_source_data()
        planned = video.read(root / 'execution-plan.json')
        with ThreadPoolExecutor(max_workers=16) as pool:
            results = [f.result() for f in as_completed([pool.submit(run_team, root, t) for t in planned])]
        passed = all(r['status'] == 'complete' for r in results)
        video.write(root / 'GUIDED_COMPARISON_COMPLETE.json', dict(status='complete' if passed else 'failed',
                    completed_at=video.stamp(), teams=results, samples=24))
        return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('command', choices=('prepare', 'plan', 'sync', 'run', 'launch', 'status', 'ppm-worker', 'start-ppm'))
    parser.add_argument('--root', type=Path, default=PREFILL_ROOT)
    parser.add_argument('--team-file', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command in ('ppm-worker', 'start-ppm'):
        if args.team_file is None:
            parser.error('--team-file required')
        return ppm_worker(args.team_file) if args.command == 'ppm-worker' else start_ppm(args.team_file)
    prefill_frames(root)
    if args.command == 'prepare':
        prepare(root)
    elif args.command == 'plan':
        print(json.dumps([dict(node=t['node'], recipe=t['recipe'], devs=t['devs'], clips=[c['id'] for c in t['clips']]) for t in teams(root)], indent=2))
    elif args.command == 'sync':
        sync_sources(root)
    elif args.command == 'run':
        return run(root)
    elif args.command == 'launch':
        root.mkdir(parents=True, exist_ok=True)
        (root / 'logs').mkdir(parents=True, exist_ok=True)
        process_path = root / 'controller-launch.json'
        if process_path.exists() and alive(video.read(process_path)):
            raise RuntimeError('Controller already running')
        if (root / 'STOP').exists():
            raise RuntimeError('STOP exists; no launch')
        smoke_name = 'pose-audio-guidance-prefill-smoke-20261029' if prefill_frames(root) else 'pose-audio-guidance-smoke-20261029'
        smoke_path = Path('/home/param/pose-runs') / smoke_name / 'validation.json'
        smoke = video.read(smoke_path)
        if smoke['status'] != 'passed':
            raise RuntimeError('Required GPU smoke has not passed')
        if prefill_frames(root):
            assert smoke['prefill_frames'] == 25 and len(smoke['rows']) == 2
            prepared = video.read(root / 'preparation-validation.json')
            assert prepared['status'] == 'passed' and len(prepared['rows']) == 24
        with (root / 'controller.log').open('a') as log:
            proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'run', '--root', str(root)],
                                    stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    start_new_session=True, cwd=REPO, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
        record = dict(**pid_record(proc.pid), started_at=video.stamp(),
                      root=str(root), teams=16, samples=24, sampling=SAMPLE,
                      prefill_frames=prefill_frames(root),
                      source_sync=str(root / 'source-sync.json'),
                      smoke_validation=dict(path=str(smoke_path), sha256=digest(smoke_path),
                                            status=smoke['status'], rows=len(smoke['rows']),
                                            frames_per_row=[row['frames'] for row in smoke['rows']],
                                            peak_gib=smoke['rows'][0]['peak_gib']))
        video.write(process_path, record)
        print(json.dumps(record))
    elif args.command == 'status':
        results = [video.read(p) for p in sorted((root / 'teams').glob('*.status.json'))]
        print(json.dumps(dict(controller=video.read(root / 'controller-launch.json') if (root / 'controller-launch.json').exists() else None,
                              complete=sum(x['status'] == 'complete' for x in results),
                              failed=sum(x['status'] == 'failed' for x in results), teams=results), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
