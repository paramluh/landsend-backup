#!/usr/bin/env python3
"""Persistent CPU scheduler for website requests on landsend-1 through4.

Executors are detached through SSH, including on node1. Restarting this daemon
reattaches to durable assignments; it never terminates or requeues active work.
Completed remote artifacts are copied privately, validated, then published.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time
import traceback
import uuid

import request_worker as W
from request_schema import RECIPES, media_frames

NODES = tuple(f'landsend-{i}' for i in range(1, 5))
EXECUTIONS = Path('/home/param/pose-runs/inference-request-executions-20261028')
SSH = ['ssh', '-i', '/home/param/.ssh/id_ed25519_cluster', '-o', 'BatchMode=yes',
       '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=2']
SCRIPT = Path(__file__).resolve()
STOPPING = False


def host():
    return socket.gethostname().split('.')[0]


def require_host(expected=None):
    W.require(os.getuid() == 1003 and host() in NODES, 'Dispatcher/executor controls require param UID1003 on nodes1–4')
    if expected is not None:
        W.require(host() == expected, 'Wrong execution node')


def ticks(pid):
    try:
        return int(Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')', 1)[1].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def alive(record):
    return bool(record and record.get('start_ticks') is not None and ticks(record.get('pid', 0)) == record['start_ticks'])


def optional(path):
    try:
        return W.read(path)
    except (OSError, ValueError):
        return None


def execution_job(path):
    path = Path(path)
    W.require(uuid.UUID(path.name).hex == path.name and not path.is_symlink(), 'Invalid execution job ID')
    W.require(path.resolve().parent == (EXECUTIONS / 'jobs').resolve(), 'Unexpected execution job root')
    return path


def probe(job):
    if not job.exists():
        return {'exists': False, 'active': False}
    executor, launch = optional(job / 'executor.json'), optional(job / 'launch.json')
    with (job / '.executor.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            busy = False
        except BlockingIOError:
            busy = True
    return dict(exists=True, executor=executor, launch=launch, state=optional(job / 'state.json'),
                active=busy or alive(executor) or alive(launch))


def remote_start(job, node, gpus, claim):
    require_host(node)
    assignment = W.read(job / 'assignment.json')
    W.require(assignment['claim_token'] == claim and assignment['node'] == node and assignment['gpus'] == gpus,
              'Execution assignment mismatch')
    with (job / '.launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        current = probe(job)
        ex = current.get('executor') or {}
        if current['active'] or ex.get('status') in ('complete', 'failed'):
            return current
        command = [str(W.PPM_PY), '-u', str(W.REPO / 'infer/transcript_pose/request_worker.py'),
                   'execute-job', '--node', node, '--gpu-pair', ','.join(map(str, gpus)), '--job-dir', str(job)]
        with (job / 'executor.log').open('ab') as log:
            proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True, close_fds=True, cwd=W.REPO,
                                    env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2'))
        launch = dict(pid=proc.pid, start_ticks=ticks(proc.pid), claim_token=claim, node=node,
                      gpus=gpus, launched_at=W.stamp())
        W.write(job / 'launch.json', launch)
        return dict(exists=True, active=True, launch=launch, executor=ex)


def capabilities():
    require_host()
    lines = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                                     '--format=csv,noheader,nounits'], text=True, timeout=15).splitlines()
    idle = [int(row.split(',')[0]) for row in lines if int(row.split(',')[1]) < 1024]
    return dict(node=host(), recipes=[r for r in RECIPES if (W.MODELS / 'transcript-pose' / r / 'export-200/COMPLETE').is_file()],
                idle_gpus=idle, checked_at=time.time())


def remote(node, *args, timeout=45):
    command = [str(W.PPM_PY), str(SCRIPT), *map(str, args)]
    result = subprocess.run([*SSH, node, shlex.join(command)], text=True, capture_output=True, timeout=timeout, check=True)
    return json.loads(result.stdout)


def rsync(node, source, dest, *, pull=False, extra=()):
    argv = ['rsync', '-aL', '--partial', '-e', shlex.join(SSH), *extra]
    argv += [f'{node}:{source}', str(dest)] if pull else [str(source), f'{node}:{dest}']
    subprocess.run(argv, check=True, timeout=600, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def mark(job, phase, assignment, **fields):
    fields.setdefault('error', None)
    W.state(job, 'running', phase, assigned_node=assignment['node'], gpu_pair=assignment['gpus'],
            claim_token=assignment['claim_token'], **fields)


def stage(job, assignment):
    node, target = assignment['node'], assignment['remote_job']
    # A missing capability/asset is reported before assigning new work; actual
    # source validation remains in the executor immediately before GPU use.
    subprocess.run([*SSH, node, shlex.join(['mkdir', '-p', target])], check=True, timeout=30)
    for name in ('request.json', 'assignment.json'):
        rsync(node, job / name, target + '/' + name)
    # Initial state is staged once. Subsequent polls never overwrite executor state.
    if not assignment.get('staged'):
        initial = job / '.initial-executor-state.json'
        W.write(initial, dict(id=job.name, status='queued', stage='assigned', request=W.request(job),
                              created_at=(optional(job / 'state.json') or {}).get('created_at', W.stamp()),
                              claim_token=assignment['claim_token'], assigned_node=node, gpu_pair=assignment['gpus']))
        rsync(node, initial, target + '/state.json')
        initial.unlink()
        assignment['staged'] = True
        W.write(job / 'assignment.json', assignment)


def sync_progress(job, assignment):
    # Small progress/log snapshots are private; terminal state is never copied
    # into the API's central state.json before the full-result validation.
    dest = job / 'progress'
    dest.mkdir(exist_ok=True)
    rsync(assignment['node'], assignment['remote_job'] + '/', str(dest) + '/', pull=True,
          extra=('--include=*/', '--include=*.log', '--include=state.json', '--include=executor.json',
                 '--include=meta.json', '--include=words.json', '--include=status.json', '--exclude=*', '--prune-empty-dirs'))


def validate_result(result, job, assignment):
    req = W.request(job)
    clip, _ = W.source_clip(req['clip_id'])
    W.validate_ppm(result, req, clip)
    metadata = W.read(result / 'metadata.json')
    W.require(metadata['request'] == req and metadata['node'] == assignment['node'] and metadata['gpus'] == assignment['gpus'],
              'Returned metadata does not match the assigned request')
    W.require(W.sha(result / 'video.mp4') == metadata['sha256'], 'Returned video checksum mismatch')
    if metadata.get('pose_views'):
        from pose_views import validate_pose_views
        validate_pose_views(result, video=result / 'video.mp4')
        for stem, filename in [('pose', 'pose.mp4'), ('pose_overlay', 'overlay.mp4')]:
            W.require(W.sha(result / filename) == metadata[stem + '_sha256'], 'Returned pose view checksum mismatch')
    receipt = W.read(result / 'diffusion/generation/status.json')
    W.validate_video_receipt(receipt, media_frames(req['duration_seconds']), req['video_seed'])
    remote_root = Path(assignment['remote_job'])
    for item in receipt['latent_files']:
        relative = Path(item['path']).relative_to(remote_root)
        W.require(W.sha(result / relative) == item['sha256'], 'Returned latent checksum mismatch')
        W.validate_latents(result / relative)
    streams = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json',
                                                 str(result / 'video.mp4')], timeout=60))['streams']
    v = next(s for s in streams if s['codec_type'] == 'video')
    a = next(s for s in streams if s['codec_type'] == 'audio')
    seconds = req['duration_seconds']
    W.require(int(v['nb_frames']) == seconds * 25 and abs(float(v['duration']) - seconds) < .001
              and abs(float(a['duration']) - seconds) < .05 and int(a['sample_rate']) == 24000,
              'Returned video duration/frame count mismatch')
    return metadata


def collect(job, assignment):
    mark(job, 'transferring', assignment, active_gpus=[])
    result = job / 'result'
    if not result.exists():
        incoming = job / '.incoming-result'
        incoming.mkdir(exist_ok=True)
        rsync(assignment['node'], assignment['remote_job'] + '/', str(incoming) + '/', pull=True,
              extra=('--exclude=state.json', '--exclude=assignment.json', '--exclude=request.json',
                     '--exclude=executor.json', '--exclude=launch.json', '--exclude=.*lock'))
        mark(job, 'validating', assignment, active_gpus=[])
        validate_result(incoming, job, assignment)
        incoming.replace(result)
    else:
        validate_result(result, job, assignment)
    # Atomic regular-file links retain existing API routes and do not expose a
    # partially copied video. All auxiliary artifacts remain under result/.
    metadata = W.read(result / 'metadata.json')
    names = ['video.mp4', 'metadata.json']
    views = {}
    if metadata.get('pose_views'):
        names += ['pose.mp4', 'overlay.mp4', 'pose-views.json']
        views = {k: metadata[k] for k in ('pose_video', 'pose_sha256', 'pose_overlay_video', 'pose_overlay_sha256')}
    for name in names:
        temporary = job / ('.publish-' + name)
        temporary.unlink(missing_ok=True)
        os.link(result / name, temporary)
        temporary.replace(job / name)
    W.write(job / 'dispatch-result.json', dict(status='complete', node=assignment['node'], gpu_pair=assignment['gpus'],
                                             claim_token=assignment['claim_token'], completed_at=W.stamp(),
                                             sha256=W.sha(result / 'video.mp4')))
    W.state(job, 'complete', 'complete', completed_at=W.stamp(), assigned_node=assignment['node'],
            gpu_pair=assignment['gpus'], active_gpus=[], video='video.mp4', metadata='metadata.json', **views)
    assignment['terminal'] = True
    assignment['active_gpus'] = []
    W.write(job / 'assignment.json', assignment)


def monitor(job):
    assignment = W.read(job / 'assignment.json')
    try:
        p = remote(assignment['node'], 'probe', '--job-dir', assignment['remote_job'])
        ex = p.get('executor') or {}
        if ex and ex.get('claim_token') != assignment['claim_token']:
            raise ValueError('Executor claim token mismatch')
        if ex.get('status') == 'complete' and ex.get('exit_code') == 0:
            assignment['active_gpus'] = []
            W.write(job / 'assignment.json', assignment)
            collect(job, assignment)
            return
        if ex.get('status') == 'failed':
            raise RuntimeError((p.get('state') or {}).get('error', 'Remote executor failed'))
        if p.get('active'):
            active = assignment['gpus'] if ex.get('status') == 'busy' else ex.get('active_gpus', assignment['gpus'])
            assignment['active_gpus'] = active
            assignment['last_seen'] = W.stamp()
            W.write(job / 'assignment.json', assignment)
            remote_state = p.get('state') or {}
            mark(job, remote_state.get('stage', 'launching'), assignment, active_gpus=active,
                 executor_pid=ex.get('pid', (p.get('launch') or {}).get('pid')))
            sync_progress(job, assignment)
            return
        if ex.get('status') == 'running' or (p.get('launch') and not ex):
            raise RuntimeError('Executor stopped without a terminal receipt; partial results retained, job will not be rerun')
        # No live process: a new staged assignment or a busy75 attempt is safe to
        # launch. The remote per-job launch/executor locks prevent duplicates.
        if not p.get('exists') or not assignment.get('staged'):
            mark(job, 'staging', assignment, active_gpus=assignment['gpus'])
            stage(job, assignment)
        assignment['active_gpus'] = assignment['gpus']
        W.write(job / 'assignment.json', assignment)
        mark(job, 'launching', assignment, active_gpus=assignment['gpus'])
        remote(assignment['node'], 'start', '--job-dir', assignment['remote_job'], '--node', assignment['node'],
               '--gpu-pair', ','.join(map(str, assignment['gpus'])), '--claim-token', assignment['claim_token'])
    except (subprocess.SubprocessError, OSError) as exc:
        # Network failure is not evidence that the GPU process stopped.
        mark(job, 'unreachable', assignment, error=f'Temporarily unable to inspect assigned node: {exc}',
             active_gpus=assignment.get('active_gpus', assignment['gpus']))
    except Exception as exc:
        W.state(job, 'failed', 'failed', error=str(exc), completed_at=W.stamp(), assigned_node=assignment['node'],
                gpu_pair=assignment['gpus'], active_gpus=[])
        assignment['terminal'] = True
        assignment['active_gpus'] = []
        W.write(job / 'assignment.json', assignment)
        (job / 'dispatcher-error.log').write_text(traceback.format_exc())


def claim(job, node, pair):
    with (job / '.claim.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if W.job_status(job) != 'queued' or (job / 'assignment.json').exists():
            return False
        assignment = dict(claim_token=uuid.uuid4().hex, node=node, gpus=pair, active_gpus=pair,
                          remote_job=str(EXECUTIONS / 'jobs' / job.name), claimed_at=W.stamp(), staged=False)
        W.write(job / 'assignment.json', assignment)
        mark(job, 'claiming', assignment, active_gpus=pair)
        return True


def allocations(queue):
    occupied = {node: set() for node in NODES}
    active = []
    for job in W.jobs(queue):
        a = optional(job / 'assignment.json')
        if a and not a.get('terminal'):
            occupied[a['node']].update(a.get('active_gpus', a['gpus']))
            active.append(job)
    return occupied, active


def heartbeat(queue, caps):
    occupied, active = allocations(queue)
    slots = []
    for job in active:
        a = W.read(job / 'assignment.json');s = optional(job / 'state.json') or {}
        slots.append(dict(node=a['node'], gpu_pair=a['gpus'], active_gpus=a.get('active_gpus', a['gpus']),
                          job_id=job.name, stage=s.get('stage', 'assigned')))
    available = 0
    free_total = 0
    for node in NODES:
        fresh = time.time() - caps.get(node, {}).get('checked_at', 0) < 30
        free = sorted(set(caps.get(node, {}).get('idle_gpus', [])) - occupied[node]) if fresh else []
        available += len(free) // 2
        free_total += len(free)
        for i in range(0, len(free) - 1, 2):
            slots.append(dict(node=node, gpu_pair=free[i:i+2], active_gpus=[], job_id=None, stage='free'))
    states = [optional(j / 'state.json') or {} for j in W.jobs(queue)]
    W.write(queue / 'dispatcher-status.json', dict(updated_at=W.stamp(), status='stopping' if STOPPING else 'running',
            capacity_total=16, capacity_busy=16-available, capacity_free=available,
            gpus_total=32, gpus_busy=32-free_total, gpus_free=free_total,
            queued_jobs=sum(s.get('status') == 'queued' for s in states),
            running_jobs=sum(s.get('status') == 'running' for s in states), slots=slots))


def run(queue, poll):
    require_host('landsend-1')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    (queue / 'jobs').mkdir(parents=True, exist_ok=True)
    with (queue / '.dispatcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Resolve lazy imports once before concurrent CPU result validation.
        W.text_tokenizer()
        import torch
        torch.set_num_threads(2)
        caps = {};last_node = -1
        inflight = {}
        with ThreadPoolExecutor(max_workers=64) as pool, ThreadPoolExecutor(max_workers=4) as probes:
            while not STOPPING:
                for job_id, future in list(inflight.items()):
                    if future.done():
                        try:future.result()
                        except Exception:traceback.print_exc()
                        del inflight[job_id]
                futures = {probes.submit(remote, node, 'capabilities'): node for node in NODES}
                for f in as_completed(futures):
                    try:caps[futures[f]] = f.result()
                    except Exception:pass
                occupied, active = allocations(queue)
                if not (queue / 'DISPATCH_STOP').exists():
                    for job in W.jobs(queue):
                        if W.job_status(job) != 'queued' or (job / 'assignment.json').exists():continue
                        try:req = W.request(job)
                        except Exception as exc:
                            W.state(job, 'failed', 'failed', error=str(exc));continue
                        choices = []
                        for index, node in enumerate(NODES):
                            cap = caps.get(node, {})
                            if time.time()-cap.get('checked_at',0)>30 or req['recipe'] not in cap.get('recipes',[]):continue
                            free = sorted(set(cap.get('idle_gpus', []))-occupied[node])
                            if len(free)>=2:choices.append((len(occupied[node]),(index-last_node-1)%4,index,node,free[:2]))
                        if not choices:continue
                        _,_,index,node,pair=min(choices)
                        if claim(job,node,pair):
                            occupied[node].update(pair);active.append(job);last_node=index
                for job in active:
                    if job.name not in inflight:
                        inflight[job.name] = pool.submit(monitor, job)
                heartbeat(queue,caps)
                for _ in range(poll):
                    if STOPPING:break
                    time.sleep(1)
        status=optional(queue/'dispatcher-status.json') or {};status.update(status='stopped',updated_at=W.stamp());W.write(queue/'dispatcher-status.json',status)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('command',choices=('run','capabilities','probe','start'))
    p.add_argument('--queue-root',type=Path,default=W.QUEUE)
    p.add_argument('--poll-seconds',type=int,default=5)
    p.add_argument('--job-dir',type=Path)
    p.add_argument('--node',choices=NODES)
    p.add_argument('--gpu-pair')
    p.add_argument('--claim-token')
    a=p.parse_args()
    if a.command=='run':
        def stop(*_):
            global STOPPING
            STOPPING=True
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        run(a.queue_root.resolve(),a.poll_seconds)
    elif a.command=='capabilities':print(json.dumps(capabilities()))
    else:
        require_host();job=execution_job(a.job_dir)
        if a.command=='probe':print(json.dumps(probe(job)))
        else:
            gpus=list(map(int,a.gpu_pair.split(',')));W.require(len(gpus)==2 and len(set(gpus))==2 and all(0<=g<8 for g in gpus),'InvalidGPU pair')
            print(json.dumps(remote_start(job,a.node,gpus,a.claim_token)))


if __name__=='__main__':main()
