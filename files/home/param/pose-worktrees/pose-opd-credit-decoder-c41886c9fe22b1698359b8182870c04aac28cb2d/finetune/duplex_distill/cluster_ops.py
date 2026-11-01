"""Detached seven-node distillation launch, read-only preflight and status.

This launcher never evicts existing jobs. A failed launch rolls back only its
own recorded process trees. landsend-0 is both the controller and a GPU worker.
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time
import uuid

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
HOSTS = tuple(f"landsend-{i}" for i in (0, 1, 2, 3, 4, 5, 12))
SSH = ["ssh", "-i", "/home/param/.ssh/id_ed25519_cluster", "-o", "BatchMode=yes",
       "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=yes"]
LOCK = Path('/home/param/runs/pose-runs/.duplex-distill-gpus.lock')
FFMPEG = REPO/'.venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'
FFMPEG_SHA256 = 'e7e7fb30477f717e6f55f9180a70386c62677ef8a4d4d1a5d948f4098aa3eb99'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def local_host():
    return socket.gethostname().split('.')[0]


def permitted(hosts, mode, *, require_full=True):
    if mode not in ('smoke', 'train'):
        raise ValueError(f'Unknown launch mode: {mode}')
    if not hosts or len(set(hosts)) != len(hosts) or not set(hosts) <= set(HOSTS):
        raise ValueError(f'GPU hosts must be distinct members of {HOSTS}')
    if require_full and mode == 'train' and tuple(hosts) != HOSTS:
        raise ValueError('Training requires all seven configured nodes in their configured order')


def process_identity(pid):
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'start_ticks': fields[19], 'state': fields[0]}


def live(record):
    try:
        actual = process_identity(record['pid'])
        return actual['start_ticks'] == record['start_ticks'] and actual['state'] != 'Z'
    except (FileNotFoundError, ProcessLookupError):
        return False


def remote(host, *args, timeout=120, input_text=None):
    if host not in HOSTS:
        raise ValueError(f'Forbidden remote GPU host: {host}')
    return subprocess.run(SSH + [host, shlex.join(map(str, args))], text=True,
                          input=input_text, capture_output=True, timeout=timeout, check=True)


def query_gpu():
    if local_host() not in HOSTS:
        raise RuntimeError('GPU inspection/worker operations are restricted to training hosts')
    rows = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,utilization.gpu',
         '--format=csv,noheader,nounits'], text=True).splitlines()
    pids = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
        text=True).strip().splitlines()
    return rows, pids


def preflight():
    host = local_host()
    rows, pids = query_gpu()
    assert len(rows) == 8 and all('H100' in row for row in rows), rows
    assert not pids, f'Existing GPU jobs: {pids}. Stop these separately before launch.'
    assert all(int(row.split(',')[3].strip()) == 0 for row in rows), f'GPU memory in use: {rows}'
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    paths = [REPO/'.venv/bin/torchrun', REPO/'.venv/bin/python',
             REPO/'.cache/kernels/vllm-flash-attn3/build/torch28-cxx11-cu128-x86_64-linux/__init__.py',
             REPO/'finetune/transcript_pose/environment/activate.sh',
             Path('/home/param/models/imported-weights/mimi/model.safetensors'),
             Path('/home/param/models/imported-weights/pose-tokenizer-32cb-200k-noshift/model.safetensors'),
             Path('/home/param/models/imported-weights/ppm-811-17850-base-bf16/configs/Qwen3-32B/tokenizer.json'),
             Path('/home/param/models/imported-weights/aloe-ft-vox17127-nopersona/assets/voxtral-realtime/audio_tower.safetensors'),
             FFMPEG]
    missing = [str(p) for p in paths if not p.is_file()]
    assert not missing, f'Missing environment/dependencies: {missing}'
    assert hashlib.sha256(FFMPEG.read_bytes()).hexdigest() == FFMPEG_SHA256, 'Pinned FFmpeg binary differs'
    models = {
        'student': Path('/home/param/models/imported-weights/ppm-aloe-poseduplex-v5-final/export-498-dcp'),
        'pose_teacher': Path('/home/param/models/imported-weights/transcript-pose/xabi-original/export-200'),
        'audio_teacher': Path('/home/param/models/imported-weights/ppm-aloe-samsep-vox17127-final/export-249-dcp'),
    }
    model_files = {}
    for name, root in models.items():
        assert (root/'COMPLETE').is_file(), f'Missing model completion marker: {root}'
        shards = list(root.glob('rank*.pt')) if name == 'pose_teacher' else list(root.glob('*.distcp'))
        assert len(shards) == 64 and all(p.stat().st_size > 0 for p in shards), f'Incomplete model: {root}'
        if name == 'pose_teacher':
            assert (root/'rep.pt').is_file() and (root/'meta.json').is_file()
        else:
            assert (root/'.metadata').is_file() and (root/'train_cfg.json').is_file()
        model_files[name] = {'path': str(root), 'shards': len(shards)}
    ready_path = Path('/home/param/datasets/aloe-duplex/DATASET_READY.json')
    ready = json.loads(ready_path.read_text())
    assert ready.get('training_ready'), f'Dataset is not ready: {ready_path}'
    addresses = json.loads(subprocess.check_output(['ip', '-j', '-6', 'address', 'show', 'dev', 'ibs14'], text=True))
    ipv6 = [a['local'] for interface in addresses for a in interface['addr_info']
            if a['family'] == 'inet6' and a['scope'] == 'global']
    assert ipv6, 'No global IPv6 address on ibs14'
    hcas = sorted(p.name for p in Path('/sys/class/infiniband').iterdir())
    assert hcas, 'No InfiniBand HCA found'
    packages = subprocess.check_output([str(REPO/'.venv/bin/python'), '-c',
        'import json,torch,transformers,wandb;print(json.dumps(dict(torch=torch.__version__,transformers=transformers.__version__,wandb=wandb.__version__)))'], text=True)
    return {'host': host, 'passed': True, 'checked_utc': now(), 'gpus': rows,
            'ib_address': ipv6[0], 'hcas': hcas, 'packages': json.loads(packages),
            'models': model_files, 'ffmpeg': str(FFMPEG)}


def worker(spec_path):
    spec_path = Path(spec_path)
    spec = json.loads(spec_path.read_text())
    permitted(spec['hosts'], spec['mode'])
    host = local_host()
    assert host in spec['hosts']
    rank = spec['hosts'].index(host)
    job = spec_path.parent
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(job/'supervisor.json', process_identity(os.getpid()))
        write_json(job/'started.json', {'started_utc': now(), 'host': host, 'rank': rank})
        env = os.environ.copy()
        env.update({'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
                    'NCCL_SOCKET_IFNAME': '=ibs14', 'NCCL_SOCKET_FAMILY': 'AF_INET6',
                    'GLOO_SOCKET_IFNAME': 'ibs14', 'NCCL_IB_DISABLE': '0', 'NCCL_NET': 'IB',
                    'TORCH_NCCL_ASYNC_ERROR_HANDLING': '1', 'OMP_NUM_THREADS': '1',
                    'NCCL_DEBUG': 'WARN', 'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7',
                    'PYTHONUNBUFFERED': '1', 'WANDB_BASE_URL': 'https://api.forge.coreweave.com',
                    'WANDB_ENTITY': 'canopy-labs', 'S4_FFMPEG': str(FFMPEG)})
        command = [str(REPO/'.venv/bin/torchrun'), f'--nnodes={len(spec["hosts"])}',
                   '--nproc_per_node=8', f'--node_rank={rank}',
                   f'--master_addr={spec["master_addr"]}', f'--master_port={spec["port"]}',
                   '--max_restarts=0', str(HERE/'train.py'), '--run-dir', spec['run_dir'],
                   '--mode', spec['mode'], *spec['trainer_args']]
        activation = REPO/'finetune/transcript_pose/environment/activate.sh'
        child = subprocess.Popen(['bash', '-c', f'source {shlex.quote(str(activation))}; exec "$@"',
                                  'duplex-distill', *command], env=env, cwd=REPO,
                                 start_new_session=True)
        write_json(job/'torchrun.json', process_identity(child.pid))
        def terminate(signum, _frame):
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        signal.signal(signal.SIGTERM, terminate)
        signal.signal(signal.SIGINT, terminate)
        rc = child.wait()
        write_json(job/'finished.json', {'finished_utc': now(), 'exit_code': rc})
        raise SystemExit(rc if rc >= 0 else 128-rc)


def start(encoded):
    spec = json.loads(base64.b64decode(encoded))
    permitted(spec['hosts'], spec['mode'])
    assert local_host() in spec['hosts']
    run = Path(spec['run_dir'])
    assert run.is_absolute() and str(run) != '/'
    job = run/'cluster'/spec['launch_id']
    if job.exists():
        # The controller also trains: launch() has already written this exact
        # launch directory locally. Never attach a second worker to a run.
        assert local_host() == 'landsend-0', f'Existing worker launch directory: {job}'
        assert json.loads((job/'launch.json').read_text()) == spec
        assert not any((job/name).exists() for name in ('supervisor.json', 'torchrun.json', 'started.json'))
    else:
        job.mkdir(parents=True, exist_ok=False)
    write_json(job/'launch.json', spec)
    latest = run/'cluster/latest'
    tmp = latest.with_name('latest.' + spec['launch_id'])
    tmp.symlink_to(spec['launch_id'])
    tmp.replace(latest)
    with (job/'node.log').open('ab') as log:
        child = subprocess.Popen([sys.executable, str(__file__), '_worker', str(job/'launch.json')],
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True, close_fds=True)
    time.sleep(.5)
    if child.poll() is not None:
        raise RuntimeError((job/'node.log').read_text()[-4000:])
    return {'host': local_host(), 'supervisor_pid': child.pid, 'log': str(job/'node.log')}


def stop(run_dir, launch_id):
    assert local_host() in HOSTS
    job = Path(run_dir)/'cluster'/launch_id
    signaled = {}
    for name in ('torchrun', 'supervisor'):
        path = job/f'{name}.json'
        if path.exists():
            record = json.loads(path.read_text())
            if live(record):
                if name == 'torchrun' and os.getpgid(record['pid']) == record['pid']:
                    os.killpg(record['pid'], signal.SIGTERM)
                else:
                    os.kill(record['pid'], signal.SIGTERM)
                signaled[name] = record['pid']
    return {'host': local_host(), 'signaled': signaled}


def node_status(run_dir, tail):
    assert local_host() in HOSTS
    job = Path(run_dir)/'cluster/latest'
    result = {'host': local_host(), 'launch': job.resolve().name}
    for name in ('supervisor', 'torchrun'):
        path = job/f'{name}.json'
        record = json.loads(path.read_text()) if path.exists() else None
        result[name] = dict(record, live=live(record)) if record else None
    result['finished'] = json.loads((job/'finished.json').read_text()) if (job/'finished.json').exists() else None
    rows, pids = query_gpu()
    result.update(gpus=rows, gpu_pids=pids)
    log = job/'node.log'
    if tail and log.exists():
        with log.open('rb') as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell()-32768))
            result['tail'] = stream.read().decode(errors='replace').splitlines()[-tail:]
    return result


def sync(host):
    if host == local_host():
        return host
    for source in (HERE, REPO/'finetune/storage_paths.py', REPO/'finetune/transcript_pose/environment'):
        directory = source.is_dir()
        remote(host, 'mkdir', '-p', source if directory else source.parent)
        suffix = '/' if directory else ''
        subprocess.run(['rsync', '-a', '--no-owner', '--no-group', '--exclude=__pycache__/',
                        '--exclude=.cache/', '--exclude=validation/', '--exclude=benchmarks/', '--exclude=*.pyc',
                        '-e', shlex.join(SSH), str(source)+suffix, f'{host}:{source}{suffix}'],
                       capture_output=True, text=True, check=True, timeout=180)
    return host


def parallel(hosts, function):
    values, errors = {}, {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(hosts)) as pool:
        tasks = {pool.submit(function, host): host for host in hosts}
        for task in concurrent.futures.as_completed(tasks):
            host = tasks[task]
            try:
                values[host] = task.result()
            except Exception as exc:
                errors[host] = str(exc) + ('\n'+exc.stderr if getattr(exc, 'stderr', None) else '')
    if errors:
        raise RuntimeError(json.dumps(errors, indent=2))
    return values


def launch(args, trainer_args):
    assert local_host() == 'landsend-0', 'Use landsend-0 as the launch controller'
    hosts = tuple(args.hosts.split(',')) if args.hosts else HOSTS
    permitted(hosts, args.mode)
    run = Path(args.run_dir)
    if not run.is_absolute() or str(run) == '/':
        raise ValueError('--run-dir must be an explicit absolute directory')
    if not (HERE/'train.py').is_file():
        raise FileNotFoundError(HERE/'train.py')
    if not 1024 <= args.port <= 65535:
        raise ValueError('--port must be between 1024 and 65535')
    cluster = run/'cluster'
    cluster.mkdir(parents=True, exist_ok=True)
    with (cluster/'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        launch_id = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
        job = cluster/launch_id
        job.mkdir()
        parallel(hosts, sync)
        results = parallel(hosts, lambda host: json.loads(remote(host, '/usr/bin/python3', __file__, '_preflight').stdout))
        write_json(job/'preflight.json', results)
        for host in hosts:
            print(f'{host}: preflight passed; 8 free H100s; {results[host]["ib_address"]}', flush=True)
        spec = {'launch_id': launch_id, 'created_utc': now(), 'hosts': list(hosts),
                'world_size': len(hosts)*8, 'mode': args.mode, 'run_dir': str(run),
                'master_addr': results[hosts[0]]['ib_address'], 'port': args.port,
                'trainer_args': trainer_args}
        # Check the rendezvous port without occupying any GPUs.
        bind_code = ('import socket,sys; s=socket.socket(socket.AF_INET6,socket.SOCK_STREAM); '
                     's.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1); '
                     's.bind((sys.argv[1],int(sys.argv[2]))); s.close()')
        remote(hosts[0], '/usr/bin/python3', '-c', bind_code, spec['master_addr'], args.port)
        write_json(job/'launch.json', spec)
        if args.check_only:
            print(f'Source sync and preflight passed; no GPU launch. Records: {job}')
            return
        encoded = base64.b64encode(json.dumps(spec).encode()).decode()
        try:
            starts = parallel(hosts, lambda host: json.loads(remote(host, '/usr/bin/python3', __file__, '_start', encoded).stdout))
            write_json(job/'starts.json', starts)
        except BaseException:
            for host in hosts:
                try:
                    remote(host, '/usr/bin/python3', __file__, '_stop', str(run), launch_id)
                except Exception as exc:
                    print(f'Rollback error on {host}: {exc}', file=sys.stderr)
            raise
        latest = cluster/'latest'
        tmp = latest.with_name('latest.'+launch_id)
        tmp.symlink_to(launch_id)
        tmp.replace(latest)
        print(f'Detached {args.mode} launch {launch_id}: {len(hosts)} nodes / {len(hosts)*8} GPUs')
        print(f'Status: {HERE}/status_cluster.sh --run-dir {shlex.quote(str(run))}')


def main():
    command, *rest = sys.argv[1:]
    if command == '_preflight':
        print(json.dumps(preflight()))
    elif command == '_worker':
        worker(rest[0])
    elif command == '_start':
        print(json.dumps(start(rest[0])))
    elif command == '_stop':
        print(json.dumps(stop(*rest)))
    elif command == '_status':
        print(json.dumps(node_status(rest[0], int(rest[1]))))
    elif command == 'launch':
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--run-dir', required=True)
        parser.add_argument('--mode', choices=('smoke', 'train'), default='train')
        parser.add_argument('--hosts', help='Comma-separated configured hosts; subsets allowed only for smoke')
        parser.add_argument('--port', type=int, default=29573)
        parser.add_argument('--check-only', action='store_true')
        args, trainer_args = parser.parse_known_args(rest)
        launch(args, trainer_args)
    elif command in ('status', 'stop'):
        parser = argparse.ArgumentParser()
        parser.add_argument('--run-dir', required=True)
        parser.add_argument('--tail', type=int, default=5)
        args = parser.parse_args(rest)
        spec = json.loads((Path(args.run_dir)/'cluster/latest/launch.json').read_text())
        hosts = spec['hosts']
        # Existing six-node runs remain inspectable/stoppable after expansion.
        permitted(hosts, spec['mode'], require_full=False)
        if command == 'stop':
            results = parallel(hosts, lambda host: json.loads(remote(host, '/usr/bin/python3', __file__, '_stop', args.run_dir, spec['launch_id']).stdout))
            for host in hosts:
                print(json.dumps(results[host]))
            return
        results = parallel(hosts, lambda host: json.loads(remote(host, '/usr/bin/python3', __file__, '_status', args.run_dir, args.tail, timeout=30).stdout))
        for host in hosts:
            print(json.dumps(results[host]))
        if any((d['finished'] is not None and d['finished']['exit_code'] != 0)
               or (d['finished'] is None and not (d['supervisor'] or {}).get('live'))
               for d in results.values()):
            raise SystemExit(1)
    else:
        raise ValueError(f'Unknown operation: {command}')


if __name__ == '__main__':
    main()
