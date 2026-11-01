"""Production supervisor. Its child runs normal checkpoint-enabled training."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from common import ACTIVATE, HOSTS, LOCK, command, gpu_check, record, verify_files, write_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('spec', type=Path)
    parser.add_argument('--check', action='store_true', help='Read-only preflight; never launch')
    parser.add_argument('--dry-command', action='store_true')
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    host = socket.gethostname().split('.')[0]
    cmd = command(spec, host)
    if args.dry_command:
        print(json.dumps(cmd)); return
    job = args.spec.parent
    verify_files(spec, job)
    if args.check:
        for name in ('started-launch.json', 'supervisor.json', 'torchrun.json', 'READY.json', 'GO.json'):
            if (job / name).exists():
                raise RuntimeError('Prior launch marker exists: ' + name)
        if (Path(spec['run_dir']) / 'config.json').exists():
            raise RuntimeError('Existing training identity; fresh launch refused')
        metrics = Path(spec['run_dir']) / 'metrics.jsonl'
        if metrics.exists() and metrics.stat().st_size:
            raise RuntimeError('Existing training metrics; fresh launch refused')
        if LOCK.exists():
            with LOCK.open('r') as stream:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                devices = gpu_check()
        else:
            devices = gpu_check()
        print(json.dumps({'host': host, 'read_only': True, 'gpus': devices, 'command': cmd})); return
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gpu_check()
        with (job / 'supervisor.json').open('x') as stream:
            json.dump(dict(record(os.getpid()), host=host, started_at=time.time()), stream)
        write_json(job / 'command.json', cmd)
        write_json(job / 'READY.json', dict(record(os.getpid()), host=host))
        child = None
        def stop(signum, frame):
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
            else:
                raise SystemExit(128 + signum)
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        deadline = time.monotonic() + 120
        while not (job / 'GO.json').is_file():
            if time.monotonic() > deadline:
                raise RuntimeError('No all-host GO within 120 seconds; no training was started')
            time.sleep(0.2)
        go = json.loads((job / 'GO.json').read_text())
        if go['source_commit'] != spec['source_commit'] or go['config_sha256'] != spec['config_sha256']:
            raise RuntimeError('GO identity mismatch')
        gpu_check()
        env = os.environ.copy()
        env.update(PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', NCCL_SOCKET_IFNAME='=ibs14',
                   NCCL_SOCKET_FAMILY='AF_INET6', GLOO_SOCKET_IFNAME='ibs14', NCCL_IB_DISABLE='0',
                   NCCL_NET='IB', TORCH_NCCL_ASYNC_ERROR_HANDLING='1', OMP_NUM_THREADS='1', NCCL_DEBUG='WARN',
                   CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
                   S4_FFMPEG='/home/param/pose-repo/.venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2')
        with (job / 'training.log').open('ab', buffering=0) as output:
            old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT})
            try:
                child = subprocess.Popen(['bash', '-c', 'set -e; source ' + ACTIVATE + '; exec "$@"',
                                          'pose-opd-production', *cmd], env=env, cwd=spec['repo'],
                                         stdout=output, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                          start_new_session=True,
                                          preexec_fn=lambda: signal.pthread_sigmask(signal.SIG_SETMASK, old_mask))
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
            write_json(job / 'torchrun.json', dict(record(child.pid), started_at=time.time()))
            rc = child.wait()
        write_json(job / 'finished.json', {'time': time.time(), 'exit_code': rc})
        return rc

if __name__ == '__main__':
    sys.exit(main())
