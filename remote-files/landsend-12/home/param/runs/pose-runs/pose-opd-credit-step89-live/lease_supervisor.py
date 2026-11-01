"""Hold the node GPU lease from staging through the live services' lifetime.

The operator writes START.json after LEASE_READY.json appears. Its schema is
{lease_id, services: [{name, argv, cwd, env?, env_file?}]}. Environment JSON stays
in private files and is never copied into status output. STOP.json or SIGTERM
stops only this supervisor's verified process groups. Existing services are not
adopted, killed, or restarted. Any foreign GPU process is reported for review.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time
import uuid


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.chmod(0o600)
    temporary.replace(path)


def process_identity(pid):
    try:
        path = Path(f'/proc/{int(pid)}')
        fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': int(pid), 'start_ticks': fields[19], 'uid': path.stat().st_uid,
                'pgrp': int(fields[2]), 'session': int(fields[3]), 'state': fields[0]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_process(record):
    current = process_identity(record['pid'])
    return current is not None and current['start_ticks'] == record['start_ticks']


def gpu_processes():
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name',
                                      '--format=csv,noheader,nounits'], text=True)
    result = []
    for line in output.splitlines():
        gpu, pid, name = [part.strip() for part in line.split(',', 2)]
        identity = process_identity(int(pid))
        # NVML can briefly retain entries after an owned process exits.
        if identity is not None:
            result.append({'gpu_uuid': gpu, 'process_name': name, **identity})
    return result


class Service:
    def __init__(self, spec, run):
        self.name = spec['name']
        environment = dict(os.environ)
        if spec.get('env_file'):
            environment.update(json.loads(Path(spec['env_file']).read_text()))
        environment.update(spec.get('env', {}))
        if any(not isinstance(k, str) or not isinstance(v, str) for k, v in environment.items()):
            raise ValueError('Service environment keys and values must be strings')
        self.log = (run / f'{self.name}.log').open('ab', buffering=0)
        self.child = subprocess.Popen(spec['argv'], cwd=spec['cwd'], env=environment,
                                      stdin=subprocess.DEVNULL, stdout=self.log, stderr=subprocess.STDOUT,
                                      start_new_session=True)
        self.identity = process_identity(self.child.pid)
        if self.identity is None:
            raise RuntimeError(f'Service {self.name} exited before its identity could be recorded')
        self.members = {self.child.pid: self.identity}
        self.returncode = None

    def refresh(self):
        self.returncode = self.child.poll()
        # A verified member anchors the session identity before finding new
        # children. Other sessions cannot join this independently created group.
        if any(same_process(record) for record in self.members.values()):
            for path in Path('/proc').iterdir():
                if not path.name.isdigit():
                    continue
                record = process_identity(int(path.name))
                if (record and record['uid'] == os.getuid()
                        and record['pgrp'] == self.identity['pgrp']
                        and record['session'] == self.identity['session']):
                    self.members[record['pid']] = record

    def signal(self, value):
        self.refresh()
        if any(same_process(record) for record in self.members.values()):
            try:
                os.killpg(self.identity['pgrp'], value)
            except ProcessLookupError:
                pass

    def alive(self):
        self.refresh()
        return any(same_process(record) and process_identity(record['pid'])['state'] != 'Z'
                   for record in self.members.values())

    def receipt(self):
        return {'name': self.name, 'leader': self.identity, 'members': list(self.members.values()),
                'exit_code': self.returncode}


def validate_start(spec, lease_id):
    if spec.get('lease_id') != lease_id:
        raise ValueError('START.json does not name this GPU lease')
    services = spec.get('services')
    if not isinstance(services, list) or not services:
        raise ValueError('START.json requires a nonempty services list')
    names = set()
    for service in services:
        name = service.get('name')
        if not isinstance(name, str) or not re.fullmatch('[a-z][a-z0-9_-]*', name) or name in names:
            raise ValueError('Service names must be unique safe identifiers')
        names.add(name)
        argv = service.get('argv')
        if (not isinstance(argv, list) or not argv or any(not isinstance(arg, str) for arg in argv)
                or not Path(argv[0]).is_absolute() or not Path(argv[0]).is_file()):
            raise ValueError('Service argv must begin with an existing absolute executable')
        if not Path(service.get('cwd', '')).is_absolute() or not Path(service['cwd']).is_dir():
            raise ValueError('Service cwd must be an existing absolute directory')
        if service.get('env_file') and not Path(service['env_file']).is_absolute():
            raise ValueError('Environment file path must be absolute')


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--lock', type=Path,
                        default=Path('/home/param/runs/pose-runs/.duplex-distill-gpus.lock'))
    parser.add_argument('--startup-timeout', type=float, default=3600)
    args = parser.parse_args(argv)
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    run.chmod(0o700)
    if any((run / name).exists() for name in ('LEASE_READY.json', 'STARTED.json', 'FINISHED.json')):
        raise RuntimeError('Use a fresh supervisor directory; refusing to replace an existing lease')
    services, stopped = [], {'requested': False}
    signal.signal(signal.SIGTERM, lambda *_: stopped.update(requested=True))
    signal.signal(signal.SIGINT, lambda *_: stopped.update(requested=True))
    started, lease_id = time.time(), str(uuid.uuid4())
    with args.lock.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if gpu_processes():
            raise RuntimeError('GPU processes already exist; no services were touched')
        devices = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid',
                                           '--format=csv,noheader'], text=True).splitlines()
        identity = {'lease_id': lease_id, 'host': socket.gethostname(), 'supervisor': process_identity(os.getpid()),
                    'lock': str(args.lock), 'reserved_devices': devices, 'started': started}
        atomic_json(run / 'LEASE_READY.json', identity)
        status, error = 'waiting_for_start', None
        try:
            while not stopped['requested'] and not (run / 'STOP.json').exists():
                if not services and (run / 'START.json').exists():
                    raw = (run / 'START.json').read_bytes()
                    spec = json.loads(raw)
                    validate_start(spec, lease_id)
                    if gpu_processes():
                        raise RuntimeError('GPU processes appeared before launch; no existing processes touched')
                    for item in spec['services']:
                        service = Service(item, run)
                        services.append(service)
                        atomic_json(run / 'STARTED.json', {**identity, 'manifest_sha256': hashlib.sha256(raw).hexdigest(),
                                    'services': [value.receipt() for value in services]})
                    status = 'running'
                for service in services:
                    service.refresh()
                    if service.returncode is not None:
                        raise RuntimeError(f'Live service {service.name} exited with code {service.returncode}')
                known = {pid for service in services for pid, record in service.members.items() if same_process(record)}
                foreign = [record for record in gpu_processes() if record['pid'] not in known]
                if foreign:
                    atomic_json(run / 'GPU_CONFLICT.json', {'time': time.time(), 'processes': foreign,
                                'action': 'reported only; no foreign processes signaled'})
                atomic_json(run / 'STATUS.json', {**identity, 'status': status, 'time': time.time(),
                            'services': [service.receipt() for service in services], 'foreign_gpu_processes': foreign})
                if not services and time.time() - started > args.startup_timeout:
                    raise TimeoutError('No START.json received within the staging lease timeout')
                time.sleep(2)
            status = 'stopped'
        except BaseException as exc:
            status, error = 'failed', f'{type(exc).__name__}: {exc}'
            raise
        finally:
            for service in services:
                service.signal(signal.SIGTERM)
            until = time.monotonic() + 30
            while any(service.alive() for service in services) and time.monotonic() < until:
                time.sleep(.25)
            for service in services:
                service.signal(signal.SIGKILL)
                service.child.wait(timeout=10)
                service.log.close()
            atomic_json(run / 'FINISHED.json', {**identity, 'status': status, 'error': error, 'time': time.time(),
                        'services': [service.receipt() for service in services]})


if __name__ == '__main__':
    main()
