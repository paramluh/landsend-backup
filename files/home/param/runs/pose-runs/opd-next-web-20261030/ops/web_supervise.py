#!/usr/bin/env python3
"""Keep Amu's unchanged proxy and its Cloudflare quick tunnel running."""
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import threading
import time

ROOT = Path('/home/param/pose-runs/opd-next-web-20261030')
PY = '/home/param/pose-repo/.venv/bin/python'
STOP = threading.Event()
CHILDREN = set()
LOCK = threading.Lock()


def record(name, **data):
    data.update(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), host=socket.gethostname())
    target = ROOT / 'ops' / (name + '.json')
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(target)
    print(name, json.dumps(data), flush=True)


def run(name, command, tunnel=False):
    while not STOP.is_set():
        logpath = ROOT / 'logs' / f'{name}-{time.time_ns()}.log'
        with logpath.open('x') as log:
            child = subprocess.Popen(command, cwd=ROOT / 'srv', stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            with LOCK:
                CHILDREN.add(child)
            record(name, state='running', pid=child.pid, log=str(logpath))
            url = None
            while child.poll() is None and not STOP.wait(2):
                if tunnel and not url:
                    matches = re.findall(r'https://[a-z0-9-]+\.trycloudflare\.com', logpath.read_text())
                    if matches:
                        url = matches[0]
                        temporary = ROOT / 'URL.tmp'
                        temporary.write_text(url + '\n')
                        temporary.replace(ROOT / 'URL.txt')
                        record(name, state='running', pid=child.pid, log=str(logpath), url=url)
            if STOP.is_set() and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
            child.wait()
            with LOCK:
                CHILDREN.discard(child)
            record(name, state='exited', returncode=child.returncode, log=str(logpath))
        STOP.wait(10)


def stop(signum, frame):
    STOP.set()
    with LOCK:
        for child in list(CHILDREN):
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)


if __name__ == '__main__':
    assert socket.gethostname().split('.')[0] == 'landsend-5'
    lock = (ROOT / 'ops/web.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    record('web_supervisor', state='running', pid=os.getpid())
    workers = [
        threading.Thread(target=run, args=('proxy', [PY, '-u', str(ROOT / 'srv/proxy.py'),
            '--port', '8917', '--backend', '8918', '--token-file', str(ROOT / 'backend.token'),
            '--sessions', str(ROOT / 'state/sessions'), '--lb', str(ROOT / 'lb.json')])),
        threading.Thread(target=run, args=('cloudflared', ['/home/param/bin/cloudflared', 'tunnel',
            '--no-autoupdate', '--protocol', 'http2', '--url', 'http://127.0.0.1:8917'], True)),
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
