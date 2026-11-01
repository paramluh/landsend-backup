"""Pause only the audited renderer between jobs, always resume it afterward."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

root = Path('/home/param/outputs/pose-opd-video-worker-20261031')
pid = 8523
def start_ticks():
    return Path(f'/proc/{pid}/stat').read_text().split()[21]
ticks = start_ticks()
status = json.loads((root/'status.json').read_text())
job = status['job_id']
print(json.dumps({'event':'waiting_for_job_boundary', 'pid':pid, 'start_ticks':ticks, 'job':job}), flush=True)
while not (root/'completed'/job/'result.json').is_file():
    if start_ticks() != ticks:
        raise RuntimeError('Worker identity changed while waiting')
    time.sleep(.1)
paused = False
probe = None
try:
    assert start_ticks() == ticks
    os.kill(pid, signal.SIGSTOP)
    paused = True
    print(json.dumps({'event':'worker_paused', 'pid':pid, 'time':time.time(), 'completed_job':job}), flush=True)
    time.sleep(2)  # Permit any already-submitted GPU work to drain.
    command = 'source /home/param/outputs/pose-opd-video-worker-20261031/environment.sh && exec python -u /home/param/outputs/pose-opd-video-worker-20261031/vae-batch-probe/probe.py'
    with (root/'vae-batch-probe/probe.log').open('w') as log:
        probe = subprocess.Popen(['bash','-c',command], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = probe.wait(timeout=115)
            print(json.dumps({'event':'probe_exit','code':code}),flush=True)
        except subprocess.TimeoutExpired:
            os.killpg(probe.pid, signal.SIGTERM)
            try:
                probe.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(probe.pid, signal.SIGKILL)
                probe.wait()
            print(json.dumps({'event':'probe_timeout'}),flush=True)
finally:
    if paused and start_ticks() == ticks:
        os.kill(pid, signal.SIGCONT)
        print(json.dumps({'event':'worker_resumed','pid':pid,'time':time.time()}),flush=True)
