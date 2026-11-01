import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

variant = sys.argv[1]
root = Path('/home/param/outputs/pose-opd-credit-bench')
job = root / ('actor-' + variant)
job.mkdir(exist_ok=True)
repo = Path('/home/param/pose-worktrees/opd-bench-actor8-fp32-60s-v6s')
def receipt(pid):
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'start_ticks': fields[19]}
with Path('/home/param/runs/pose-runs/.duplex-distill-gpus.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
    if busy:
        raise RuntimeError('Existing GPU process: ' + busy)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PPM_AUTOTUNE='0', PYTHONUNBUFFERED='1', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
               S4_FFMPEG='/home/param/pose-repo/.venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2')
    command = ['/home/param/pose-repo/.venv/bin/python', str(root / 'actor_variants.py'),
               '--repo', str(repo), '--config', str(repo / 'benchmark-config.json'),
               '--output', str(root / ('actor-' + variant + '.json')), '--frames', '750',
               '--capacity-frames', '7500', '--profile-stages']
    if variant == 'tuned':
        command += ['--tune-gemv', '--variants', 'baseline,tuned_down,baseline,tuned_all,baseline']
    elif variant != 'linear':
        raise ValueError(variant)
    (job / 'supervisor.json').write_text(json.dumps(receipt(os.getpid())))
    (job / 'command.json').write_text(json.dumps(command, indent=2))
    child = subprocess.Popen(['bash', '-c', 'source /home/param/pose-repo/finetune/transcript_pose/environment/activate.sh; exec "$@"',
                              'actor-bench', *command], env=env, cwd=repo, start_new_session=True)
    (job / 'child.json').write_text(json.dumps(receipt(child.pid)))
    def stop(*_):
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        code = child.wait(timeout=900)
    except subprocess.TimeoutExpired:
        stop()
        code = child.wait(timeout=30)
    (job / 'finished.json').write_text(json.dumps({'time': time.time(), 'exit_code': code}))
    sys.exit(code)
