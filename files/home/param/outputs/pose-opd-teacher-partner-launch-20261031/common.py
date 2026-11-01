"""Pinned seven-node production training launch; saved checkpoints and latest optimizer."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess

HOSTS = ['landsend-0', 'landsend-1', 'landsend-2', 'landsend-3', 'landsend-4', 'landsend-5', 'landsend-12']
RUN = '/home/param/runs/pose-runs/pose-opd-teacher-partner-b336-general-20261031'
PYTHON = '/home/param/pose-repo/.venv/bin/python'
ACTIVATE = '/home/param/pose-repo/finetune/transcript_pose/environment/activate.sh'
LOCK = Path('/home/param/runs/pose-runs/.duplex-distill-gpus.lock')
SSH = ['ssh', '-i', '/home/param/.ssh/id_ed25519_cluster', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=10']
HELPERS = ('common.py', 'worker.py')
EXTRA_ARGS = []

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    with tmp.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)

def validate_spec(spec):
    if spec['hosts'] != HOSTS or spec['run_dir'] != RUN or spec['world_size'] != 56:
        raise ValueError('Unexpected production hosts, run directory or world size')
    if len(spec['source_commit']) != 40 or any(c not in '0123456789abcdef' for c in spec['source_commit']):
        raise ValueError('Expected a complete git commit SHA')
    expected = RUN + '/code'
    if spec['repo'] != expected or spec['config'] != expected + '/finetune/pose_opd_credit/recipe.teacher_partner.json':
        raise ValueError('Checkout/config paths do not match the immutable production snapshot')
    if (spec.get('mode') != 'three-epoch-production' or spec.get('extra_args') != EXTRA_ARGS
            or spec.get('checkpointing_enabled') is not True or spec.get('normal_adam_initialization') is not True):
        raise ValueError('Only the full three-epoch run with saved checkpoints and normal Adam initialization is supported')

def command(spec, host):
    validate_spec(spec)
    return [PYTHON, '-m', 'torch.distributed.run', '--nnodes=7', '--nproc_per_node=8',
            '--node_rank=' + str(HOSTS.index(host)), '--master_addr=' + spec['master_addr'],
            '--master_port=' + str(spec['port']), '--max_restarts=0',
            '-m', 'finetune.pose_opd_credit.train', '--config', spec['config'], '--run-dir', RUN, *spec['extra_args']]

def record(pid):
    fields = Path('/proc/%d/stat' % pid).read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'start_ticks': int(fields[19])}

def remote(host, argv):
    return subprocess.run(SSH + [host, shlex.join(list(map(str, argv)))], text=True,
                          capture_output=True, check=True, timeout=180)

def verify_files(spec, job):
    validate_spec(spec)
    job = Path(job)
    if sha(job / 'source.tar') != spec['archive_sha256']:
        raise RuntimeError('Source archive SHA mismatch')
    if sha(job / 'source-files.json') != spec['source_manifest_sha256']:
        raise RuntimeError('Source manifest SHA mismatch')
    manifest = json.loads((job / 'source-files.json').read_text())
    root = Path(spec['repo'])
    actual = {str(path.relative_to(root)) for path in root.rglob('*') if path.is_file() or path.is_symlink()}
    if actual != set(manifest):
        raise RuntimeError('Source checkout contains missing or unexpected files')
    for relative, expected in manifest.items():
        path = root / relative
        if path.is_symlink() or sha(path) != expected:
            raise RuntimeError('Committed source file mismatch: ' + relative)
        if path.stat().st_mode & 0o222:
            raise RuntimeError('Source checkout is not read-only: ' + relative)
    if sha(spec['config']) != spec['config_sha256']:
        raise RuntimeError('Recipe SHA mismatch')
    for name, expected in spec['helper_sha256'].items():
        if sha(job / name) != expected:
            raise RuntimeError('Launch helper SHA mismatch: ' + name)
    if not Path(PYTHON).is_file() or not Path(ACTIVATE).is_file():
        raise RuntimeError('Original environment is missing')

def gpu_check():
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                   '--format=csv,noheader,nounits'], text=True).strip()
    if apps:
        raise RuntimeError('Existing GPU compute processes: ' + apps)
    lines = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                    '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    rows = [[int(value.strip()) for value in line.split(',')] for line in lines]
    if len(rows) != 8 or sorted(row[0] for row in rows) != list(range(8)):
        raise RuntimeError('Expected exactly eight visible GPUs')
    if any(memory >= 2048 or utilization != 0 for _, memory, utilization in rows):
        raise RuntimeError('GPU memory/utilization is not idle: ' + repr(rows))
    return rows
