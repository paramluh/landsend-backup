#!/usr/bin/env python3
"""Read-only allowlisted telemetry mirror. Never reads checkpoints or weights."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

HOST = 'landsend-0'
SOURCE = '/home/param/runs/pose-runs/pose-opd-teacher-partner-b336-general-20261031'
MIRROR = Path('/home/param/outputs/pose-opd-teacher-partner-b336-general-live-20261031')
SSH = ['ssh', '-i', '/home/param/.ssh/id_ed25519_cluster', '-o', 'BatchMode=yes',
       '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=5']
# Individual hard limits, total decoded transfer <=32MiB. No globbing/recursion.
FILES = {'config.json': 2<<20, 'plan.json': 4<<20, 'metrics.jsonl': 4<<20,
         'events-rank0.jsonl': 16<<20, 'RESULT.json': 2<<20,
         'data-selection.json': 2<<20, 'supervisor/command.json': 1<<16,
         'supervisor/spec.json': 1<<16, 'supervisor/supervisor.json': 1<<16,
         'supervisor/torchrun.json': 1<<16, 'supervisor/finished.json': 1<<16,
         'supervisor/training.log': 1<<20}

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    temp.write_text(json.dumps(value, indent=2) + '\n'); os.replace(temp, path)

def reader_script():
    return '''import base64,hashlib,json,pathlib,time
root=pathlib.Path(%r);limits=%r;out={};total=0
for name,limit in limits.items():
 p=root/name
 if not p.exists():continue
 if p.is_symlink() or not p.is_file():raise RuntimeError('Nonregular telemetry path: '+name)
 stat=p.stat();size=stat.st_size
 with p.open('rb') as stream:
  truncated=size>limit
  if truncated:
   if not name.endswith(('.jsonl','.log')):raise RuntimeError('Oversized metadata: '+name)
   stream.seek(size-limit)
  data=stream.read(limit)
 if truncated:data=data.split(b'\\n',1)[-1]
 if name.endswith('.jsonl') and not data.endswith(b'\\n'):data=data.rsplit(b'\\n',1)[0]+b'\\n' if b'\\n' in data else b''
 total+=len(data)
 if total>32*(1<<20):raise RuntimeError('Telemetry exceeds total limit')
 out[name]={'base64':base64.b64encode(data).decode(),'sha256':hashlib.sha256(data).hexdigest(),'source_bytes':size,'copied_bytes':len(data),'truncated':truncated,'source_mtime':stat.st_mtime}
print(json.dumps({'source_exists':root.is_dir(),'raw_exists':(root/'raw-rollouts').is_dir(),'remote_time':time.time(),'files':out}))
''' % (SOURCE, FILES)

def install(snapshot, mirror=MIRROR):
    mirror = Path(mirror); mirror.mkdir(parents=True, exist_ok=True)
    generations = mirror / 'snapshots'; generations.mkdir(exist_ok=True)
    stamp = time.time(); target = generations / (str(time.time_ns()) + '-' + str(os.getpid()))
    target.mkdir()
    receipts, total = {}, 0
    for name, value in snapshot['files'].items():
        if name not in FILES:
            raise ValueError('Unapproved telemetry file: ' + name)
        data = base64.b64decode(value['base64'], validate=True)
        if len(data) > FILES[name] or hashlib.sha256(data).hexdigest() != value['sha256']:
            raise ValueError('Telemetry size/hash mismatch: ' + name)
        total += len(data)
        if total > 32 << 20:
            raise ValueError('Telemetry exceeds total limit')
        if name.endswith('.json'):
            json.loads(data)
        path = target / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
        if 'source_mtime' in value:
            os.utime(path,(value['source_mtime'],value['source_mtime']))
        receipts[name] = {key: value[key] for key in ('sha256', 'source_bytes', 'copied_bytes', 'truncated')}
    status = ('live' if 'events-rank0.jsonl' in receipts else
              'waiting_for_metrics' if snapshot['source_exists'] else 'waiting_for_run')
    health = {'status': status, 'source_host': HOST, 'source_run_dir': SOURCE, 'last_attempt': stamp,
              'last_success': stamp, 'remote_time': snapshot['remote_time'], 'files': receipts,
              'read_only': True, 'interval_seconds': 5, 'weights_copied': False}
    write_json(target / 'mirror-receipt.json', health)
    link = mirror / ('current.tmp-' + str(os.getpid()))
    link.symlink_to(target); os.replace(link, mirror / 'current')
    write_json(mirror / 'collector-health.json', health)
    import shutil
    for old in sorted(generations.iterdir(), key=lambda item: item.name)[:-6]:
        if old.is_dir():shutil.rmtree(old)
    return health

def collect_once(mirror=MIRROR):
    try:
        result = subprocess.run(SSH + [HOST, shlex.join(['python3', '-c', reader_script()])],
                                capture_output=True, text=True, check=True, timeout=20)
        snapshot=json.loads(result.stdout)
        health=install(snapshot, mirror)
        if snapshot.get('raw_exists'):
            # Copy payloads first and publish manifests only in the second pass.
            # Relative allowlists never traverse checkpoint/model directories.
            dest=Path(mirror)/'raw-rollouts';dest.mkdir(exist_ok=True)
            base=['rsync','-rlt','--no-links','--max-size=32m','--delay-updates','-e',shlex.join(SSH)]
            source=HOST+':'+SOURCE+'/raw-rollouts/'
            for pattern in ('/update-??????/sample-*.npz','/update-??????/manifest.json'):
                subprocess.run(base+['--include=/update-??????/','--include='+pattern,'--exclude=*',source,str(dest)+'/'],
                               check=True,capture_output=True,text=True,timeout=45)
            health['raw_rollouts_mirrored']=True
            write_json(Path(mirror)/'collector-health.json',health)
        return health
    except Exception as error:
        path = Path(mirror) / 'collector-health.json'
        previous = json.loads(path.read_text()) if path.exists() else {}
        previous.update(status='error', last_attempt=time.time(), error=str(error)[:1000],
                        source_host=HOST, source_run_dir=SOURCE, read_only=True, weights_copied=False)
        write_json(path, previous)
        return previous

def main():
    global SOURCE, MIRROR
    parser = argparse.ArgumentParser(); parser.add_argument('--once', action='store_true')
    parser.add_argument('--source-run-dir',default=SOURCE)
    parser.add_argument('--mirror',type=Path,default=MIRROR)
    args = parser.parse_args(); SOURCE=args.source_run_dir; MIRROR=args.mirror
    if SOURCE != '/home/param/runs/pose-runs/pose-opd-teacher-partner-b336-general-20261031':
        parser.error('Collector source must be an explicitly selected pose OPD production attempt')
    while True:
        started = time.monotonic(); print(json.dumps(collect_once(MIRROR)), flush=True)
        if args.once:return
        time.sleep(max(1., 5. - (time.monotonic() - started)))

if __name__ == '__main__':main()
