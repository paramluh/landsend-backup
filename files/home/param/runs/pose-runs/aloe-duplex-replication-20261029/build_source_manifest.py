#!/usr/bin/env python3
"""Read-only, stable SHA256 inventory of the staged Aloe duplex dataset."""
import concurrent.futures
import datetime
import hashlib
import json
import os
import pathlib
import socket
import stat
import time

ROOT = pathlib.Path('/home/param/datasets/aloe-duplex')
OUT = pathlib.Path('/home/param/pose-runs/aloe-duplex-replication-20261029')
OUT.mkdir(parents=True, exist_ok=True)

def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def save(name, value):
    target = OUT / name
    temp = target.with_suffix(target.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(target)

started = utc()
ready = json.loads((ROOT / 'DATASET_READY.json').read_text())
assert ready.get('training_ready') is True, ready
files = []
for directory, subdirs, names in os.walk(ROOT, followlinks=False):
    subdirs[:] = sorted(name for name in subdirs if name != '.cache')
    for name in subdirs:
        path = pathlib.Path(directory) / name
        assert not path.is_symlink(), f'Unexpected directory symlink: {path}'
    for name in sorted(names):
        path = pathlib.Path(directory) / name
        relative = path.relative_to(ROOT).as_posix()
        if relative == 'DATASET_READY.json':
            continue
        before = path.lstat()
        assert stat.S_ISREG(before.st_mode), f'Unexpected non-regular file: {path}'
        files.append({'path': relative, 'size': before.st_size,
                      'mtime_ns': before.st_mtime_ns, 'inode': before.st_ino,
                      'device': before.st_dev})
files.sort(key=lambda item: item['path'])
base = {'source_host': socket.gethostname(), 'source_root': str(ROOT),
        'started_utc': started, 'excluded': ['**/.cache/**', 'DATASET_READY.json'],
        'symlinks': [], 'total_files': len(files),
        'total_bytes': sum(item['size'] for item in files),
        'source_readiness': ready}
save('source-files.json', dict(base, files=files))
print(json.dumps({'phase': 'inventory_ready', 'files': len(files),
                  'bytes': base['total_bytes']}), flush=True)

def signature(info):
    return info.st_size, info.st_mtime_ns, info.st_ino, info.st_dev

def calculate(item):
    path = ROOT / item['path']
    before = path.lstat()
    expected = item['size'], item['mtime_ns'], item['inode'], item['device']
    assert stat.S_ISREG(before.st_mode) and signature(before) == expected, str(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        assert signature(os.fstat(stream.fileno())) == expected, str(path)
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
        assert signature(os.fstat(stream.fileno())) == expected, str(path)
    after = path.lstat()
    assert stat.S_ISREG(after.st_mode) and signature(after) == expected, str(path)
    return {'path': item['path'], 'size': item['size'], 'sha256': digest.hexdigest()}

done = []
completed_bytes = 0
last_update = time.monotonic()
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    futures = [pool.submit(calculate, item) for item in files]
    for future in concurrent.futures.as_completed(futures):
        item = future.result()
        done.append(item)
        completed_bytes += item['size']
        if time.monotonic() - last_update >= 20:
            progress = {'phase': 'hashing', 'completed_files': len(done),
                        'completed_bytes': completed_bytes, 'total_files': len(files),
                        'total_bytes': base['total_bytes'], 'updated_utc': utc()}
            save('source-hash-progress.json', progress)
            print(json.dumps(progress), flush=True)
            last_update = time.monotonic()
for item in files:
    now = (ROOT / item['path']).lstat()
    assert signature(now) == (item['size'], item['mtime_ns'], item['inode'], item['device']), item['path']
done.sort(key=lambda item: item['path'])
assert len(done) == base['total_files'] and completed_bytes == base['total_bytes']
result = dict(base, completed_utc=utc(), stable_files=True,
              hash_algorithm='sha256', hash_workers=4, files=done)
save('source-sha256.json', result)
print(json.dumps({'phase': 'complete', 'files': len(done), 'bytes': completed_bytes}), flush=True)
