#!/usr/bin/env python3
"""Download and hash the pinned PPMVOX parent without loading any model."""
import os
os.environ.update(HF_HUB_DISABLE_PROGRESS_BARS="1", HF_HUB_DISABLE_TELEMETRY="1",
                  HF_XET_HIGH_PERFORMANCE="1", HF_XET_NUM_CONCURRENT_RANGE_GETS="8")
import argparse
import concurrent.futures as cf
import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import socket
import threading
import time
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import logging as hf_logging

hf_logging.set_verbosity_error()
parser = argparse.ArgumentParser()
parser.add_argument('--audit', type=Path, required=True)
parser.add_argument('--workers', type=int, default=12)
args = parser.parse_args()
audit = args.audit
manifest = json.loads((audit / 'source-manifest.json').read_text())
target = Path('/home/param/models/imported-weights/ppmvox-v5-18100')
stage = target.with_name('.ppmvox-v5-18100.staging-9de2b20a')
host = socket.gethostname().split('.')[0]
started = time.time()
lock = threading.Lock()
active = {}
results = []
failures = []
def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()
def atomic_json(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as stream:
        json.dump(data, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)
def status(phase):
    with lock:
        current = dict(active)
    doc = dict(host=host, pid=os.getpid(), phase=phase, utc=utc(),
               started_epoch=started, elapsed_seconds=round(time.time()-started, 1),
               repo=manifest['repo'], revision=manifest['revision'],
               target=str(target), staging=str(stage),
               verified_files=len(results), total_files=manifest['total_file_count'],
               verified_bytes=sum(x['size'] for x in results),
               total_bytes=manifest['total_bytes'], active=current, failures=failures)
    atomic_json(audit / 'node-status.json', doc)
    return doc
def verify(path, entry):
    before = path.stat()
    if path.is_symlink() or before.st_size != entry['size']:
        raise ValueError('size-or-symlink-mismatch')
    sha256 = hashlib.sha256()
    sha1 = hashlib.sha1(b'blob ' + str(before.st_size).encode() + b'\0') if not entry['lfs_sha256'] else None
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            sha256.update(block)
            if sha1 is not None:
                sha1.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError('file-changed-during-verification')
    actual = sha256.hexdigest()
    if entry['lfs_sha256']:
        if actual != entry['lfs_sha256']:
            raise ValueError('sha256-mismatch')
    elif sha1.hexdigest() != entry['blob_id']:
        raise ValueError('git-blob-sha1-mismatch')
    return dict(path=entry['path'], size=entry['size'], sha256=actual,
                verification='hf-lfs-sha256' if entry['lfs_sha256'] else 'hf-git-blob-sha1',
                expected_hash=entry['lfs_sha256'] or entry['blob_id'])
def fetch(entry):
    name = entry['path']
    assert not Path(name).is_absolute() and '..' not in Path(name).parts
    last_error = None
    for attempt in range(1, 5):
        try:
            with lock:
                active[name] = dict(phase='downloading', attempt=attempt)
            path = Path(hf_hub_download(repo_id=manifest['repo'], revision=manifest['revision'],
                                       filename=name, local_dir=stage, force_download=attempt>1))
            assert path == stage / name
            with lock:
                active[name] = dict(phase='hashing', attempt=attempt)
            result = verify(path, entry)
            with lock:
                active.pop(name, None)
            return result
        except Exception as exc:
            last_error = type(exc).__name__
            with lock:
                active[name] = dict(phase='retry', attempt=attempt, error_type=last_error)
            if attempt < 4:
                time.sleep(min(2**attempt, 16))
    with lock:
        active.pop(name, None)
    raise RuntimeError(name + ': ' + str(last_error))
def main():
    if target.exists() or target.is_symlink():
        raise FileExistsError('canonical-target-already-exists')
    target.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(target.parent).free < manifest['total_bytes'] + 20*1024**3:
        raise RuntimeError('insufficient-free-space')
    stage.mkdir(exist_ok=True)
    prior = stage / 'SOURCE.json'
    if prior.exists():
        old = json.loads(prior.read_text())
        assert old['repo'] == manifest['repo'] and old['revision'] == manifest['revision']
    atomic_json(prior, manifest)
    status('downloading')
    # Large-first schedules avoid a long tail; every repeated source file is retained.
    entries = sorted(manifest['files'], key=lambda x: x['size'], reverse=True)
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(fetch, entry): entry['path'] for entry in entries}
        while pending:
            done, _ = cf.wait(pending, timeout=10, return_when=cf.FIRST_COMPLETED)
            for future in done:
                name = pending.pop(future)
                try:
                    results.append(future.result())
                except Exception as exc:
                    failures.append(dict(path=name, error_type=type(exc).__name__))
            status('downloading-and-verifying')
    if failures:
        status('failed')
        raise RuntimeError('download-or-verification-failures')
    expected = {x['path'] for x in manifest['files'] if x['path'].startswith('w-18100/')}
    actual = {str(x.relative_to(stage)) for x in (stage/'w-18100').rglob('*') if x.is_file()}
    assert actual == expected and len(actual) == 208
    assert not (stage/'w-17992').exists() and not (stage/'w-18084').exists()
    metas = []
    for path in sorted((stage/'w-18100').glob('*/meta.json')):
        data = json.loads(path.read_text())
        assert data['step'] == 18100
        metas.append(dict(path=str(path.relative_to(stage)), step=data['step'],
                          run_tag=data.get('run_tag'), world=data.get('world'), src=data.get('src')))
    assert len(metas) == 16
    receipt = dict(host=host, completed_utc=utc(), repo=manifest['repo'], revision=manifest['revision'],
                   target=str(target), checkpoint_file_count=208, checkpoint_bytes=manifest['checkpoint_bytes'],
                   total_files=len(results), total_bytes=sum(x['size'] for x in results),
                   all_hashes_pass=True, original_layout_preserved=True,
                   metadata=metas, files=sorted(results,key=lambda x:x['path']))
    atomic_json(stage/'VERIFIED.json', receipt)
    atomic_json(stage/'DOWNLOAD_COMPLETE.json', {k:v for k,v in receipt.items() if k not in ('files','metadata')})
    if target.exists() or target.is_symlink():
        raise FileExistsError('target-appeared-before-publication')
    stage.rename(target)
    atomic_json(audit/'node-verification.json', receipt)
    status('complete')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('files','metadata')}), flush=True)
try:
    main()
except BaseException as exc:
    failures.append(dict(error_type=type(exc).__name__))
    status('failed')
    print(json.dumps(dict(host=host,status='failed',error_type=type(exc).__name__)),flush=True)
    raise SystemExit(1)
