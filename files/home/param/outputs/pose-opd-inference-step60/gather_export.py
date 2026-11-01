"""Materialize a verified inference copy without altering training checkpoints."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time

SOURCE = Path('/home/param/models/trained-weights/pose-opd-credit-2min-b320-attempt2/checkpoints/step-000060/export')
DEST = Path('/home/param/models/inference/pose-opd-credit-step60')
RUN = Path('/home/param/runs/pose-runs/pose-opd-credit-step60-live')
HOSTS = ['landsend-2', 'landsend-3', 'landsend-4', 'landsend-5', 'landsend-12']


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            result.update(block)
    return result.hexdigest()


def write(path, obj):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2) + '\n')
    temporary.replace(path)


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    began = time.time()
    fields = Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()
    write(RUN / 'gather-process.json', {'pid': os.getpid(), 'start_ticks': fields[19], 'started': began})
    if DEST.exists():
        raise RuntimeError(f'Refusing to overwrite existing inference export: {DEST}')
    original = SOURCE.joinpath('meta.json').read_bytes()
    meta = json.loads(original)
    if (meta['step'] != 60 or meta['world'] != 40
            or meta['storage_layout'] != 'node-local-shards'):
        raise RuntimeError('Unexpected source checkpoint identity')
    receipts = []
    for host in HOSTS:
        command = ['ssh', '-o', 'BatchMode=yes', host, 'python3', '-c',
                   "'import json,pathlib,hashlib; p=pathlib.Path(\"" + str(SOURCE)
                   + "\"); b=(p/\"meta.json\").read_bytes(); print(json.dumps({\"meta\":json.loads(b),\"meta_sha256\":hashlib.sha256(b).hexdigest(),\"complete\":json.loads((p/\"NODE_COMPLETE\").read_text())}))'"]
        receipt = json.loads(subprocess.check_output(command, text=True))
        source_meta, complete = receipt['meta'], receipt['complete']
        if (source_meta['step'] != 60 or source_meta['world'] != 40 or source_meta['host'] != host
                or source_meta['cfg'] != meta['cfg'] or source_meta['files'] != meta['files']
                or complete['host'] != host or complete['step'] != 60
                or complete['ranks'] != [i for i, owner in enumerate(meta['hosts']) if owner == host]):
            raise RuntimeError(f'Source host manifest identity differs: {host}')
        receipts.append({'host': host, 'metadata_sha256': receipt['meta_sha256'], 'node_complete': complete})
    entries = {}
    for entry in meta['files']:
        name = entry['file']
        if name in entries and any(entry[k] != entries[name][k] for k in ('bytes', 'sha256')):
            raise RuntimeError(f'Replicated source file mismatch: {name}')
        entries.setdefault(name, entry)
    if set(entries) != {f'rank{i}.pt' for i in range(40)} | {'rep.pt'}:
        raise RuntimeError('Export does not describe exactly forty shards and replicated state')
    DEST.parent.mkdir(parents=True, exist_ok=True)
    total = sum(e['bytes'] for e in entries.values())
    if shutil.disk_usage(DEST.parent).free < total + (5 << 30):
        raise RuntimeError('Insufficient disk for independent inference copy')
    temporary = DEST.with_name(DEST.name + f'.staging-{os.getpid()}')
    temporary.mkdir()
    (temporary / 'SOURCE_META.json').write_bytes(original)
    write(RUN / 'gather-plan.json', {'source': str(SOURCE), 'destination': str(DEST),
          'staging': str(temporary), 'files': len(entries), 'bytes': total, 'sources': receipts})
    verified = []

    def gather_host(host):
        result = []
        for name, entry in entries.items():
            if entry['host'] != host:
                continue
            path = temporary / name
            if host == socket.gethostname().split('.')[0]:
                subprocess.run(['cp', '--reflink=auto', str(SOURCE / name), str(path)], check=True)
            else:
                subprocess.run(['rsync', '-a', '--whole-file', '--partial', '--protect-args',
                                '-e', 'ssh -o BatchMode=yes', host + ':' + str(SOURCE / name), str(path)], check=True)
            if path.stat().st_size != entry['bytes'] or digest(path) != entry['sha256']:
                raise RuntimeError(f'Transferred export failed size/SHA256 verification: {name}')
            result.append(dict(entry, verified_sha256=True))
            print(json.dumps({'event': 'verified', 'host': host, 'file': name, 'bytes': entry['bytes'],
                              'elapsed_seconds': time.time() - began}), flush=True)
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        for result in pool.map(gather_host, HOSTS):
            verified.extend(result)
    meta['storage_layout'] = 'gathered-shards'
    meta['files'] = [entries[name] for name in sorted(entries)]
    meta['source_storage_layout'] = 'node-local-shards'
    meta['source_metadata_sha256'] = hashlib.sha256(original).hexdigest()
    write(temporary / 'meta.json', meta)
    provenance = {'format': 'pose-opd-inference-export-v1', 'step': 60, 'source_world': 40,
                  'source': str(SOURCE), 'destination': str(DEST), 'sources': receipts,
                  'source_metadata_sha256': hashlib.sha256(original).hexdigest(),
                  'files': sorted(verified, key=lambda e: e['file']), 'tensor_bytes_unchanged': True,
                  'metadata_changes': ['storage_layout', 'deduplicate identical rep.pt receipts',
                                       'source_storage_layout', 'source_metadata_sha256'],
                  'elapsed_seconds': time.time() - began}
    write(temporary / 'INFERENCE_EXPORT.json', provenance)
    (temporary / 'COMPLETE').write_text('60\n')
    for path in temporary.iterdir():
        path.chmod(0o444)
    temporary.rename(DEST)
    write(RUN / 'gather-complete.json', provenance)
    print(json.dumps({'event': 'complete', 'destination': str(DEST), 'elapsed_seconds': time.time() - began}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        RUN.mkdir(parents=True, exist_ok=True)
        write(RUN / 'gather-error.json', {'type': type(exc).__name__, 'error': str(exc), 'time': time.time()})
        raise
