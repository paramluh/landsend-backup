"""Replicate the verified Zoom text snapshot to the configured Pose workers."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import uuid

AUDIT = Path(__file__).resolve().parent
SOURCE = Path('/home/param/datasets/zoom-strangers-5k')
HOSTS = ('landsend-1', 'landsend-2', 'landsend-3', 'landsend-4', 'landsend-5', 'landsend-12')
SSH = ['ssh', '-i', '/home/param/.ssh/id_ed25519_cluster', '-o', 'BatchMode=yes',
       '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=10']
MANIFEST_SHA = hashlib.sha256((SOURCE / 'SOURCE_MANIFEST.json').read_bytes()).hexdigest()

VERIFY = r'''
from pathlib import Path
import datetime, hashlib, json, socket, sys
stage, target, expected_host, expected_manifest_sha = sys.argv[1:]
stage, target = Path(stage), Path(target)
host = socket.gethostname().split('.')[0]
assert host == expected_host, (host, expected_host)
manifest_bytes = (stage / 'SOURCE_MANIFEST.json').read_bytes()
assert hashlib.sha256(manifest_bytes).hexdigest() == expected_manifest_sha
manifest = json.loads(manifest_bytes)
for item in manifest['files']:
    relative = Path(item['path'])
    assert not relative.is_absolute() and '..' not in relative.parts
    path = stage / relative
    assert path.is_file() and not path.is_symlink()
    assert path.stat().st_size == item['bytes'], item['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], item['path']
counts = {}
seen = set()
for split, expected in manifest['splits'].items():
    count = 0
    with (stage / (split + '.jsonl')).open() as stream:
        for line in stream:
            row = json.loads(line)
            key = json.dumps(row['id'], sort_keys=True)
            assert key not in seen
            seen.add(key)
            messages = row['messages']
            assert isinstance(messages, list) and messages
            assert messages[0]['role'] == 'user' and messages[-1]['role'] == 'assistant'
            assert all(message['role'] in ('user', 'assistant')
                       and isinstance(message['content'], str) and message['content'].strip()
                       for message in messages)
            assert isinstance(row.get('meta'), dict)
            count += 1
    assert count == expected, (split, count, expected)
    counts[split] = count
ready = {'raw_dataset_ready': True, 'hostname': host, 'path': str(target),
         'revision': manifest['revision'], 'splits': counts, 'bytes': manifest['bytes'],
         'source_manifest_sha256': expected_manifest_sha, 'all_file_hashes_verified': True,
         'verified_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'derived_zoom_table_present': False,
         'note': 'Verified raw text splits; renderer-specific zoom_table.npz is not generated.'}
if stage != target:
    assert not target.exists() and not target.is_symlink(), str(target)
(stage / 'DATASET_READY.json').write_text(json.dumps(ready, indent=2) + '\n')
if stage != target:
    stage.rename(target)
print(json.dumps(ready))
'''


def remote(host, arguments, *, input_text=None, timeout=300):
    return subprocess.run(SSH + [host, shlex.join(map(str, arguments))], input=input_text,
                          text=True, capture_output=True, timeout=timeout, check=True)


def deploy(host):
    staging = SOURCE.parent / ('.zoom-strangers-5k-sync-' + uuid.uuid4().hex)
    try:
        remote(host, ['python3', '-c',
                      'from pathlib import Path; import sys; p=Path(sys.argv[1]); '
                      't=Path(sys.argv[2]); assert not t.exists() and not t.is_symlink(); '
                      'p.mkdir(mode=0o700)', staging, SOURCE])
        copied = subprocess.run(['rsync', '-a', '--exclude=.cache/', '--exclude=DATASET_READY.json',
                                 '-e', shlex.join(SSH), str(SOURCE) + '/',
                                 f'{host}:{staging}/'], text=True, capture_output=True,
                                timeout=300, check=True)
        (AUDIT / f'{host}-transfer.log').write_text(copied.stdout + copied.stderr)
        checked = remote(host, ['python3', '-', staging, SOURCE, host, MANIFEST_SHA], input_text=VERIFY)
        result = json.loads(checked.stdout)
        (AUDIT / f'{host}-verified.json').write_text(json.dumps(result, indent=2) + '\n')
        return result
    except Exception as exc:
        result = {'hostname': host, 'raw_dataset_ready': False, 'error': str(exc),
                  'staging_directory': str(staging)}
        if isinstance(exc, subprocess.CalledProcessError):
            result['stderr'] = exc.stderr
        (AUDIT / f'{host}-error.json').write_text(json.dumps(result, indent=2) + '\n')
        return result


local = subprocess.run(['python3', '-', str(SOURCE), str(SOURCE), 'landsend-0', MANIFEST_SHA],
                       input=VERIFY, text=True, capture_output=True, check=True)
results = [json.loads(local.stdout)]
(AUDIT / 'landsend-0-verified.json').write_text(json.dumps(results[0], indent=2) + '\n')
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(deploy, host) for host in HOSTS]
    for future in concurrent.futures.as_completed(futures):
        result = future.result()
        results.append(result)
        print(json.dumps({'host': result['hostname'], 'verified': result['raw_dataset_ready'],
                          'splits': result.get('splits'), 'error': result.get('error')}), flush=True)
complete = all(result['raw_dataset_ready'] for result in results) and len(results) == 7
summary = {'complete': complete, 'repository': 'InternalCan/zoom-strangers-5k',
           'path_on_every_node': str(SOURCE), 'source_manifest_sha256': MANIFEST_SHA,
           'nodes': sorted(results, key=lambda result: result['hostname'])}
(AUDIT / 'REPLICATION_COMPLETE.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({'complete': complete, 'nodes_verified': sum(r['raw_dataset_ready'] for r in results),
                  'report': str(AUDIT / 'REPLICATION_COMPLETE.json')}), flush=True)
raise SystemExit(0 if complete else 1)
