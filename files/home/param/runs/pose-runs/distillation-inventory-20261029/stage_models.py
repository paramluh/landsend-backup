#!/usr/bin/env python3
"""Stage pinned production exports, verifying every file against HF hashes."""
import concurrent.futures, fcntl, hashlib, json, os, pathlib, shutil, socket, time
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
os.environ.setdefault('HF_XET_CHUNK_CACHE_SIZE_BYTES', '0')
os.environ.setdefault('HF_XET_HIGH_PERFORMANCE', '1')
os.environ.setdefault('HF_XET_NUM_CONCURRENT_RANGE_GETS', '8')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '120')
from huggingface_hub import hf_hub_download
CONTROL = pathlib.Path(__file__).resolve().parent
MANIFEST = json.loads((CONTROL / 'model-manifest.json').read_text())
HOST = socket.gethostname().split('.')[0]
STATEPATH = CONTROL / ('staging-' + HOST + '.json')
lock = (CONTROL / 'staging.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
def now(): return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
def atomic(p, obj):
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    tmp.replace(p)
def verify(path, entry):
    if not path.is_file() or path.stat().st_size != entry['size']: return False
    h = hashlib.sha256() if entry.get('sha256') else hashlib.sha1()
    if not entry.get('sha256'): h.update(f"blob {entry['size']}\0".encode())
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''): h.update(chunk)
    return h.hexdigest() == (entry.get('sha256') or entry['git_blob_sha1'])
def fetch(model, entry):
    root = pathlib.Path(model['local_root']); path = root / entry['path']
    if verify(path, entry): return 'existing'
    # Preserve any pre-existing file with mismatching content before replacement.
    if path.exists():
        backup = path.with_name(path.name + '.before-staging-' + str(time.time_ns()))
        path.rename(backup)
    for attempt in range(5):
        try:
            hf_hub_download(repo_id=model['repo'], revision=model['revision'],
                filename=entry['path'], local_dir=root, force_download=attempt > 0)
            if not verify(path, entry): raise ValueError('Repository checksum mismatch')
            return 'downloaded'
        except Exception as exc:
            print(json.dumps({'event':'retry','model':model['key'],'path':entry['path'],
                  'attempt':attempt+1,'error_type':type(exc).__name__}), flush=True)
            if attempt == 4: raise RuntimeError(type(exc).__name__) from None
            time.sleep(min(20, 2**attempt))
models = MANIFEST['models']
entries = [(m, f) for m in models for f in m['files']]
missing_bytes = sum(e['size'] for m,e in entries if not (pathlib.Path(m['local_root'])/e['path']).is_file())
assert shutil.disk_usage('/home/param').free > missing_bytes + 80 * 10**9, 'Insufficient disk space'
state = {'host':HOST,'pid':os.getpid(),'started_utc':now(),'state':'running',
    'files_total':len(entries),'bytes_total':sum(e['size'] for _,e in entries),
    'files_verified':0,'bytes_verified':0,'existing_verified':0,'downloaded_verified':0,'failures':[],
    'models': {m['key']:{'repo':m['repo'],'revision':m['revision'],'root':m['local_root'],
        'files_total':len(m['files']),'files_verified':0,'bytes_verified':0} for m in models}}
def report():
    state['updated_utc']=now(); atomic(STATEPATH,state)
    print(json.dumps({k:state[k] for k in ('host','state','files_verified','files_total','bytes_verified','bytes_total','updated_utc')}),flush=True)
report()
# Small metadata first, then weights. Eight simultaneous files bound CPU/disk/network use.
entries.sort(key=lambda pair: (pair[1]['size'] > 1024*1024, pair[0]['key'], pair[1]['path']))
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    pending = {pool.submit(fetch,m,e):(m,e) for m,e in entries}
    last = time.monotonic()
    while pending:
        done,_=concurrent.futures.wait(pending, timeout=10, return_when=concurrent.futures.FIRST_COMPLETED)
        for future in done:
            m,e=pending.pop(future)
            try: origin=future.result()
            except Exception as exc:
                state['failures'].append({'model':m['key'],'path':e['path'],'error_type':str(exc)})
            else:
                state['files_verified']+=1; state['bytes_verified']+=e['size']; state[origin+'_verified']+=1
                state['models'][m['key']]['files_verified']+=1
                state['models'][m['key']]['bytes_verified']+=e['size']
        if time.monotonic()-last >= 10 or not pending: report(); last=time.monotonic()
state['state']='failed' if state['failures'] else 'complete'; state['finished_utc']=now(); report()
for model in models:
    status=state['models'][model['key']]
    if status['files_verified']==status['files_total']:
        atomic(pathlib.Path(model['local_root'])/'PRODUCTION_VERIFIED.json',
               {**status,'host':HOST,'verified_utc':now(),'verification':'HF LFS SHA-256 or Git blob SHA-1, every export file'})
raise SystemExit(bool(state['failures']))
