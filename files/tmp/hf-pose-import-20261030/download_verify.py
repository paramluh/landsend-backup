import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
from huggingface_hub import HfApi, snapshot_download

repo, revision = sys.argv[1:]
dest = Path('/home/param/models/imported-weights') / repo.split('/')[-1]
info = HfApi().model_info(repo, revision=revision, files_metadata=True)
files = info.siblings
total_bytes = sum(f.size for f in files)
print(json.dumps({'event':'start','repo':repo,'revision':info.sha,'destination':str(dest),'files':len(files),'bytes':total_bytes}), flush=True)
dest.mkdir(parents=True, exist_ok=True)
manifest = {'repo_id':repo,'revision':info.sha,'total_bytes':total_bytes,'files':[{'path':f.rfilename,'size':f.size,'sha256':f.lfs.sha256 if f.lfs else None,'git_blob_sha1':f.blob_id} for f in files]}
manifest_path = dest / '.cache/huggingface/import-manifest.json'
manifest_path.parent.mkdir(parents=True, exist_ok=True)
manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
snapshot_download(repo_id=repo, revision=info.sha, local_dir=dest, max_workers=8)
print(json.dumps({'event':'download_complete','repo':repo}), flush=True)

def verify(item):
    p = dest / item['path']
    if not p.is_file() or p.stat().st_size != item['size']:
        raise RuntimeError('Missing file or wrong size: '+item['path'])
    h = hashlib.sha256() if item['sha256'] else hashlib.sha1()
    if not item['sha256']:
        h.update(b'blob '+str(item['size']).encode()+b'\0')
    with p.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):
            h.update(chunk)
    expected = item['sha256'] or item['git_blob_sha1']
    if h.hexdigest() != expected:
        raise RuntimeError('Checksum mismatch: '+item['path'])
    return item['size']

verified_bytes = 0
last_report = time.monotonic()
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    tasks = [pool.submit(verify,item) for item in manifest['files']]
    for index, future in enumerate(concurrent.futures.as_completed(tasks),1):
        verified_bytes += future.result()
        if time.monotonic()-last_report >= 30 or index == len(tasks):
            print(json.dumps({'event':'verify_progress','files':index,'total_files':len(tasks),'bytes':verified_bytes,'total_bytes':total_bytes}), flush=True)
            last_report = time.monotonic()
manifest['verification'] = {'status':'passed','verified_files':len(files),'verified_bytes':verified_bytes,'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'method':'HF LFS SHA-256 or Git blob SHA-1'}
manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
print(json.dumps({'event':'verified','repo':repo,'destination':str(dest),'files':len(files),'bytes':verified_bytes}), flush=True)
