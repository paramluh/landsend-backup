#!/usr/bin/env python3
"""Prove local dataset files are already retrievable at pinned HF revisions."""
import concurrent.futures,hashlib,json,time
from pathlib import Path
from huggingface_hub import HfApi
import orphan_backup as B

ROOT=Path('/home/param/datasets')
jobs=[]
for manifest in (ROOT/'aloe-duplex/SOURCE_MANIFEST.json',ROOT/'aloe-duplex/pose/SOURCE_MANIFEST.json'):
    for f in json.loads(manifest.read_text())['files']:
        jobs.append({'local':str(manifest.parent/f['directory']/f['path']),'repo':f['repo'],'revision':f['revision'],'path':f['path'],'bytes':f['size'],'sha256':f.get('sha256'),'blob':f.get('git_blob_sha1')})
manifest=ROOT/'blackband-repro-20261031/source-files.json';x=json.loads(manifest.read_text())
for f in x['files']:
    jobs.append({'local':str(manifest.parent/'source'/f['path']),'repo':'InternalCan/stage2-handover-pose-mimi-sam-v1','revision':x['revision'],'path':f['path'],'bytes':f['size'],'sha256':f['sha256']})
manifest=ROOT/'xabi-v2pure-cache/provenance/source-cache-manifest.json';x=json.loads(manifest.read_text())
for f in x['files']:
    jobs.append({'local':str(ROOT/'xabi-v2pure-cache/cache'/f['path']),'repo':'InternalCan/xabi-v2pure-production-cache','revision':'75a43997f9775202a479287b85b9e04be7a9a0e6','path':'cache/'+f['path'],'bytes':f['bytes'],'sha256':f['sha256']})
manifest=ROOT/'zoom-strangers-5k/SOURCE_MANIFEST.json';x=json.loads(manifest.read_text())
for f in x['files']:
    jobs.append({'local':str(manifest.parent/f['path']),'repo':x['repository'],'revision':x['revision'],'path':f['path'],'bytes':f['bytes'],'sha256':f['sha256'],'blob':f['hf_blob_id']})
api=HfApi();remote={};failures=[]
for repo,rev in sorted({(j['repo'],j['revision']) for j in jobs}):
    try:
        info=api.repo_info(repo,repo_type='dataset',revision=rev,files_metadata=True)
        remote[(repo,rev)]={f.rfilename:f for f in info.siblings}
        print(json.dumps({'remote':repo,'revision':info.sha,'files':len(info.siblings)}),flush=True)
    except Exception as exc:
        failures.append({'repo':repo,'revision':rev,'error':str(exc)})

def verify(j):
    try:
        p=Path(j['local']);f=remote[(j['repo'],j['revision'])][j['path']]
        if p.stat().st_size!=j['bytes'] or f.size!=j['bytes']:raise ValueError('local/remote size mismatch')
        sha=getattr(f.lfs,'sha256',None) if f.lfs else None
        if isinstance(f.lfs,dict):sha=f.lfs.get('sha256')
        if sha and j.get('sha256') and sha!=j['sha256']:raise ValueError('manifest/remote SHA256 mismatch')
        if not sha and j.get('blob') and f.blob_id!=j['blob']:raise ValueError('manifest/remote Git blob mismatch')
        if sha:
            with p.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
            if actual!=sha:raise ValueError('local/remote SHA256 mismatch')
        else:
            data=p.read_bytes();actual=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
            if actual!=f.blob_id:raise ValueError('local/remote Git blob mismatch')
        return {**j,'verified':True,'remote_sha256':sha,'remote_blob':f.blob_id}
    except Exception as exc:return {**j,'verified':False,'error':str(exc)}

verified=[];started=time.time();size=0
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    for i,result in enumerate(pool.map(verify,jobs),1):
        if result['verified']:verified.append(result);size+=result['bytes']
        else:failures.append(result)
        if i%500==0:print(json.dumps({'processed':i,'total':len(jobs),'verified':len(verified),'GiB':size/2**30,'seconds':time.time()-started}),flush=True)
B.atomic_json(B.BASE/'verified-existing-dataset-paths.json',[r['local'] for r in verified])
B.atomic_json(B.BASE/'verified-existing-datasets.json',{'verified':verified,'failures':failures,'verified_bytes':size,'seconds':time.time()-started})
print(json.dumps({'state':'finished','verified':len(verified),'failures':len(failures),'GiB':size/2**30,'seconds':time.time()-started}),flush=True)
