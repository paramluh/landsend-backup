#!/usr/bin/env python3
"""Check initial archive writers are stopped and verify the current private HF tree."""
import concurrent.futures,hashlib,json,shlex,socket,subprocess,time
from pathlib import Path
import orphan_backup as B

COLLECT=r'''
import pathlib,json,socket,hashlib
b=pathlib.Path('/home/param/outputs/landsend-backup-20261031/orphans');workers=[];groups=[]
for p in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
 try:a=[x.decode(errors='replace') for x in p.read_bytes().split(b'\0') if x]
 except OSError:continue
 if a and 'python' in a[0] and any(pathlib.Path(x).name in {'remote_orphan_worker.py','backup_source_archives.py','upload_remainder.py','coordinate_uploads.py'} for x in a[1:]):workers.append({'pid':int(p.parent.name),'script':[x for x in a[1:] if x.endswith('.py')]})
for group in ('artifacts','datasets','source-archives'):
 out=b/'archives'/group
 if not out.exists():continue
 marker=out/'COMPLETE.json'
 if not marker.exists():marker=out/'PARTIAL.json'
 if not marker.exists():groups.append({'group':group,'state':'in_progress','parts':len(list(out.glob('*.uploaded.json')))});continue
 result=json.loads(marker.read_text());files=[]
 for f in [marker,out/'aliases.json',*sorted(out.glob('*.files.json'))]:
  if not f.exists():continue
  data=f.read_bytes();relative=group+'/'+(('manifests/' if f.name=='aliases.json' or f.name.endswith('.files.json') else '')+f.name)
  files.append({'path':socket.gethostname()+'/'+relative,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'blob':hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()})
 groups.append({'group':group,'state':result['state'],'errors':result.get('errors',0),'archives':result['archives'],'files':files})
print(json.dumps({'host':socket.gethostname(),'workers':workers,'groups':groups}))
'''
def collect(node):
    if node==0:cmd=['/home/param/pose-repo/.venv/bin/python','-c',COLLECT]
    else:cmd=['ssh','-i','/home/param/.ssh/id_ed25519_cluster','-o','BatchMode=yes','-o','ConnectTimeout=10',f'landsend-{node}','/home/param/pose-repo/.venv/bin/python','-c',shlex.quote(COLLECT)]
    p=subprocess.run(cmd,text=True,capture_output=True,timeout=60)
    if p.returncode:raise RuntimeError(f'node{node}: '+p.stderr[:300])
    return json.loads(p.stdout)

def main():
    nodes=list(concurrent.futures.ThreadPoolExecutor(max_workers=7).map(collect,[0,1,2,3,4,5,12]))
    result={'checked_unix':time.time(),'nodes':nodes,'hf_repo':B.HF_REPO}
    if any(n['workers'] or any(g['state']=='in_progress' for g in n['groups']) for n in nodes):
        result['state']='writers_running';B.atomic_json(B.BASE.parent/'INITIAL_ARTIFACT_VERIFICATION.json',result)
        print(json.dumps({'state':result['state'],'workers':{n['host']:n['workers'] for n in nodes if n['workers']}}));return
    from huggingface_hub import HfApi
    api=HfApi();info=api.repo_info(B.HF_REPO,repo_type='dataset');assert info.private;revision=info.sha
    expected={}
    for node in nodes:
        for group in node['groups']:
            for a in group['archives']:expected[a['hf_path']]={'bytes':a['bytes'],'sha256':a['sha256'],'archive':True}
            for f in group['files']:expected[f['path']]=f
    verified=[];failures=[];paths=sorted(expected)
    for start in range(0,len(paths),100):
        batch=paths[start:start+100];remote=api.get_paths_info(B.HF_REPO,batch,repo_type='dataset',revision=revision);remote={r.path:r for r in remote}
        for path in batch:
            local=expected[path];r=remote.get(path)
            if r is None:failures.append({'path':path,'reason':'missing'});continue
            sha=(r.lfs.get('sha256') if isinstance(r.lfs,dict) else getattr(r.lfs,'sha256',None)) if r.lfs else None
            ok=r.size==local['bytes'] and ((sha==local['sha256']) if sha else (r.blob_id==local.get('blob')))
            if not ok:failures.append({'path':path,'reason':'size_or_hash_mismatch'})
            else:verified.append({'path':path,'bytes':r.size,'sha256':sha,'blob':r.blob_id,'archive':local.get('archive',False)})
    result.update(state='verified' if not failures else 'verification_failed',revision=revision,verified=verified,failures=failures,archive_bytes=sum(r['bytes'] for r in verified if r['archive']))
    B.atomic_json(B.BASE.parent/'INITIAL_ARTIFACT_VERIFICATION.json',result)
    print(json.dumps({k:result[k] for k in ('state','revision','archive_bytes','failures')}))
    if failures:raise SystemExit(1)

if __name__=='__main__':main()
