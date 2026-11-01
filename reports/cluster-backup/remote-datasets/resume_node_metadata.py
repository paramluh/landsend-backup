"""Finish metadata after all archive bytes were verified; retry transient HF commit failures."""
import json,hashlib,time,socket,os
from pathlib import Path
from huggingface_hub import HfApi,CommitOperationAdd
BASE=Path('/home/param/outputs/landsend-backup-20261031/remote-datasets');ROOT=BASE/'upload/archives/datasets';HOST=socket.gethostname().split('.')[0];REPO='InternalCan/landsend-artifacts-backup-param';api=HfApi()
def retry(fn):
 for attempt in range(1,11):
  try:return fn()
  except Exception as exc:
   print(json.dumps({'attempt':attempt,'error_type':type(exc).__name__}),flush=True)
   if attempt==10:raise
   time.sleep(min(30,attempt*5))
rows=json.loads((BASE/'upload-selected.json').read_text());receipts=[json.loads(p.read_text()) for p in sorted(ROOT.glob('*.uploaded.json'))];archives=list(ROOT.glob('*.tar.zst'));assert len(receipts)==len(archives) and all(r['state']=='verified' and not r['errors'] for r in receipts)
manifest=[]
for p in ROOT.glob('*.files.json'):manifest.extend(json.loads(p.read_text()))
assert len(manifest)==len(rows) and not any(r.get('error') for r in manifest)
status={'state':'complete','errors':0,'hf_repo':REPO,'group':'datasets','archives':receipts,'files':len(rows),'aliases':len(json.loads((ROOT/'aliases.json').read_text())),'completed_unix':time.time()};(ROOT/'COMPLETE.json').write_text(json.dumps(status,indent=2)+'\n')
files={HOST+'/datasets/manifests/'+p.name:p for p in [ROOT/'aliases.json',*sorted(ROOT.glob('*.files.json'))]};files[HOST+'/datasets/COMPLETE.json']=ROOT/'COMPLETE.json'
names=['SUMMARY.json','verification.json','upload-required.json','upload-selected.json','cross-node-aliases.json']
for name in names:files[HOST+'/datasets/verification/'+name]=BASE/name
commit=retry(lambda:api.create_commit(repo_id=REPO,repo_type='dataset',operations=[CommitOperationAdd(path_in_repo=k,path_or_fileobj=str(v)) for k,v in files.items()],commit_message='Complete verified dataset archives and restore manifests for '+HOST))
remote={r.path:r for r in retry(lambda:api.get_paths_info(REPO,list(files),repo_type='dataset',revision=commit.oid))}
for name,p in files.items():
 data=p.read_bytes();r=remote[name];assert r.size==len(data);sha=getattr(r.lfs,'sha256',None) if r.lfs else None
 if isinstance(r.lfs,dict):sha=r.lfs.get('sha256')
 if sha:assert sha==hashlib.sha256(data).hexdigest()
 else:assert r.blob_id==hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
complete={'state':'complete','host':HOST,'hf_repo':REPO,'unique_files_uploaded':len(rows),'unique_bytes_uploaded':sum(r['bytes'] for r in rows),'metadata':[{'file':name,'revision':commit.oid,'bytes':(BASE/name).stat().st_size,'sha256':hashlib.file_digest((BASE/name).open('rb'),'sha256').hexdigest()} for name in names],'completed_unix':time.time()};p=BASE/'COMPLETE.pending.json';p.write_text(json.dumps(complete,indent=2)+'\n');remote_path=HOST+'/datasets/verification/COMPLETE.json';end=retry(lambda:api.upload_file(path_or_fileobj=str(p),path_in_repo=remote_path,repo_id=REPO,repo_type='dataset'));r=retry(lambda:api.get_paths_info(REPO,[remote_path],repo_type='dataset',revision=end.oid))[0];data=p.read_bytes();assert r.size==len(data) and r.blob_id==hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest();os.replace(p,BASE/'COMPLETE.json');print(json.dumps(complete),flush=True)
