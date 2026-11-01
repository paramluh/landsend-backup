"""Upload only globally deduplicated remote dataset contents plus exact restore evidence."""
import sys,json,socket,hashlib,time
from pathlib import Path
BASE=Path('/home/param/outputs/landsend-backup-20261031/remote-datasets')
sys.path.insert(0,str(BASE.parent/'orphans'))
import orphan_backup as B
from huggingface_hub import HfApi
B.BASE=BASE/'upload';B.BASE.mkdir(parents=True,exist_ok=True)
B.HF_REPO='InternalCan/landsend-artifacts-backup-param'
rows=json.loads((BASE/'upload-selected.json').read_text())
for row in rows:
 p=Path(row['path']);s=p.stat()
 if (s.st_size,s.st_mtime_ns)!=(row['bytes'],row['mtime_ns']):raise RuntimeError('Dataset changed since verification: '+str(p))
B.pack(rows,'datasets')
archive_status=B.BASE/'archives/datasets/COMPLETE.json'
if not archive_status.exists():raise RuntimeError('Dataset archive upload incomplete')
archive_result=json.loads(archive_status.read_text())
if archive_result.get('state')!='complete' or archive_result.get('errors'):raise RuntimeError('Dataset archive upload reported errors')
api=HfApi();host=socket.gethostname().split('.')[0];receipts=[]
for name in ['SUMMARY.json','verification.json','upload-required.json','upload-selected.json','cross-node-aliases.json']:
 p=BASE/name
 if not p.exists():continue
 commit=api.upload_file(path_or_fileobj=str(p),path_in_repo=host+'/datasets/verification/'+name,repo_id=B.HF_REPO,repo_type='dataset',commit_message='Verified remote dataset evidence: '+host)
 remote=api.get_paths_info(B.HF_REPO,[host+'/datasets/verification/'+name],repo_type='dataset',revision=commit.oid)[0]
 assert remote.size==p.stat().st_size
 digest=hashlib.file_digest(p.open('rb'),'sha256').hexdigest();sha=getattr(remote.lfs,'sha256',None) if remote.lfs else None
 if isinstance(remote.lfs,dict):sha=remote.lfs.get('sha256')
 if sha:assert sha==digest
 else:
  data=p.read_bytes();assert remote.blob_id==hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
 receipts.append({'file':name,'revision':commit.oid,'bytes':p.stat().st_size,'sha256':digest})
complete={'state':'complete','host':host,'hf_repo':B.HF_REPO,'unique_files_uploaded':len(rows),'unique_bytes_uploaded':sum(r['bytes'] for r in rows),'metadata':receipts,'completed_unix':time.time()}
B.atomic_json(BASE/'COMPLETE.json',complete)
api.upload_file(path_or_fileobj=str(BASE/'COMPLETE.json'),path_in_repo=host+'/datasets/verification/COMPLETE.json',repo_id=B.HF_REPO,repo_type='dataset')
print(json.dumps(complete),flush=True)
