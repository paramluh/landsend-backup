import json,time,hashlib
from pathlib import Path
from huggingface_hub import HfApi,CommitOperationAdd
ROOT=Path('/home/param/outputs/landsend-backup-20261031/remote-datasets');REPO='InternalCan/landsend-artifacts-backup-param';proof=json.loads((ROOT/'VERIFIED_REMOTE_DATASETS.json').read_text());assert proof['state']=='complete' and not proof['errors']
names=['VERIFIED_REMOTE_DATASETS.json','README.md','restore_remote_dataset.py','node0-derived-archive-proof.json','catalog-uploaded.json','upload-plan.json','restore-test.json','remote-download-restore-test.json']
api=HfApi();result=api.create_commit(repo_id=REPO,repo_type='dataset',operations=[CommitOperationAdd(path_in_repo='datasets-audit/'+name,path_or_fileobj=str(ROOT/name)) for name in names],commit_message='Verify all remote dataset copies and preserve unique data with restore tools')
remote={r.path:r for r in api.get_paths_info(REPO,['datasets-audit/'+name for name in names],repo_type='dataset',revision=result.oid)};receipts=[]
for name in names:
 p=ROOT/name;data=p.read_bytes();r=remote['datasets-audit/'+name];assert r.size==len(data);sha=hashlib.sha256(data).hexdigest();remote_sha=getattr(r.lfs,'sha256',None) if r.lfs else None
 if isinstance(r.lfs,dict):remote_sha=r.lfs.get('sha256')
 if remote_sha:assert remote_sha==sha
 else:assert r.blob_id==hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
 receipts.append({'path':'datasets-audit/'+name,'bytes':len(data),'sha256':sha})
receipt={'state':'verified','repo':REPO,'revision':result.oid,'files':receipts,'all_dataset_writes_settled':True,'completed_unix':time.time()};(ROOT/'FINAL_UPLOAD.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
