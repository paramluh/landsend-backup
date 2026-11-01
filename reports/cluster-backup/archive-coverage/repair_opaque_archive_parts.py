"""Replace credential-bearing nested archives in completed private HF artifact parts."""
import hashlib,io,json,os,socket,subprocess,sys,tarfile,time
from pathlib import Path
BASE=Path('/home/param/outputs/landsend-backup-20261031');sys.path.insert(0,str(BASE/'orphans'))
import orphan_backup as B
from huggingface_hub import HfApi
OUT=BASE/'archive-coverage';GROUP=B.BASE/'archives/artifacts'
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def upload(api,path,remote):
 for attempt in range(8):
  try:return api.upload_file(path_or_fileobj=str(path),path_in_repo=remote,repo_id=B.HF_REPO,repo_type='dataset',commit_message=f'{socket.gethostname()}: sanitize embedded source archive')
  except Exception:
   if attempt==7:raise
   time.sleep(min(60,5*(attempt+1)))
def verify(api,path,remote,revision):
 info=api.get_paths_info(B.HF_REPO,[remote],repo_type='dataset',revision=revision)[0]
 assert info.size==path.stat().st_size,remote
 if info.lfs:
  digest=info.lfs.get('sha256') if isinstance(info.lfs,dict) else info.lfs.sha256
  assert digest==sha(path),remote
 else:
  data=path.read_bytes();assert info.blob_id==hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest(),remote

def main():
 assert hasattr(B,'sanitize_container_file'),'Updated recursive sanitizer is not deployed'
 assert (GROUP/'COMPLETE.json').exists(),'Initial artifact group must be complete before repair'
 audit=json.loads((OUT/'opaque-archives-audit.json').read_text())
 flagged={r['path']:r for r in audit['archives'] if r['flagged_members']}
 assert flagged,'No flagged archive inputs'
 # The initial worker must have stopped so its in-memory receipts cannot overwrite repairs.
 for p in Path('/proc').iterdir():
  try:
   if p.name.isdigit() and p.stat().st_uid==os.getuid():assert not any(Path(a).name=='remote_orphan_worker.py' for a in p.joinpath('cmdline').read_bytes().decode().split('\0')),'Initial artifact worker is still live'
  except OSError:pass
 api=HfApi();assert api.repo_info(B.HF_REPO,repo_type='dataset').private
 replacements={};proof=[]
 for source,a in flagged.items():
  handle,redactions,removed=B.sanitize_container_file(Path(source));handle.seek(0)
  clean=OUT/('sanitized-'+a['sha256']+'.tar.gz')
  with clean.open('wb') as out:
   import shutil
   shutil.copyfileobj(handle,out,4*1024**2)
  handle.close()
  # Independently inspect the resulting inner tar without exposing member data.
  matches=0
  with tarfile.open(clean,'r:*') as archive:
   for member in archive:
    if member.isfile():
     with archive.extractfile(member) as f:matches+=B.sanitize(f.read())[1]
  assert matches==0,'Sanitized container still matches a credential pattern'
  replacements[source.lstrip('/')]={'path':clean,'redactions':redactions,'removed_members':removed,'original_sha256':a['sha256'],'sanitized_sha256':sha(clean)}
 for manifest in GROUP.glob('part-*.files.json'):
  rows=json.loads(manifest.read_text());affected=[r for r in rows if r['path'] in flagged]
  if not affected:continue
  part=manifest.with_name(manifest.name.removesuffix('.files.json'));old_hash=sha(part)
  meta_path=part.with_name(part.name+'.json');receipt_path=part.with_name(part.name+'.uploaded.json')
  meta=json.loads(meta_path.read_text());old_receipt=json.loads(receipt_path.read_text())
  temp=part.with_name(part.name+'.sanitizing');replaced=[]
  with temp.open('wb') as out:
   decomp=subprocess.Popen(['zstd','-q','-d','-c',str(part)],stdout=subprocess.PIPE)
   comp=subprocess.Popen(['zstd','-q','-1','-T2','-c'],stdin=subprocess.PIPE,stdout=out)
   with tarfile.open(fileobj=decomp.stdout,mode='r|') as source,tarfile.open(fileobj=comp.stdin,mode='w|') as target:
    for member in source:
     key=member.name.lstrip('/')
     if key in replacements and member.isfile():
      replacement=replacements[key]
      with source.extractfile(member) as f:assert hashlib.file_digest(f,'sha256').hexdigest()==replacement['original_sha256']
      member.size=replacement['path'].stat().st_size
      with replacement['path'].open('rb') as f:target.addfile(member,f)
      replaced.append(key)
     elif member.isfile():
      with source.extractfile(member) as f:target.addfile(member,f)
     else:target.addfile(member)
   comp.stdin.close();decomp.stdout.close();assert decomp.wait()==0;assert comp.wait()==0
  assert set(replaced)=={r['archive_path'] for r in affected}
  os.replace(part,part.with_name(part.name+'.unsafe-original'));os.replace(temp,part)
  for row in affected:
   replacement=replacements[row['archive_path']]
   row.update(archived_bytes=replacement['path'].stat().st_size,nested_archive_redactions=replacement['redactions'],nested_removed_members=replacement['removed_members'],original_nested_sha256=replacement['original_sha256'],sanitized_nested_sha256=replacement['sanitized_sha256'])
  B.atomic_json(manifest,rows)
  meta.update(bytes=part.stat().st_size,sha256=sha(part),sanitized_unix=time.time(),superseded_sha256=old_hash)
  B.atomic_json(meta_path,meta)
  prefix=f'{socket.gethostname()}/artifacts';commit=upload(api,part,f'{prefix}/{part.name}');verify(api,part,f'{prefix}/{part.name}',commit.oid)
  receipt={**meta,'state':'verified','revision':commit.oid,'hf_path':f'{prefix}/{part.name}','superseded_revision':old_receipt['revision']};B.atomic_json(receipt_path,receipt)
  manifest_commit=upload(api,manifest,f'{prefix}/manifests/{manifest.name}');verify(api,manifest,f'{prefix}/manifests/{manifest.name}',manifest_commit.oid)
  complete=json.loads((GROUP/'COMPLETE.json').read_text());complete['archives']=[receipt if r['archive']==part.name else r for r in complete['archives']];complete['sanitized_unix']=time.time();B.atomic_json(GROUP/'COMPLETE.json',complete)
  complete_commit=upload(api,GROUP/'COMPLETE.json',f'{prefix}/COMPLETE.json');verify(api,GROUP/'COMPLETE.json',f'{prefix}/COMPLETE.json',complete_commit.oid)
  proof.append({'part':part.name,'sources':replaced,'state':'verified','old_sha256':old_hash,'sha256':meta['sha256'],'revision':complete_commit.oid,'nested_credential_matches_after':0})
 assert sum(len(x['sources']) for x in proof)==len(flagged),'Not all flagged archives were repaired'
 result={'state':'current_head_sanitized_history_scrub_required','host':socket.gethostname(),'hf_repo':B.HF_REPO,'parts':proof,'time':time.time()}
 B.atomic_json(OUT/'OPAQUE_ARCHIVE_REPAIR.json',result)
 commit=upload(api,OUT/'OPAQUE_ARCHIVE_REPAIR.json',f'{socket.gethostname()}/artifacts/metadata/OPAQUE_ARCHIVE_REPAIR.json');verify(api,OUT/'OPAQUE_ARCHIVE_REPAIR.json',f'{socket.gethostname()}/artifacts/metadata/OPAQUE_ARCHIVE_REPAIR.json',commit.oid)
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
