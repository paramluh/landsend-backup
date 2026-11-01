"""Wait for pending sanitized source containers, then verify uploaded bytes and contents."""
import hashlib,io,json,subprocess,sys,tarfile,time,traceback
from pathlib import Path
BASE=Path('/home/param/outputs/landsend-backup-20261031');OUT=BASE/'archive-coverage';sys.path.insert(0,str(BASE/'orphans'))
import orphan_backup as B
from huggingface_hub import HfApi

def main():
 audit=json.loads((OUT/'opaque-archives-audit.json').read_text());wanted={x['path']:x for x in audit['archives'] if x['flagged_members']};assert len(wanted)==3
 api=HfApi();verified={};deadline=1793515500.0 # 06:45 UTC, leaves time for root history scrub
 while len(verified)<len(wanted) and time.time()<deadline:
  for manifest in (B.BASE/'archives/artifacts').glob('part-*.files.json'):
   rows=json.loads(manifest.read_text());matches=[x for x in rows if x['path'] in wanted and x['path'] not in verified]
   if not matches:continue
   part=manifest.with_name(manifest.name.removesuffix('.files.json'));receipt_path=part.with_name(part.name+'.uploaded.json')
   if not receipt_path.exists():continue
   receipt=json.loads(receipt_path.read_text());assert receipt['state']=='verified'
   for row in matches:assert row.get('redactions',0)>0 and not row.get('error'),f'Missing nested credential redaction: {row["path"]}'
   info=api.repo_info(B.HF_REPO,repo_type='dataset');assert info.private
   remote=api.get_paths_info(B.HF_REPO,[receipt['hf_path']],repo_type='dataset',revision=info.sha)[0]
   sha=remote.lfs.get('sha256') if isinstance(remote.lfs,dict) else remote.lfs.sha256
   assert remote.size==receipt['bytes'] and sha==receipt['sha256'],'Remote archive differs from local verified receipt'
   with part.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==receipt['sha256']
   pending={r['archive_path']:r for r in matches};decomp=subprocess.Popen(['zstd','-q','-d','-c',str(part)],stdout=subprocess.PIPE)
   try:
    with tarfile.open(fileobj=decomp.stdout,mode='r|') as outer:
     for member in outer:
      row=pending.get(member.name)
      if row is None:continue
      with outer.extractfile(member) as f:data=f.read()
      digest=hashlib.sha256(data).hexdigest();assert digest!=wanted[row['path']]['sha256'],'Sanitized source still equals original unsafe bytes'
      count=0;members=0
      with tarfile.open(fileobj=io.BytesIO(data),mode='r:*') as inner:
       for entry in inner:
        if entry.isfile():
         with inner.extractfile(entry) as f:count+=B.sanitize(f.read())[1]
         members+=1
      assert count==0,'Sanitized nested archive still matches a credential pattern'
      verified[row['path']]={'state':'verified','part':receipt['hf_path'],'archive_sha256':receipt['sha256'],'nested_sha256':digest,'original_nested_sha256':wanted[row['path']]['sha256'],'manifest_redactions':row['redactions'],'inspected_nested_files':members,'nested_credential_matches_after':0,'revision':info.sha,'verified_unix':time.time()}
      pending.pop(member.name)
      if not pending:break
   finally:
    decomp.stdout.close()
    if decomp.poll() is None:decomp.terminate()
    decomp.wait()
   assert not pending,'Expected container missing from verified part'
  B.atomic_json(OUT/'NODE0_OPAQUE_VERIFICATION.json',{'state':'verified' if len(verified)==len(wanted) else 'waiting','sources':verified,'pending_sources':sorted(set(wanted)-set(verified)),'updated_unix':time.time()})
  if len(verified)<len(wanted):time.sleep(20)
 assert len(verified)==len(wanted),'Deadline reached before pending source containers were verified'
 repairs=json.loads((OUT/'OPAQUE_REPAIRS_VERIFIED.json').read_text());assert len(repairs)==2 and all(x['receipt'] for x in repairs)
 B.atomic_json(OUT/'FIVE_SOURCE_CONTAINERS_VERIFIED.json',{'state':'verified_current_head_history_scrub_required','repo_id':B.HF_REPO,'node0':verified,'repaired_nodes':[{k:v for k,v in x.items() if k!='log'} for x in repairs],'all_five_nested_credential_scans_clean':True,'time':time.time()})
 print(json.dumps({'state':'verified','sources':len(verified)}),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  B.atomic_json(OUT/'NODE0_OPAQUE_VERIFICATION.json',{'state':'failed','error':f'{type(e).__name__}: {e}','traceback':traceback.format_exc(),'time':time.time()});raise
