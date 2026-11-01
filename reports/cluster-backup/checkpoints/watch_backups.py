import json,os,pathlib,subprocess,time,traceback
ROOT=pathlib.Path('/home/param/outputs/landsend-backup-20261031/checkpoints');PY='/home/param/pose-repo/.venv/bin/python';REPO='InternalCan/landsend-checkpoints-backup-param'
def atomic(p,x):
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2));q.replace(p)
last_hash=0
while True:
 try:
  subprocess.run(['python3',str(ROOT/'collect_progress.py')],check=True,stdout=subprocess.DEVNULL,timeout=60)
  progress=json.loads((ROOT/'progress.json').read_text())
  if progress['all_complete'] or time.time()-last_hash>=300:
   subprocess.run([PY,str(ROOT/'verify_manifest_hashes.py')],check=True,stdout=subprocess.DEVNULL,timeout=120)
   last_hash=time.time()
  atomic(ROOT/'monitor-status.json',{'state':'verifying' if progress['all_complete'] else 'monitoring','time':time.time(),'completed_bytes':progress['completed_bytes'],'errors':[s for s in progress['states'] if s['state'] not in ['uploading','complete']]})
  if progress['all_complete']:
   verification=json.loads((ROOT/'manifest-hash-verification.json').read_text())
   if not verification['verified']:raise RuntimeError('Remote checkpoint SHA256 mismatch')
   from huggingface_hub import HfApi
   api=HfApi()
   complete={'state':'complete','time':time.time(),'repo':REPO,'hosts':[s['host'] for s in progress['states']],'completed_bytes':progress['completed_bytes'],'completed_leaves':sum(s['completed_tasks'] for s in progress['states']),'manifest_hash_files':verification['manifest_hash_files'],'manifest_hash_mismatches':verification['mismatches'],'checked_weights_revision':verification['revision']}
   atomic(ROOT/'COMPLETE.json',complete)
   commit=api.upload_folder(repo_id=REPO,folder_path=ROOT,path_in_repo='inventory',allow_patterns=['*.json','*.jsonl','*.py','README.md'],ignore_patterns=['*upload-status*','*upload-processes*'],commit_message='Finalize all-node checkpoint backup with completion receipts and SHA256 verification')
   complete['inventory_revision']=commit.oid;atomic(ROOT/'COMPLETE.json',complete)
   api.upload_file(repo_id=REPO,path_or_fileobj=ROOT/'COMPLETE.json',path_in_repo='COMPLETE.json',commit_message='Publish checkpoint backup completion marker')
   atomic(ROOT/'monitor-status.json',complete)
   print(json.dumps(complete),flush=True)
   break
 except Exception as e:
  atomic(ROOT/'monitor-status.json',{'state':'monitor-error','time':time.time(),'error':str(e)})
  print(traceback.format_exc(),flush=True)
 time.sleep(30)
