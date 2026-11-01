"""Durably refresh all-node artifacts after the verified final training save."""
import concurrent.futures,fcntl,hashlib,json,os,shlex,socket,subprocess,threading,time,traceback
from pathlib import Path
BASE=Path('/home/param/outputs/landsend-backup-20261031');ORPHANS=BASE/'orphans';WORK=BASE/'final-artifact-refresh'
HOSTS=['landsend-0','landsend-1','landsend-2','landsend-3','landsend-4','landsend-5','landsend-12']
PY='/home/param/pose-repo/.venv/bin/python';HF_REPO='InternalCan/landsend-artifacts-backup-param'
EXPIRES=1793516400.0  # 2026-11-01T07:00:00Z / Oct31 midnight PDT
SSH=['ssh','-i','/home/param/.ssh/id_ed25519_cluster','-o','BatchMode=yes','-o','ConnectTimeout=8']
LOCK=threading.RLock();STATE={'state':'waiting_for_final_training_backup','hosts':{},'started_unix':time.time(),'pid':os.getpid(),'expires_utc':'2026-11-01T07:00:00Z'}
def atomic(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f'.{os.getpid()}.{threading.get_ident()}.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)
def update(host=None,**fields):
 with LOCK:
  if host:STATE['hosts'].setdefault(host,{}).update(fields)
  else:STATE.update(fields)
  STATE['updated_unix']=time.time();atomic(BASE/'FINAL_ARTIFACT_REFRESH.json',STATE)
def remote(host,code,args=()):
 command=['python3','-',*args] if host=='landsend-0' else SSH+[host,'python3 - '+shlex.join(args)]
 result=subprocess.run(command,input=code,text=True,capture_output=True,timeout=60)
 if result.returncode:raise RuntimeError(f'{host}: remote helper failed: {result.stderr[-1500:]}')
 return json.loads(result.stdout)
def probe(host,group=None):return remote(host,(BASE/'final_artifact_probe.py').read_text(),['--group',group] if group else [])
def deploy(host):
 paths=[ORPHANS/n for n in ['orphan_backup.py','remote_orphan_worker.py','checkpoint-scopes.json','sanitize_git_bundle.py','backup_source_archives.py']]+[BASE/'final_node_refresh.py']
 for p in paths:assert p.is_file(),f'Missing final-refresh dependency: {p}'
 if host!='landsend-0':
  for p in paths:
   command=SSH+[host,'cat > '+shlex.quote(str(p))]
   result=subprocess.run(command,input=p.read_bytes(),capture_output=True,timeout=60)
   if result.returncode:raise RuntimeError(f'{host}: cannot deploy {p.name}')
 return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
def launch(host,previous,previous_sources):
 code='''import json,os,pathlib,subprocess,time
base=pathlib.Path(%r);work=base/'final-artifact-refresh';work.mkdir(parents=True,exist_ok=True)
log=work/(%r+'-delta.log')
with log.open('ab',buffering=0) as out:
 p=subprocess.Popen([%r,'-u',str(base/'final_node_refresh.py')],stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','HF_HUB_DISABLE_PROGRESS_BARS':'1','LANDSEND_BACKUP_HF_REPO':%r,'OMP_NUM_THREADS':'2'})
r={'pid':p.pid,'host':%r,'started_unix':time.time(),'previous_groups':%r,'previous_source_groups':%r,'log':str(log)}
f=work/(%r+'.process.json');f.write_text(json.dumps(r,indent=2));print(json.dumps(r))
'''%(str(BASE),host,PY,HF_REPO,host,previous,previous_sources,host)
 return remote(host,code)
def verify_remote(evidence):
 from huggingface_hub import HfApi
 assert evidence['state']=='ready' and evidence['repo']==HF_REPO
 assert not evidence['canonical']['missing'],str(evidence['canonical']['missing'])
 api=HfApi();info=api.repo_info(HF_REPO,repo_type='dataset');assert info.private;revision=info.sha
 paths=[x['path'] for x in evidence['files']];found={}
 for start in range(0,len(paths),100):found.update({x.path:x for x in api.get_paths_info(HF_REPO,paths[start:start+100],repo_type='dataset',revision=revision)})
 for expected in evidence['files']:
  actual=found.get(expected['path']);assert actual is not None and actual.size==expected['bytes'],expected['path']
  lfs=actual.lfs
  sha=lfs.get('sha256') if isinstance(lfs,dict) else getattr(lfs,'sha256',None)
  if lfs:assert sha==expected['sha256'],expected['path']
  else:assert getattr(actual,'blob_id',None)==expected.get('git_blob_sha1'),expected['path']
 return {'state':'verified','group':evidence['group'],'revision':revision,'archives':evidence['archives'],'archive_bytes':evidence['archive_bytes'],'verified_remote_files':len(found),'canonical_files':evidence['canonical']['files'],'canonical_missing':[],'verified_unix':time.time()}
def run_host(host):
 try:
  while time.time()<EXPIRES:
   initial=probe(host);job=initial.get('job')
   if job:
    update(host,initial_marker=initial['marker'],initial_warnings=initial['errors'],job=job)
    break
   if not initial['workers'] and initial['marker']:
    warnings=initial['errors'];update(host,state='initial_stopped',initial_marker=initial['marker'],initial_warnings=warnings)
    hashes=deploy(host);job=launch(host,initial['groups'],initial['source_groups']);update(host,state='delta_running',job=job,worker_source_sha256=hashes)
    break
   if not initial['workers'] and not initial['marker']:raise RuntimeError('Initial artifact worker stopped without COMPLETE/PARTIAL marker')
   update(host,state='waiting_for_initial_worker',workers=initial['workers'],initial_marker=initial['marker']);time.sleep(20)
  else:raise TimeoutError('Rental deadline reached waiting for initial artifact backup')
  while time.time()<EXPIRES:
   current=probe(host)
   active=any(x['pid']==job['pid'] for x in current['workers'])
   if active:update(host,state='delta_running',job=job);time.sleep(20);continue
   groups=[g for g in current['groups'] if g not in job['previous_groups']]
   if len(groups)!=1:raise RuntimeError(f'Delta stopped with unexpected new groups: {groups}')
   evidence=probe(host,groups[0]);atomic(WORK/(host+'-evidence.json'),evidence)
   result=verify_remote(evidence)
   assert current.get('finished',{}).get('state')=='complete','Node refresh did not complete both worker stages'
   source_groups=[g for g in current['source_groups'] if g not in job['previous_source_groups']]
   assert len(source_groups)==1,f'Unexpected new source archive groups: {source_groups}'
   source_evidence=probe(host,source_groups[0]);atomic(WORK/(host+'-source-evidence.json'),source_evidence)
   result['source_archives_final']=verify_remote(source_evidence)
   atomic(WORK/(host+'-verified.json'),result);update(host,**result)
   return result
  raise TimeoutError('Rental deadline reached during final delta upload')
 except Exception as e:
  update(host,state='failed',error=f'{type(e).__name__}: {e}',traceback=traceback.format_exc());return {'state':'failed','host':host,'error':str(e)}
def main():
 WORK.mkdir(parents=True,exist_ok=True)
 with (WORK/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);update()
  while time.time()<EXPIRES:
   path=BASE/'final-training-backup.json'
   try:training=json.loads(path.read_text())
   except (OSError,ValueError):training={}
   if training.get('state')=='verified':break
   update(training_state=training.get('state','missing'));time.sleep(20)
  else:raise TimeoutError('Final training backup was not verified before rental expiry')
  update(state='refreshing',final_training_backup=training)
  with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool:results=list(pool.map(run_host,HOSTS))
  verified=all(x['state']=='verified' for x in results)
  issues=[{'host':h,**x} for h,v in STATE['hosts'].items() for x in v.get('initial_warnings',[]) if x['classification']!='rotating_dashboard_cache']
  update(state='artifacts_verified' if verified else 'needs_attention',all_final_groups_verified=verified,unclassified_initial_omissions=issues,restorable_unique_artifacts_covered=verified and not issues)
  git_hook=BASE/'git/incremental_backup.py'
  git_result=subprocess.run(['python3',str(git_hook)],capture_output=True,text=True,timeout=600,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
  (WORK/'git-final-catchup.log').write_text(git_result.stdout+'\n'+git_result.stderr)
  assert git_result.returncode==0,'Final Git catch-up failed; see git-final-catchup.log'
  git_receipt=json.loads((BASE/'git/INCREMENTAL_LATEST.json').read_text())
  assert git_receipt.get('state')=='complete' and not git_receipt.get('errors'),'Git catch-up receipt is incomplete'
  update(git_final_catchup=git_receipt)
  while time.time()<EXPIRES:
   try:cleanup=json.loads((BASE/'artifact-history-cleanup.json').read_text())
   except (OSError,ValueError):cleanup={}
   if cleanup.get('state')=='verified' and cleanup.get('known_bad_revisions_unreachable') and cleanup.get('clean_revision') and cleanup.get('repo_id')==HF_REPO:break
   update(state='waiting_for_artifact_history_cleanup',artifact_history_cleanup_state=cleanup.get('state','missing'));time.sleep(20)
  else:raise TimeoutError('Artifact dataset history cleanup was not verified before rental expiry')
  from huggingface_hub import HfApi
  api=HfApi();head=api.repo_info(HF_REPO,repo_type='dataset');assert head.private
  history=api.list_repo_commits(HF_REPO,repo_type='dataset')
  assert cleanup['clean_revision'] in {x.commit_id for x in history},'Artifact HEAD does not descend from verified clean history'
  update(artifact_history_cleanup={**cleanup,'verified_current_head':head.sha,'clean_ancestor_verified':True})
  hook=ORPHANS/'publish_receipts.py'
  if hook.exists():
   result=subprocess.run([PY,str(hook)],capture_output=True,text=True,timeout=300,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
   (WORK/'github-finalization.log').write_text(result.stdout+'\n'+result.stderr)
   if result.returncode:raise RuntimeError('Final GitHub publication hook failed; see github-finalization.log')
   publication=json.loads((BASE/'personal-github-publication.json').read_text())
   assert publication.get('state')=='verified' and publication.get('commit'),'GitHub publication lacks verified commit receipt'
   update(github_finalization={'state':'published','completed_unix':time.time(),'hook':str(hook),'receipt':publication})
  else:
   update(github_finalization={'state':'needs_parent_finalization','reason':'publish_receipts.py was not provided','repository':'paramluh/landsend-backup'})
  update(state='complete' if verified and not issues and STATE.get('github_finalization',{}).get('state')=='published' else 'needs_attention',finished_unix=time.time())
if __name__=='__main__':
 try:main()
 except Exception as e:update(state='needs_attention',error=f'{type(e).__name__}: {e}',traceback=traceback.format_exc())
