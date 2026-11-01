import os,json,re,subprocess,hashlib,concurrent.futures
from pathlib import Path
OUT=Path('/home/param/outputs/landsend-backup-20261031/git')
REF='refs/heads/tmp/landsend-backup-param'
ENV={**os.environ,'GIT_AUTHOR_NAME':'Codex','GIT_AUTHOR_EMAIL':'codex@openai.com','GIT_COMMITTER_NAME':'Codex','GIT_COMMITTER_EMAIL':'codex@openai.com','GIT_TERMINAL_PROMPT':'0'}
HELPER='credential.helper=!/home/param/pose-repo/.venv/bin/python /home/param/outputs/pose-basic-opd-launch-20261031/git_hf_credential.py'
def git(repo,*args,data=None,env=None,check=True):
 p=subprocess.run(['git','-C',repo,'-c',HELPER,*args],input=data,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env={**ENV,**(env or {})})
 if check and p.returncode:raise RuntimeError(p.stderr.decode(errors='replace')[:2000])
 return p

def text(repo,*args,**kw):return git(repo,*args,**kw).stdout.decode().strip()
def urls(row):return sorted(set(x.split('\t')[1].split(' ')[0] for x in row['worktrees'][0]['remotes']))
def home(row):
 u=urls(row)
 if 'https://huggingface.co/InternalCan/pose-repo' in u:return 'github:canopy-labs-internal/param-pose'
 if not u:return 'orphan:'+row['repo']
 return u[0].removesuffix('.git').replace('git@github.com:','github:').replace('https://github.com/','github:')
records=[]
for file in OUT.glob('**/*-result.json'):
 row=json.loads(file.read_text());row['bundle']=str(file.with_name(file.name.replace('-result.json','.bundle')));records.append(row)
groups={}
for row in records:groups.setdefault(home(row),[]).append(row)
results=[]
for identity,group in sorted(groups.items()):
 if identity.startswith('orphan:'):continue
 local=[x for x in group if x['host']=='landsend-0']
 primary=next((x for x in local if x['repo'].startswith('/home/param/') and '/pose-worktrees/' not in x['repo']),local[0])
 repo=primary['repo']
 if identity.endswith('/param-pose'):repo='/home/param/pose-repo'
 allurls=sorted(set(u for row in group for u in urls(row)))
 parents=[]
 for row in group:
  if git(repo,'cat-file','-e',row['commit']+'^{commit}',check=False).returncode:
   dst='refs/landsend-backup/import/'+row['host']+'/'+hashlib.sha256(row['repo'].encode()).hexdigest()[:12]
   git(repo,'fetch',row['bundle'],REF+':'+dst)
  parents.append(row['commit'])
 old_remote=[]
 for url in allurls:
  found=text(repo,'ls-remote',url,REF)
  if found:
   old=found.split()[0];dst='refs/landsend-backup/remote/'+hashlib.sha256(url.encode()).hexdigest()[:12]
   git(repo,'fetch',url,REF+':'+dst);parents.append(old);old_remote.append({'url':url,'commit':old})
 parents=list(dict.fromkeys(parents))
 manifest={'format':'landsend-cluster-git-backup-v1','home':identity,'source_snapshots':group,'previous_remote_backups':old_remote,'restore':'Each source_snapshots[].worktrees[].working_commit is an exact worktree snapshot. git checkout that SHA; staged_commit preserves its original index separately. All original branch tips are parents of backup commits.'}
 index=OUT/(hashlib.sha256(identity.encode()).hexdigest()[:12]+'.final.index');index.unlink(missing_ok=True);env={'GIT_INDEX_FILE':str(index)}
 git(repo,'read-tree',primary['commit'],env=env)
 blob=text(repo,'hash-object','-w','--stdin',data=(json.dumps(manifest,indent=2)+'\n').encode())
 git(repo,'update-index','--add','--cacheinfo','100644,'+blob+',LANDSEND_BACKUP_MANIFEST.json',env=env)
 tree=text(repo,'write-tree',env=env);args=[]
 for parent in parents:args+=['-p',parent]
 commit=text(repo,'commit-tree',tree,*args,data=b'Backup all cluster worktrees and local branches before landsend rental shutdown\n')
 git(repo,'update-ref',REF,commit);index.unlink(missing_ok=True)
 result={'home':identity,'repo':repo,'commit':commit,'destinations':allurls,'snapshots':len(group),'worktrees':sum(len(x['worktrees']) for x in group),'pushes':[]}
 results.append(result)
 (OUT/'cluster-aggregate.json').write_text(json.dumps(results,indent=2))
 print(json.dumps({'assembled':identity,'commit':commit,'worktrees':result['worktrees']}),flush=True)

def push_one(item):
 result,url=item;repo=result['repo'];label=hashlib.sha256(url.encode()).hexdigest()[:12]
 command=['git','-C',repo,'-c',HELPER,'push',url,REF+':'+REF]
 with (OUT/('push-'+label+'.log')).open('w') as output:
  p=subprocess.run(command,stdout=output,stderr=subprocess.STDOUT,env=ENV,timeout=900)
 pushed=text(repo,'ls-remote',url,REF) if p.returncode==0 else ''
 receipt={'url':url,'returncode':p.returncode,'expected':result['commit'],'remote':pushed.split()[0] if pushed else None,'log':str(OUT/('push-'+label+'.log'))}
 receipt['verified']=receipt['remote']==receipt['expected']
 (OUT/('push-'+label+'.json')).write_text(json.dumps(receipt,indent=2))
 print(json.dumps(receipt),flush=True)
 return receipt
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
 list(pool.map(push_one,[(result,url) for result in results for url in result['destinations']]))
