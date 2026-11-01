"""Idempotent cluster Git catch-up. Fast-forward backup refs only; keep active indexes/worktrees intact."""
import os,json,subprocess,hashlib,concurrent.futures,time,shlex,fcntl
from pathlib import Path
BASE=Path('/home/param/outputs/landsend-backup-20261031/git');REF='refs/heads/tmp/landsend-backup-param';HOSTS=['landsend-0','landsend-1','landsend-2','landsend-3','landsend-4','landsend-5','landsend-12'];RUN=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime());OUT=BASE/'incremental'/RUN;OUT.mkdir(parents=True,exist_ok=True)
lock=(BASE/'incremental.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
SSH=['ssh','-i','/home/param/.ssh/id_ed25519_cluster','-o','BatchMode=yes','-o','ConnectTimeout=10'];SCP=['scp','-q','-i','/home/param/.ssh/id_ed25519_cluster'];RSYNC='ssh -i /home/param/.ssh/id_ed25519_cluster -o BatchMode=yes -o ConnectTimeout=10'
ENV={**os.environ,'GIT_AUTHOR_NAME':'Codex','GIT_AUTHOR_EMAIL':'codex@openai.com','GIT_COMMITTER_NAME':'Codex','GIT_COMMITTER_EMAIL':'codex@openai.com','GIT_TERMINAL_PROMPT':'0'}
HELPER='credential.helper=!/home/param/pose-repo/.venv/bin/python /home/param/outputs/pose-basic-opd-launch-20261031/git_hf_credential.py'
def git(repo,*args,data=None,env=None,check=True):
 p=subprocess.run(['git','-C',str(repo),'-c',HELPER,*args],input=data,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env={**ENV,**(env or {})})
 if check and p.returncode:raise RuntimeError('git '+args[0]+': '+p.stderr.decode(errors='replace')[:1500])
 return p.stdout
def text(repo,*args,**kw):return git(repo,*args,**kw).decode().strip()
def atomic(path,data):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');os.replace(tmp,path)
def urls(row):return sorted(set(x.split('\t')[1].split(' ')[0] for x in row['worktrees'][0]['remotes']))
def home(row):
 u=urls(row)
 if 'https://huggingface.co/InternalCan/pose-repo' in u:return 'github:canopy-labs-internal/param-pose'
 return u[0].removesuffix('.git').replace('git@github.com:','github:').replace('https://github.com/','github:') if u else 'orphan:'+row['repo']
statefile=BASE/'incremental-state.json'
if statefile.exists():baseline=json.loads(statefile.read_text())['groups']
else:
 files=list(BASE.glob('*-result.json'))+list((BASE/'remote').glob('*/*-result.json'));baseline=[json.loads(p.read_text()) for p in files]
if os.environ.get('LANDSEND_GIT_FORCE_HOME'):
 for row in baseline:
  if row['host']=='landsend-0' and home(row)==os.environ['LANDSEND_GIT_FORCE_HOME']:
   for worktree in row['worktrees']:worktree['branch']='forced-final-manifest-refresh'
for host in HOSTS:atomic(OUT/(host+'-baseline.json'),[r for r in baseline if r['host']==host])
def audit(host):
 source=OUT/(host+'-baseline.json');log=OUT/(host+'.log')
 if host=='landsend-0':cmd=['python3',str(BASE/'incremental_node.py'),RUN,str(source)]
 else:
  subprocess.run(SSH+[host,'mkdir -p '+shlex.quote(str(OUT))],check=True)
  subprocess.run(SCP+[str(BASE/'incremental_node.py'),str(BASE/'snapshot_repositories.py'),host+':'+str(BASE)+'/'],check=True)
  subprocess.run(SCP+[str(source),host+':'+str(source)],check=True)
  cmd=SSH+[host,'python3 '+shlex.quote(str(BASE/'incremental_node.py'))+' '+RUN+' '+shlex.quote(str(source))]
 with log.open('wb') as f:subprocess.run(cmd,check=True,stdout=f,stderr=subprocess.STDOUT,timeout=600)
 if host!='landsend-0':subprocess.run(['rsync','-a','-e',RSYNC,host+':'+str(OUT/host)+'/',str(OUT/host)+'/'],check=True)
 report=json.loads((OUT/host/'audit.json').read_text())
 for row in report['groups']:
  if row.get('bundle'):row['bundle']=str(OUT/host/Path(row['bundle']).name)
 return report
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:reports=list(pool.map(audit,HOSTS))
errors=[{'host':r['host'],**e} for r in reports for e in r['errors']];current=[g for r in reports for g in r['groups']]
if errors:
 result={'state':'errors','run_id':RUN,'errors':errors,'receipts':[],'note':'No remote pushes attempted because audit was incomplete.'};atomic(OUT/'COMPLETE.json',result);atomic(BASE/'INCREMENTAL_LATEST.json',result);print(json.dumps(result,indent=2));raise SystemExit(1)
groups={}
for row in current:groups.setdefault(home(row),[]).append(row)
receipts=[]
def write_tree(repo,base,files,label):
 index=OUT/(label+'.index');env={'GIT_INDEX_FILE':str(index)};git(repo,'read-tree',base,env=env)
 for name,data in files.items():
  blob=text(repo,'hash-object','-w','--stdin',data=data);git(repo,'update-index','--add','--cacheinfo','100644,'+blob+','+name,env=env)
 tree=text(repo,'write-tree',env=env);index.unlink(missing_ok=True);return tree
def push(repo,url,commit):
 git(repo,'push',url,commit+':'+REF);remote=text(repo,'ls-remote',url,REF).split()[0]
 if remote!=commit:raise RuntimeError('Remote verification mismatch: '+url)
 return {'url':url,'commit':commit,'verified':True}
def hf_mirror(source,source_commit,rows):
 mirror=BASE/'hf-pose-mirror';url='https://huggingface.co/InternalCan/pose-repo';old=text(mirror,'ls-remote',url,REF).split()[0];git(mirror,'fetch',url,'+'+REF+':refs/landsend-backup/hf-current');git(mirror,'fetch',str(source),source_commit)
 saved={}
 for name in ['LANDSEND_HF_COMMIT_MAP.csv','LANDSEND_HF_MIRROR.json']:
  data=git(mirror,'show',old+':'+name,check=False)
  if data:saved[name]=data
 oldmap=git(mirror,'show',old+':LANDSEND_HF_INCREMENTAL_MAP.json',check=False);mapping=json.loads(oldmap) if oldmap else []
 mirrored=[]
 # Mirror all current worktree snapshots from changed local repositories, preserving multi-worktree differences.
 for n,row in enumerate([w for r in rows if r['changed'] for w in r['worktrees']]):
  original=row['working_commit'];index=OUT/('hf-'+str(n)+'.index');env={'GIT_INDEX_FILE':str(index)};git(mirror,'read-tree',original,env=env)
  attrs=git(mirror,'show',original+':.gitattributes',check=False).decode(errors='replace');added=[]
  for raw in git(mirror,'ls-tree','-rlz',original).split(b'\0'):
   if not raw:continue
   meta,path=raw.split(b'\t',1);mode,kind,oid,size=meta.decode().split()
   if kind!='blob' or int(size)<=10_000_000:continue
   name=path.decode();content=git(mirror,'cat-file','blob',oid);pointer=git(mirror,'lfs','clean',name,data=content);blob=text(mirror,'hash-object','-w','--stdin',data=pointer);git(mirror,'update-index','--cacheinfo',mode+','+blob+','+name,env=env);added.append(json.dumps(name)+' filter=lfs diff=lfs merge=lfs -text')
  files={**saved,'.gitattributes':(attrs.rstrip()+'\n'+'\n'.join(added)+'\n').encode()}
  for name,data in files.items():
   blob=text(mirror,'hash-object','-w','--stdin',data=data);git(mirror,'update-index','--add','--cacheinfo','100644,'+blob+','+name,env=env)
  tree=text(mirror,'write-tree',env=env);index.unlink(missing_ok=True);commit=text(mirror,'commit-tree',tree,'-p',old,data=('HF compatible incremental worktree '+row['host']+':'+row['path']+'\n').encode());mapping.append({'original':original,'mirror':commit,'host':row['host'],'path':row['path'],'run_id':RUN});mirrored.append(commit)
 primary=next((m['mirror'] for m in reversed(mapping) if m['run_id']==RUN and m['host']=='landsend-0' and m['path']=='/home/param/pose-repo'),mirrored[0]);tree=write_tree(mirror,primary,{'LANDSEND_HF_INCREMENTAL_MAP.json':(json.dumps(mapping,indent=2)+'\n').encode()},'hf-final');args=['-p',old]
 for p in mirrored:args.extend(['-p',p])
 commit=text(mirror,'commit-tree',tree,*args,data=('Append cluster Git backup '+RUN+'; original GitHub commit '+source_commit+'\n').encode());receipt=push(mirror,url,commit);git(mirror,'update-ref',REF,commit);receipt['source_commit']=source_commit;receipt['mirror_map']='LANDSEND_HF_INCREMENTAL_MAP.json';return receipt
for identity,rows in sorted(groups.items()):
 changed=[r for r in rows if r['changed']]
 if not changed:continue
 if identity.startswith('orphan:'):
  receipts.append({'home':identity,'state':'bundle_ready_for_personal_repo','bundles':[r['bundle'] for r in changed]});continue
 try:
  local=[r for r in rows if r['host']=='landsend-0'];primary=next((r for r in local if r['repo'].startswith('/home/param/') and '/pose-worktrees/' not in r['repo']),local[0] if local else None)
  repo=Path(primary['repo']) if primary else OUT/('home-'+hashlib.sha256(identity.encode()).hexdigest()[:12]+'.git')
  if not primary:subprocess.run(['git','init','--bare',str(repo)],check=True,stdout=subprocess.DEVNULL)
  allurls=sorted(set(u for r in rows for u in urls(r)));parents=[]
  for row in changed:git(repo,'fetch',row['bundle'],row['bundle_ref']);parents.append(row['commit'])
  originalurls=[u for u in allurls if not (identity.endswith('/param-pose') and 'huggingface.co' in u)]
  for url in originalurls:
   found=text(repo,'ls-remote',url,REF)
   if found:old=found.split()[0];git(repo,'fetch',url,REF);parents.append(old)
  # Every new bundle is a parent; earlier unchanged snapshots remain in the old remote backup.
  treebase=next((w['tree'] for r in rows for w in r['worktrees'] if w['host']=='landsend-0' and w['path']==str(repo)),changed[0]['worktrees'][0]['tree'])
  manifest={'format':'landsend-incremental-git-backup-v1','run_id':RUN,'home':identity,'current_snapshots':rows,'restore':'Checkout working_commit for any worktree. Earlier snapshots remain ancestors. staged_commit preserves the staged index.'}
  tree=write_tree(repo,treebase,{'LANDSEND_BACKUP_MANIFEST.json':(json.dumps(manifest,indent=2)+'\n').encode()},hashlib.sha256(identity.encode()).hexdigest()[:12]);args=[]
  for p in dict.fromkeys(parents):args.extend(['-p',p])
  commit=text(repo,'commit-tree',tree,*args,data=('Incremental cluster backup '+RUN+'\n').encode())
  for group in rows:
   for worktree in group['worktrees']:
    git(repo,'merge-base','--is-ancestor',worktree['working_commit'],commit)
    if worktree['staged_commit']:git(repo,'merge-base','--is-ancestor',worktree['staged_commit'],commit)
  pushed=[push(repo,u,commit) for u in originalurls];git(repo,'update-ref',REF,commit)
  if identity.endswith('/param-pose') and any('huggingface.co' in u for u in allurls):pushed.append(hf_mirror(repo,commit,rows))
  receipts.append({'home':identity,'state':'verified','commit':commit,'pushes':pushed,'changed_worktrees':[p for r in changed for p in r['changed_worktrees']]})
 except Exception as exc:errors.append({'home':identity,'error':str(exc)})
result={'state':'complete' if not errors else 'errors','run_id':RUN,'audited_worktrees':sum(len(r['worktrees']) for r in current),'changed_groups':sum(r['changed'] for r in current),'receipts':receipts,'errors':errors,'completed_unix':time.time()};atomic(OUT/'COMPLETE.json',result);atomic(BASE/'INCREMENTAL_LATEST.json',result)
if not errors:atomic(statefile,{'groups':current,'run_id':RUN})
print(json.dumps(result,indent=2));raise SystemExit(bool(errors))
