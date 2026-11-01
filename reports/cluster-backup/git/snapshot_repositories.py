"""Create non-disruptive Git backup commits; never change a working tree or its real index."""
import os,sys,json,subprocess,tempfile,shutil,hashlib,socket,re
from pathlib import Path
OUT=Path('/home/param/outputs/landsend-backup-20261031/git');OUT.mkdir(parents=True,exist_ok=True)
HOST=socket.gethostname().split('.')[0]
BRANCH='refs/heads/tmp/landsend-backup-param'
SKIP={'.git','.cache','.venv','venv','node_modules','__pycache__','.mypy_cache','.pytest_cache','.ruff_cache','.tox'}
SECRETS=re.compile(rb'(?:hf_[A-Za-z0-9]{24,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|AKIA[A-Z0-9]{16})')
IDENT={'GIT_AUTHOR_NAME':'Codex','GIT_AUTHOR_EMAIL':'codex@openai.com','GIT_COMMITTER_NAME':'Codex','GIT_COMMITTER_EMAIL':'codex@openai.com'}
def run(repo,*args,env=None,data=None,check=True):
 p=subprocess.run(['git','-C',str(repo),*args],input=data,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env={**os.environ,**IDENT,**(env or {})})
 if check and p.returncode:raise RuntimeError('git '+args[0]+': '+p.stderr.decode(errors='replace')[:1200])
 return p.stdout

def strgit(repo,*args,**kw):return run(repo,*args,**kw).decode().strip()
def safe(path):
 if path.is_symlink():return None
 if path.name in {'.env','.netrc','.git-credentials','credentials','token','stored_tokens','id_rsa','id_ed25519'}:return 'credential-filename'
 try:
  if path.stat().st_size>50_000_000:return 'large-file-to-artifact-backup'
  data=path.read_bytes()
 except (FileNotFoundError,IsADirectoryError,PermissionError):return None
 if SECRETS.search(data):return 'credential-content'
 return None

def snapshot(repo):
 repo=Path(repo);head=strgit(repo,'rev-parse','HEAD');common=strgit(repo,'rev-parse','--path-format=absolute','--git-common-dir')
 label=re.sub('[^A-Za-z0-9_.-]','_',str(repo).strip('/'))
 status_before=run(repo,'status','--porcelain=v1','-z')
 branch=strgit(repo,'branch','--show-current')
 temp=OUT/(label+'.index');temp.unlink(missing_ok=True)
 env={'GIT_INDEX_FILE':str(temp)}
 index_path=Path(strgit(repo,'rev-parse','--path-format=absolute','--git-path','index'))
 if index_path.exists():shutil.copyfile(index_path,temp)
 else:run(repo,'read-tree',head,env=env)
 staged_excluded=[]
 # Scan the staged bytes themselves: a safe working file can hide an older unsafe staged version.
 for raw in run(repo,'diff','--cached','--name-only','-z',head,env=env).split(b'\0'):
  if not raw:continue
  rel=raw.decode();blob=run(repo,'show',':'+rel,env=env,check=False)
  reason='credential-content' if SECRETS.search(blob) else ('large-file-to-artifact-backup' if len(blob)>50_000_000 else None)
  if Path(rel).name in {'.env','.netrc','.git-credentials','credentials','token','stored_tokens','id_rsa','id_ed25519'}:reason='credential-filename'
  if reason:
   staged_excluded.append({'path':rel,'reason':reason});run(repo,'reset','-q',head,'--',rel,env=env)
 staged_tree=strgit(repo,'write-tree',env=env)
 staged_commit=None
 if staged_tree!=strgit(repo,'rev-parse',head+'^{tree}'):
  staged_commit=strgit(repo,'commit-tree',staged_tree,'-p',head,data=f'Preserve staged index: {HOST}:{repo}\n'.encode())
 # Working-tree snapshot starts at HEAD; staging is done in the temporary index.
 run(repo,'read-tree',head,env=env)
 changed=run(repo,'diff','--name-only','-z','HEAD').split(b'\0')
 excludes=[]
 for raw in changed:
  if not raw:continue
  rel=raw.decode();reason=safe(repo/rel)
  if reason:excludes.append({'path':rel,'reason':reason,'tracked':True})
 run(repo,'add','-u',env=env)
 # Paths newly staged in the real index are not reported as untracked; explicitly snapshot their current bytes too.
 if any(changed):
  tracked_spec=OUT/(label+'.tracked.pathspec');tracked_spec.write_bytes(b'\0'.join(x for x in changed if x)+b'\0')
  run(repo,'add','--all','--pathspec-from-file='+str(tracked_spec),'--pathspec-file-nul',env=env)
 # Restore unsafe tracked files to HEAD in snapshot only, leaving real index/worktree intact.
 for item in excludes:
  run(repo,'reset','-q',head,'--',item['path'],env=env)
 paths=run(repo,'ls-files','--others','--exclude-standard',*[f'--exclude={x}/' for x in SKIP],'-z').split(b'\0')
 allowed=[]
 for raw in paths:
  if not raw:continue
  rel=raw.decode();p=repo/rel
  if any(x in SKIP for x in Path(rel).parts):continue
  reason=safe(p)
  if reason:excludes.append({'path':rel,'reason':reason,'tracked':False});continue
  allowed.append(raw)
 if allowed:
  spec=OUT/(label+'.pathspec');spec.write_bytes(b'\0'.join(allowed)+b'\0')
  run(repo,'add','--pathspec-from-file='+str(spec),'--pathspec-file-nul',env=env)
 tree=strgit(repo,'write-tree',env=env)
 parents=['-p',head]
 if staged_commit:parents+=['-p',staged_commit]
 commit=strgit(repo,'commit-tree',tree,*parents,data=f'Backup working tree: {HOST}:{repo}\nOriginal branch: {branch or "detached"}\n'.encode())
 run(repo,'update-ref','refs/landsend-backup/'+HOST+'/'+label,commit)
 temp.unlink(missing_ok=True)
 assert status_before==run(repo,'status','--porcelain=v1','-z'),'Working-tree status changed concurrently'
 return dict(host=HOST,path=str(repo),common=common,branch=branch,original_head=head,staged_commit=staged_commit,working_commit=commit,tree=tree,excluded=excludes,staged_excluded=staged_excluded,untracked_added=len(allowed),status_count=len([x for x in status_before.split(b'\0') if x]),remotes=strgit(repo,'remote','-v').splitlines())

paths=json.loads(sys.argv[1]);rows=[]
for path in paths:
 try:
  row=snapshot(path);rows.append(row);print(json.dumps({'snapshot':path,'commit':row['working_commit'],'excluded':len(row['excluded'])}),flush=True)
 except Exception as exc:
  rows.append({'host':HOST,'path':path,'error':str(exc)});print(json.dumps(rows[-1]),flush=True)
(OUT/(HOST+'-snapshots.json')).write_text(json.dumps(rows,indent=2))
groups={}
for row in rows:
 if 'common' in row:groups.setdefault(row['common'],[]).append(row)
for common,group in groups.items():
 repo=group[0]['path'];base=next((x for x in group if Path(x['path']).name=='pose-repo'),group[0])
 refs=strgit(repo,'for-each-ref','refs/heads','--format=%(refname) %(objectname)').splitlines()
 parents=list(dict.fromkeys([x['working_commit'] for x in group]+[x.rsplit(' ',1)[1] for x in refs]))
 manifest={'format':'landsend-git-backup-v1','host':HOST,'worktrees':group,'local_branches':refs,'restore':'git checkout <working_commit> to recover a worktree; staged_commit preserves its original index separately.'}
 temp=OUT/(hashlib.sha256(common.encode()).hexdigest()[:12]+'.aggregate.index');temp.unlink(missing_ok=True);env={'GIT_INDEX_FILE':str(temp)}
 run(repo,'read-tree',base['tree'],env=env)
 blob=strgit(repo,'hash-object','-w','--stdin',data=(json.dumps(manifest,indent=2)+'\n').encode())
 run(repo,'update-index','--add','--cacheinfo','100644,'+blob+',LANDSEND_BACKUP_MANIFEST.json',env=env)
 tree=strgit(repo,'write-tree',env=env);args=[]
 for parent in parents:args+=['-p',parent]
 commit=strgit(repo,'commit-tree',tree,*args,data=f'Preserve all {HOST} worktrees and local branches before rental shutdown\n'.encode())
 run(repo,'update-ref',BRANCH,commit)
 label=Path(common).parent.name+'-'+hashlib.sha256(common.encode()).hexdigest()[:8]
 bundle=OUT/(HOST+'-'+label+'.bundle')
 run(repo,'bundle','create',str(bundle),BRANCH)
 record={'host':HOST,'repo':repo,'common':common,'commit':commit,'bundle':str(bundle),'worktrees':group,'local_branches':refs}
 (OUT/(HOST+'-'+label+'-result.json')).write_text(json.dumps(record,indent=2))
 temp.unlink(missing_ok=True)
 print(json.dumps({'aggregate':repo,'commit':commit,'bundle':str(bundle),'bytes':bundle.stat().st_size}),flush=True)
