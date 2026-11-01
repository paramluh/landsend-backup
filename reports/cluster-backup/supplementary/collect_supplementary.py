"""Read-only metadata/environment backup. Never export authentication databases."""
import os,sys,re,json,sqlite3,subprocess,socket,hashlib,base64,time,stat
from pathlib import Path
ROOT=Path('/home/param/outputs/landsend-backup-20261031/supplementary')
HOST=socket.gethostname().split('.')[0];OUT=ROOT/HOST;OUT.mkdir(parents=True,exist_ok=True)
REDACTIONS=0
PATTERNS=[
 re.compile(r'\b(?:hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-(?:proj-)?[A-Za-z0-9_-]{20,})\b'),
 re.compile(r'-----BEGIN (?:[A-Z ]*PRIVATE KEY)-----[\s\S]*?-----END (?:[A-Z ]*PRIVATE KEY)-----'),
 re.compile(r'\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b'),
 re.compile(r'(?i)(?:"?(?:access_token|refresh_token|id_token|api_key|secret_key|password|HF_TOKEN|GITHUB_TOKEN|GH_TOKEN)"?\s*[:=]\s*)["\x27][A-Za-z0-9_./+\-=]{16,}["\x27]'),
 re.compile(r'(?i)\bAuthorization\s*:\s*Bearer\s+[A-Za-z0-9_./+\-=]{12,}'),
 re.compile(r'(https?://)[^\s/@:]+:[^\s/@]+@'),
]
def scrub(value):
 global REDACTIONS
 if isinstance(value,str):
  for p in PATTERNS:value,n=p.subn('[REDACTED_CREDENTIAL]',value);REDACTIONS+=n
  return value
 if isinstance(value,bytes):return {'__base64__':base64.b64encode(value).decode()}
 if isinstance(value,list):return [scrub(x) for x in value]
 if isinstance(value,dict):return {k:scrub(v) for k,v in value.items()}
 return value

def save(name,data):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(scrub(data),indent=2)+'\n')

def command(args,timeout=40):
 try:
  p=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
  return {'command':args,'returncode':p.returncode,'stdout':scrub(p.stdout),'stderr':scrub(p.stderr)}
 except Exception as e:return {'command':args,'error':type(e).__name__+': '+str(e)}

# SQLite connections see committed WAL data; only selected non-auth metadata is exported.
for name in ['state_5.sqlite','memories_1.sqlite','goals_1.sqlite','thread_history_1.sqlite']:
 source=Path('/home/param/.codex')/name
 if not source.exists():continue
 conn=sqlite3.connect('file:'+str(source)+'?mode=ro',uri=True);conn.execute('BEGIN')
 tables=[]
 for table,sql in conn.execute("select name,sql from sqlite_master where type='table' and name not like 'sqlite_%'"):
  columns=[r[1] for r in conn.execute('pragma table_info('+json.dumps(table)+')')]
  count=conn.execute('select count(*) from '+json.dumps(table)).fetchone()[0]
  omit=table=='remote_control_enrollments' or (name=='thread_history_1.sqlite' and table in ['thread_items','thread_realtime_items'])
  record={'name':table,'sql':sql,'columns':columns,'row_count':count,'exported':not omit}
  if not omit:record['rows']=[list(r) for r in conn.execute('select * from '+json.dumps(table))]
  else:record['reason']='Sensitive connection metadata or conversation items already preserved in session JSONL archives.'
  tables.append(record)
 conn.rollback();conn.close()
 save('codex/'+name+'.sanitized.json',{'source':str(source),'snapshot_unix':time.time(),'tables':tables})

# Small reproducible configuration and service files; secrets are scrubbed before writing.
files=[]
starts=[Path('/home/param/.config/systemd'),Path('/home/param/.config/uv')]
for p in Path('/etc/systemd/system').glob('param*'):starts.append(p)
starts += [Path('/home/param')/n for n in ['.bashrc','.profile','.bash_logout','.bash_history','.gitconfig']]
for start in starts:
 candidates=[start] if start.is_file() else list(start.rglob('*')) if start.is_dir() else []
 for p in candidates:
  if p.is_symlink() or not p.is_file() or p.stat().st_size>1_000_000:continue
  try:data=p.read_text()
  except (UnicodeDecodeError,OSError):continue
  target=OUT/'files'/str(p).lstrip('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_text(scrub(data))
  files.append({'source':str(p),'backup':str(target.relative_to(OUT)),'mode':p.stat().st_mode&0o777,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
save('configuration-files.json',files)

# Ownership-aware storage inventory. Other users' directories are never traversed.
owned=[];tops=[]
for root in ['/home/param','/tmp','/var/tmp','/dev/shm','/opt','/mnt','/workspace','/data']:
 p=Path(root)
 if not p.is_dir():continue
 for q in p.iterdir():
  try:s=q.lstat()
  except OSError:continue
  row={'path':str(q),'owner_uid':s.st_uid,'directory':stat.S_ISDIR(s.st_mode),'symlink':stat.S_ISLNK(s.st_mode)}
  if q.is_symlink():row['target']=os.readlink(q)
  if root=='/home/param':tops.append(row)
  if s.st_uid==os.getuid():owned.append(row)
save('owned-storage-roots.json',owned);save('home-top-level.json',tops)

prune={'.git','.cache','.venv','venv','node_modules','__pycache__','.nv','.triton','packages','datasets','models'}
links=[];hidden_repos=[];skills=[]
for base,dirs,names in os.walk('/home/param',followlinks=False):
 if str(ROOT.parent) in base:dirs[:]=[];continue
 if '.git' in names or Path(base,'.git').is_dir():
  if any(x.startswith('.') for x in Path(base).relative_to('/home/param').parts):hidden_repos.append(base)
 for name in dirs+names:
  p=Path(base,name)
  if p.is_symlink():links.append({'path':str(p),'target':os.readlink(p)})
 dirs[:]=[x for x in dirs if x not in prune and not Path(base,x).is_symlink()]
for p in Path('/home/param/.codex/skills').rglob('*'):
 if p.is_file():skills.append({'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bundled':'.system' in p.parts})
save('symlinks.json',links);save('hidden-git-repositories.json',hidden_repos);save('skills-inventory.json',skills)

venvs=[]
for root in ['/home/param','/tmp','/opt']:
 for base,dirs,names in os.walk(root,followlinks=False):
  p=Path(base)
  if str(ROOT.parent) in base:dirs[:]=[];continue
  if root!='/opt' and base!=root:
   try:
    if p.stat().st_uid!=os.getuid():dirs[:]=[];continue
   except OSError:dirs[:]=[];continue
  if 'pyvenv.cfg' in names:venvs.append(str(p));dirs[:]=[];continue
  dirs[:]=[x for x in dirs if x not in {'.git','.cache','node_modules','__pycache__','datasets','models','runs','.codex','site-packages'} and not Path(base,x).is_symlink()]
envs=[]
for venv in sorted(set(venvs)):
 py=Path(venv)/'bin/python'
 if not py.exists():continue
 label=hashlib.sha256(venv.encode()).hexdigest()[:12]
 code='import importlib.metadata as m,json,sys,platform; print(json.dumps({"python":sys.version,"executable":sys.executable,"platform":platform.platform(),"packages":sorted([{"name":d.metadata.get("Name", ""),"version":d.version,"direct_url":d.read_text("direct_url.json")} for d in m.distributions()],key=lambda x:x["name"].lower())}))'
 result=command([str(py),'-c',code]);save('environments/'+label+'-metadata.json',result)
 freeze=command([str(py),'-m','pip','freeze','--all']);save('environments/'+label+'-pip-freeze.json',freeze)
 cfg=scrub((Path(venv)/'pyvenv.cfg').read_text())
 envs.append({'path':venv,'label':label,'pyvenv_cfg':cfg,'metadata_returncode':result.get('returncode'),'freeze_returncode':freeze.get('returncode')})
save('environments/index.json',envs)
save('system.json',{'uname':command(['uname','-a']),'os_release':Path('/etc/os-release').read_text(),'gpu':command(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader']),'cuda':command(['/usr/local/cuda/bin/nvcc','--version']),'dpkg':command(['dpkg-query','-W','-f=${Package}\t${Version}\n']),'user_crontab':command(['crontab','-l']),'systemd_user_units':command(['systemctl','--user','list-unit-files','--no-pager'])})
save('SUMMARY.json',{'host':HOST,'completed_unix':time.time(),'redactions':REDACTIONS,'environments':len(envs),'hidden_git_repositories':hidden_repos,'owned_external_roots':[x['path'] for x in owned if not x['path'].startswith('/home/param/')],'symlinks':len(links),'skills':len(skills),'custom_skills':sum(not x['bundled'] for x in skills),'configuration_files':len(files)})
print(json.dumps({'host':HOST,'environments':len(envs),'redactions':REDACTIONS,'hidden_git_repositories':hidden_repos,'output':str(OUT)}))
