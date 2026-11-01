import os,json,subprocess,socket
from pathlib import Path
skip={'.git','.cache','.venv','node_modules','__pycache__','.codex','.nv','.triton','.local','.hf-cli','.gnupg','.ssh'}
repos=[]
for root in [Path('/home/param'),Path('/tmp')]:
 for base,dirs,files in os.walk(root,followlinks=False):
  try: owned=os.stat(base).st_uid==os.getuid()
  except OSError:continue
  if not owned and str(base)!='/tmp': dirs[:]=[];continue
  dirs[:]=[x for x in dirs if x not in skip and not Path(base,x).is_symlink()]
  if '.git' in files or Path(base,'.git').is_dir():repos.append(base)
rows=[]
for path in repos:
 def git(*args):return subprocess.run(['git','-C',path,*args],capture_output=True,text=True).stdout.strip()
 rows.append(dict(path=path,common=git('rev-parse','--path-format=absolute','--git-common-dir'),head=git('rev-parse','HEAD'),branch=git('branch','--show-current'),remotes=git('remote','-v').splitlines(),status=git('status','--porcelain=v1'),refs=git('for-each-ref','refs/heads','--format=%(refname) %(objectname)').splitlines()))
print(json.dumps({'host':socket.gethostname(),'repos':rows},indent=2))
