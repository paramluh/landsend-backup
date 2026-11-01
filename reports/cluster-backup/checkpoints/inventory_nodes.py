import concurrent.futures,json,pathlib,subprocess,time
ROOT=pathlib.Path('/home/param/outputs/landsend-backup-20261031/checkpoints')
PROBE=r'''
import os,json,socket
from pathlib import Path
home=Path('/home/param')
weight_suffixes={'.pt','.pth','.ckpt','.safetensors','.bin','.distcp'}
prune={'.git','.cache','.venv','node_modules','__pycache__','.codex','.local','.nv','.triton','.gnupg','.ssh','.config','datasets'}
files=[]
for root,dirs,names in os.walk(home,followlinks=False):
 dirs[:]=[d for d in dirs if d not in prune and not os.path.islink(os.path.join(root,d))]
 for name in names:
  p=Path(root)/name
  if p.suffix not in weight_suffixes or p.is_symlink():continue
  try:st=p.stat()
  except OSError:continue
  if st.st_size < 10_000_000:continue
  files.append({'path':str(p),'bytes':st.st_size,'mtime':st.st_mtime,'inode':st.st_ino,'device':st.st_dev})
print(json.dumps({'host':socket.gethostname(),'files':files}))
'''
def probe(i):
 host=f'landsend-{i}'
 cmd=['python3','-'] if i==0 else ['ssh','-i','/home/param/.ssh/id_ed25519_cluster','-o','BatchMode=yes','-o','ConnectTimeout=6','-o','StrictHostKeyChecking=yes',host,'python3 -']
 try:
  p=subprocess.run(cmd,input=PROBE,text=True,capture_output=True,timeout=120)
  if p.returncode:result={'host':host,'error':p.stderr[-1000:]}
  else:result=json.loads(p.stdout)
 except Exception as e:result={'host':host,'error':str(e)}
 (ROOT/(host+'.json')).write_text(json.dumps(result,indent=2))
 return {'host':host,'files':len(result.get('files',[])),'bytes':sum(x['bytes'] for x in result.get('files',[])),'error':result.get('error')}
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
 rows=list(pool.map(probe,range(16)))
(ROOT/'node-summary.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
