"""Snapshot tracked plus unignored WIP source and stage it without starting GPUs."""
from concurrent.futures import ThreadPoolExecutor
import hashlib,io,json,pathlib,shlex,subprocess,tarfile,time
source=pathlib.Path('/home/param/pose-worktrees/pose-opd-credit')
output=pathlib.Path('/home/param/outputs/pose-opd-teacher-partner-bench');output.mkdir(exist_ok=True)
root='/home/param/runs/pose-runs/pose-opd-teacher-partner-bench'
files=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=source).split(b'\0')
names=sorted({x.decode() for x in files if x and '__pycache__' not in pathlib.PurePath(x.decode()).parts and not x.decode().endswith(('.pyc','.pyo'))})
archive=output/f'source-{time.time_ns()}.tar.gz';manifest={}
with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
 for name in names:
  path=source/name
  if not path.is_file():continue
  data=path.read_bytes();info=tar.gettarinfo(str(path),arcname=name);info.size=len(data)
  tar.addfile(info,io.BytesIO(data));manifest[name]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
 receipt={'source':str(source),'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip(),'created_unix':time.time(),'files':manifest}
 data=(json.dumps(receipt,indent=2)+'\n').encode();info=tarfile.TarInfo('SOURCE_SNAPSHOT.json');info.size=len(data);tar.addfile(info,io.BytesIO(data))
receipt['archive']=str(archive);receipt['archive_sha256']=hashlib.sha256(archive.read_bytes()).hexdigest()
(output/'staging-latest.json').write_text(json.dumps(receipt,indent=2)+'\n')
ssh_base=['ssh','-i','/home/param/.ssh/id_ed25519_cluster','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes']
def stage(host):
 def remote(args):return subprocess.check_output(ssh_base+[host,shlex.join(args)],text=True)
 remote(['mkdir','-p',root+'/code'])
 subprocess.run(['rsync','-a','-e',shlex.join(ssh_base),str(archive),host+':'+root+'/source.tar.gz'],check=True)
 remote(['tar','xzf',root+'/source.tar.gz','-C',root+'/code'])
 script='''import pathlib,json,hashlib,shutil
root=pathlib.Path(%r);code=root/'code';m=json.loads((code/'SOURCE_SNAPSHOT.json').read_text());bad=[]
for name,e in m['files'].items():
 p=code/name
 if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=e['sha256']:bad.append(name)
assert not bad,bad
shutil.copyfile(code/'finetune/pose_opd_credit/recipe.teacher_partner.json',root/'recipe.teacher_partner.json')
print(json.dumps({'host':%r,'files_verified':len(m['files']),'code':str(code),'snapshot_unix':m['created_unix']}))
'''%(root,host)
 return json.loads(remote(['python3','-c',script]))
with ThreadPoolExecutor(max_workers=2) as pool:staged=list(pool.map(stage,['landsend-2','landsend-3']))
print(json.dumps({'archive_sha256':receipt['archive_sha256'],'staged':staged},indent=2))
