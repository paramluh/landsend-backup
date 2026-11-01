from concurrent.futures import ThreadPoolExecutor
import json,shlex,shutil,subprocess,time
from pathlib import Path
from common import HOSTS,RUN,EXTRA_ARGS,HELPERS,SSH,sha,write_json,remote,command
base=Path(__file__).resolve().parent
prior=Path('/home/param/outputs/pose-opd-teacher-partner-b336-quickbench-v2/snapshot-1793456658036044427')
art=base/('snapshot-'+str(time.time_ns()));art.mkdir()
for name in ('source.tar','source-files.json','recipe.teacher_partner.json'):shutil.copy2(prior/name,art/name)
spec=json.loads((prior/'benchmark-launch.json').read_text());spec.update(mode='two-update-production-benchmark',hosts=HOSTS,world_size=56,run_dir=RUN,repo=RUN+'/code',config=RUN+'/code/finetune/pose_opd_credit/recipe.teacher_partner.json',normal_adam_initialization=True,extra_args=EXTRA_ARGS,helper_sha256={n:sha(base/n) for n in HELPERS})
specpath=art/'benchmark-launch.json';write_json(specpath,spec);write_json(art/'commands.json',{h:command(spec,h) for h in HOSTS})
job=Path(RUN)/'supervisor'
script='''import json,os,pathlib,tarfile
from common import sha,verify_files,write_json
job=pathlib.Path(%r);spec=json.loads((job/'spec.json').read_text());root=pathlib.Path(spec['repo'])
if any((job/name).exists() for name in ('started-launch.json','supervisor.json','torchrun.json','READY.json','GO.json')):raise RuntimeError('Already started')
if sha(job/'source.tar')!=spec['archive_sha256']:raise RuntimeError('Archive mismatch')
if not root.exists():
 temp=root.with_name(root.name+'.staging-'+str(os.getpid()));temp.mkdir()
 with tarfile.open(job/'source.tar') as tar:tar.extractall(temp,filter='data')
 for path in sorted(temp.rglob('*'),reverse=True):path.chmod(path.stat().st_mode & ~0o222)
 temp.chmod(temp.stat().st_mode & ~0o222);os.rename(temp,root)
verify_files(spec,job)
write_json(job/'STAGED.json',{'source_commit':spec['source_commit'],'config_sha256':spec['config_sha256'],'archive_sha256':spec['archive_sha256']})
print('staged '+str(root))
'''%str(job)
def stage(host):
 remote(host,['python3','-c','from pathlib import Path;p=Path('+repr(str(job))+');p.mkdir(parents=True,exist_ok=True);assert not (p/"started-launch.json").exists()'])
 files=[(art/'source.tar','source.tar'),(art/'source-files.json','source-files.json'),(specpath,'spec.json')]+[(base/n,n) for n in HELPERS]
 for src,name in files:subprocess.run(['rsync','-a','-e',shlex.join(SSH),str(src),host+':'+str(job/name)],check=True,capture_output=True)
 result=remote(host,['bash','-c','cd '+shlex.quote(str(job))+' && python3 -c '+shlex.quote(script)])
 return {'host':host,'result':result.stdout.strip()}
with ThreadPoolExecutor(max_workers=7) as pool:receipts=list(pool.map(stage,HOSTS))
write_json(art/'STAGED.json',{'time':time.time(),'hosts':receipts})
write_json(base/'PREPARED.json',{'spec':str(specpath),'staged':receipts,'updates':2,'started':False})
print(json.dumps({'spec':str(specpath),'staged':receipts},indent=2))
