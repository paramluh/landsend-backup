from concurrent.futures import ThreadPoolExecutor
import hashlib,io,json,pathlib,shlex,subprocess,tarfile,time
base=pathlib.Path(__file__).resolve().parent;source=pathlib.Path('/home/param/pose-worktrees/pose-opd-credit');run='/home/param/runs/pose-runs/teacher-partner-shortmatrix-v1';ssh=['ssh','-i','/home/param/.ssh/id_ed25519_cluster','-o','BatchMode=yes']; hosts=['landsend-0','landsend-3','landsend-5']
def remote(h,args):return subprocess.run(ssh+[h,shlex.join(args)],capture_output=True,text=True,check=True)
manifest={}
with tarfile.open(base/'source.tar','w') as t:
 for n in sorted(set(subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=source).decode().split('\0'))):
  if not n or '__pycache__' in n:continue
  p=source/n
  if not p.is_file() or p.is_symlink():continue
  b=p.read_bytes();info=t.gettarinfo(str(p),arcname=n);t.addfile(info,io.BytesIO(b));manifest[n]=hashlib.sha256(b).hexdigest()
for n,d in manifest.items():assert hashlib.sha256((source/n).read_bytes()).hexdigest()==d,n
(base/'manifest.json').write_text(json.dumps(manifest,indent=2))
choices={'landsend-0':[(24,0,33,0),(24,0,33,24)],'landsend-3':[(16,0,33,0),(32,0,33,0)],'landsend-5':[(24,64,33,0),(24,0,35,0)]}
def stage(h):
 remote(h,['mkdir','-p',run])
 for name in ['source.tar','manifest.json','supervisor.py']:subprocess.run(['rsync','-a','-e',shlex.join(ssh),str(base/name),h+':'+run+'/'+name],check=True,capture_output=True)
 remote(h,['python3','-c',"import pathlib,tarfile;p=pathlib.Path("+repr(run)+");(p/'code').mkdir(exist_ok=True);tarfile.open(p/'source.tar').extractall(p/'code',filter='data')"])
 jobs=[]
 for index,(batch,profile,cut,offset) in enumerate(choices[h]):
  name=f'team{index}-b{batch}-p{cut}-profile{profile}';args=['/home/param/pose-repo/.venv/bin/python','-m','finetune.pose_opd_credit.teacher_partner_benchmark','--config',run+'/code/finetune/pose_opd_credit/recipe.teacher_partner.json','--output',run+'/'+name,'--instances',str(batch),'--frames','375','--capacity-frames','1500','--repeats','2','--teacher-replay','block','--student-backbone-cut','25','--partner-backbone-cut',str(cut),'--profile-calls',str(profile),'--profile-warmup','25','--row-offset',str(offset)]
  command=['bash','-c','source /home/param/pose-repo/finetune/transcript_pose/environment/activate.sh; exec "$@"','short-rollout',*args]
  jobs.append({'name':name,'env':{'CUDA_VISIBLE_DEVICES':'0,1,2,3' if index==0 else '4,5,6,7','OMP_NUM_THREADS':'1','PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1','PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'},'command':command})
 spec={'code':run+'/code','jobs':jobs,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip(),'source_manifest_file':'manifest.json'}
 remote(h,['python3','-c',"from pathlib import Path;Path("+repr(run+'/launch.json')+").write_text("+repr(json.dumps(spec))+")"])
 script="import subprocess,pathlib,json;root=pathlib.Path("+repr(run)+");log=open(root/'supervisor.log','ab',buffering=0);p=subprocess.Popen(['python3',str(root/'supervisor.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True);print(json.dumps({'pid':p.pid}))"
 result=json.loads(remote(h,['python3','-c',script]).stdout);print(json.dumps({'host':h,**result}),flush=True);return {'host':h,**result}
with ThreadPoolExecutor(max_workers=3) as p:receipts=list(p.map(stage,hosts))
(base/'launched.json').write_text(json.dumps({'runtime':run,'receipts':receipts,'time':time.time()},indent=2))
