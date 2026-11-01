import fcntl,json,os,signal,socket,subprocess,time
from pathlib import Path
run=Path('/home/param/runs/pose-runs/teacher-partner-layout38-v1')
def write(name,value):
 path=run/name;tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
def identity(pid):
 return {'pid':pid,'start_ticks':int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19])}
lock=open('/home/param/runs/pose-runs/.duplex-distill-gpus.lock','a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
if apps: raise RuntimeError('Nonempty GPU node: '+apps)
write('supervisor.json',dict(identity(os.getpid()),host=socket.gethostname(),time=time.time()))
command=['/home/param/pose-repo/.venv/bin/python','-m','finetune.pose_opd_credit.teacher_partner_layout_benchmark','--config',str(run/'code/finetune/pose_opd_credit/recipe.teacher_partner.json'),'--output',str(run/'results'),'--candidates','fanout2:25:33:0:threaded:38','--warm-trials','1','--more-candidates-file',str(run/'more-candidates.json'),'--disable-partner-text-history','--items-json','/home/param/runs/pose-runs/teacher-partner-layout38-v1/prior-selection/rows.json','--prepared-rows','/home/param/runs/pose-runs/teacher-partner-layout38-v1/prepared.pkl']
write('command.json',command)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',OMP_NUM_THREADS='1',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
with (run/'benchmark.log').open('ab',buffering=0) as output:
 child=subprocess.Popen(['bash','-c','source /home/param/pose-repo/finetune/transcript_pose/environment/activate.sh; exec "$@"','layout-benchmark',*command],cwd=run/'code',env=env,stdout=output,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
 write('child.json',identity(child.pid))
 def stop(signum=None,frame=None):
  if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 while child.poll() is None:
  if (run/'STOP.json').exists():stop()
  time.sleep(1)
 write('finished.json',dict(time=time.time(),exit_code=child.returncode))
