import fcntl,json,os,pathlib,signal,subprocess,time
root=pathlib.Path(__file__).resolve().parent
spec=json.loads((root/'launch.json').read_text());children=[];peak={}
def write(name,value):
 p=root/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2));t.replace(p)
def rec(p):return {'pid':p,'start_ticks':int(pathlib.Path('/proc/%d/stat'%p).read_text().rsplit(')',1)[1].split()[19])}
def stop(*_):
 for p in children:
  if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
with open('/home/param/runs/pose-runs/.duplex-distill-gpus.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU busy'
 write('supervisor.json',{**rec(os.getpid()),'time':time.time()})
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 for job in spec['jobs']:
  env=os.environ.copy();env.update(job['env']);log=open(root/(job['name']+'.log'),'ab',buffering=0)
  p=subprocess.Popen(job['command'],env=env,cwd=spec['code'],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True);children.append(p)
  write(job['name']+'.pid.json',{**rec(p.pid),'time':time.time()})
 while any(p.poll() is None for p in children):
  if (root/'STOP.json').exists():stop()
  raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
  for line in raw.splitlines():
   i,m=map(int,line.split(','));peak[str(i)]=max(peak.get(str(i),0),m)
  write('status.json',{'time':time.time(),'codes':{j['name']:p.poll() for j,p in zip(spec['jobs'],children)},'peak_device_mib':peak})
  time.sleep(2)
 write('finished.json',{'time':time.time(),'codes':{j['name']:p.wait() for j,p in zip(spec['jobs'],children)},'peak_device_mib':peak})
