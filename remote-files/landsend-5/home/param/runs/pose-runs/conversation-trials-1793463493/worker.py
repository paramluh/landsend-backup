import fcntl,json,os,pathlib,signal,subprocess,time
run=pathlib.Path(__file__).parent
children=[]
def stop(*args):
    for p in children:
        if p.poll() is None: os.killpg(p.pid,signal.SIGTERM)
signal.signal(signal.SIGTERM,stop)
signal.signal(signal.SIGINT,stop)
with open('/home/param/runs/pose-runs/.duplex-distill-gpus.lock','a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if apps: raise RuntimeError('GPU processes present; not displacing them: '+apps)
    spec=json.loads((run/'spec.json').read_text())
    for item in spec['jobs']:
        p=subprocess.Popen(item['argv'],cwd=run/'code',stdout=open(run/(item['name']+'.log'),'w'),stderr=subprocess.STDOUT,start_new_session=True)
        children.append(p)
    (run/'children.json').write_text(json.dumps([{'pid':p.pid,'start_ticks':pathlib.Path('/proc/%d/stat'%p.pid).read_text().rsplit(')',1)[1].split()[19]} for p in children]))
    codes=[p.wait() for p in children]
    (run/'result.json').write_text(json.dumps({'codes':codes,'finished':time.time()}))
