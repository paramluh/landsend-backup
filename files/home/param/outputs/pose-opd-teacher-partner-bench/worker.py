import fcntl,json,os,pathlib,signal,subprocess,sys,time
spec=json.loads(pathlib.Path(sys.argv[1]).read_text()); job=pathlib.Path(sys.argv[1]).parent
lockpath=pathlib.Path('/home/param/runs/pose-runs/.duplex-distill-gpus.lock')
def record(pid):
    fields=pathlib.Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
    return {'pid':pid,'start_ticks':fields[19]}
with lockpath.open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    if busy: raise RuntimeError('Existing GPU processes: '+busy)
    (job/'supervisor.json').write_text(json.dumps(record(os.getpid())))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=spec.get('devices','0,1,2,3'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PPM_AUTOTUNE='0',PYTHONUNBUFFERED='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',S4_FFMPEG='/home/param/pose-repo/.venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2')
    command=['/home/param/pose-repo/.venv/bin/python','-m',spec['module'],*spec['args']]
    (job/'command.json').write_text(json.dumps(command,indent=2))
    child=subprocess.Popen(['bash','-c','source /home/param/pose-repo/finetune/transcript_pose/environment/activate.sh; exec "$@"','partner-bench',*command],env=env,cwd=spec['repo'],start_new_session=True)
    (job/'child.json').write_text(json.dumps(record(child.pid)))
    def stop(*_):
        if child.poll() is None: os.killpg(child.pid,signal.SIGTERM)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try: code=child.wait(timeout=spec.get('timeout',2700))
    except subprocess.TimeoutExpired:
        stop();code=child.wait(timeout=30)
    (job/'finished.json').write_text(json.dumps({'time':time.time(),'exit_code':code}))
    sys.exit(code)
