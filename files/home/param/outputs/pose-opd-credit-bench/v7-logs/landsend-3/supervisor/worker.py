"""Bounded benchmark supervisor; stops only the recorded child process group."""
import fcntl,json,os,pathlib,signal,subprocess,sys,time,socket
spec=json.loads(pathlib.Path(sys.argv[1]).read_text()); job=pathlib.Path(sys.argv[1]).parent
lockpath=pathlib.Path('/home/param/runs/pose-runs/.duplex-distill-gpus.lock');lockpath.parent.mkdir(parents=True,exist_ok=True)
def record(p):
 fields=pathlib.Path(f'/proc/{p}/stat').read_text().rsplit(')',1)[1].split()
 return {'pid':p,'start_ticks':fields[19]}
with lockpath.open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
 if pids:raise RuntimeError('Existing GPU processes; refusing benchmark: '+pids)
 (job/'supervisor.json').write_text(json.dumps(record(os.getpid())))
 env=os.environ.copy();env.update(PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',NCCL_SOCKET_IFNAME='=ibs14',NCCL_SOCKET_FAMILY='AF_INET6',GLOO_SOCKET_IFNAME='ibs14',NCCL_IB_DISABLE='0',NCCL_NET='IB',TORCH_NCCL_ASYNC_ERROR_HANDLING='1',OMP_NUM_THREADS='1',NCCL_DEBUG='WARN',CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',PYTHONUNBUFFERED='1',S4_FFMPEG='/home/param/pose-repo/.venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2')
 command=['/home/param/pose-repo/.venv/bin/torchrun','--nnodes='+str(len(spec['hosts'])),'--nproc_per_node=8','--node_rank='+str(spec['hosts'].index(socket.gethostname())),'--master_addr='+spec['master_addr'],'--master_port='+str(spec['port']),'--max_restarts=0','-m','finetune.pose_opd_credit.train','--config',spec['config'],'--run-dir',spec['run_dir'],'--max-updates',str(spec['updates']),'--no-save',*spec.get('extra_args',[])]
 (job/'command.json').write_text(json.dumps(command,indent=2))
 child=subprocess.Popen(['bash','-c','source /home/param/pose-repo/finetune/transcript_pose/environment/activate.sh; exec "$@"','pose-opd-bench',*command],env=env,cwd=spec['repo'],start_new_session=True)
 (job/'torchrun.json').write_text(json.dumps(record(child.pid)))
 def stop(sig,frame):
  if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 rc=child.wait();(job/'finished.json').write_text(json.dumps({'time':time.time(),'exit_code':rc}))
 sys.exit(rc)
