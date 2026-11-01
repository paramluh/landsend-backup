"""Bounded, isolated optimizer smoke. No production restart or checkpoint saves."""
import argparse, fcntl, hashlib, json, os, signal, socket, subprocess, time
from pathlib import Path

def write(path, value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n'); temporary.replace(path)

def process(pid):
    fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
    return {'pid':pid,'start_ticks':int(fields[19])}

def main():
    p=argparse.ArgumentParser(); p.add_argument('spec',type=Path); a=p.parse_args()
    spec=json.loads(a.spec.read_text()); root=a.spec.parent
    host=socket.gethostname().split('.')[0]
    if host not in spec['hosts'] or spec['max_updates']!=1 or spec['no_save'] is not True:
        raise RuntimeError('Wrong host or smoke bound')
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=spec['worker_sha256']:
        raise RuntimeError('Worker SHA mismatch')
    archive=root/'source.tar'
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=spec['source_sha256']:
        raise RuntimeError('Source archive SHA mismatch')
    if hashlib.sha256((root/'recipe.json').read_bytes()).hexdigest()!=spec['config_sha256']:
        raise RuntimeError('Recipe SHA mismatch')
    lock=Path('/home/param/runs/pose-runs/.duplex-distill-gpus.lock')
    with lock.open('a') as held:
        fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
        if apps: raise RuntimeError('GPUs still occupied: '+apps)
        command=[spec['python'],'-m','torch.distributed.run','--nnodes=7','--nproc_per_node=8',
            '--node_rank='+str(spec['hosts'].index(host)),'--master_addr='+spec['master_addr'],
            '--master_port='+str(spec['port']),'--max_restarts=0','-m','finetune.pose_opd_credit.train',
            '--config',str(root/'recipe.json'),'--run-dir',spec['training_run'],
            '--no-save','--max-updates','1','--profile-replay-stages','--profile-policy-agreement','--policy-agreement-frames','8']
        write(root/'READY.json',{'host':host,**process(os.getpid()),'command':command,'time':time.time()})
        deadline=time.monotonic()+180
        while not (root/'GO.json').exists():
            if time.monotonic()>deadline: raise RuntimeError('NoGOwithin180s')
            time.sleep(.2)
        env=os.environ.copy();env.update(PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
            NCCL_SOCKET_IFNAME='=ibs14',NCCL_SOCKET_FAMILY='AF_INET6',GLOO_SOCKET_IFNAME='ibs14',
            NCCL_IB_DISABLE='0',NCCL_NET='IB',TORCH_NCCL_ASYNC_ERROR_HANDLING='1',OMP_NUM_THREADS='1',
            NCCL_DEBUG='WARN',CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',PYTHONUNBUFFERED='1',
            PYTHONDONTWRITEBYTECODE='1',S4_FFMPEG='/home/param/pose-repo/.venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2')
        child=None
        def stop(sig,frame):
            if child and child.poll() is None: os.killpg(child.pid,signal.SIGTERM)
            else: raise SystemExit(128+sig)
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        with (root/'training.log').open('ab',buffering=0) as out:
            child=subprocess.Popen(['bash','-c','set -e; source "$1"; shift; exec "$@"','smoke',spec['activate'],*command],
                cwd=root/'code',env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
            write(root/'torchrun.json',{'host':host,**process(child.pid),'time':time.time()})
            rc=child.wait()
        write(root/'finished.json',{'host':host,'exit_code':rc,'time':time.time()})
        return rc

if __name__=='__main__': raise SystemExit(main())
