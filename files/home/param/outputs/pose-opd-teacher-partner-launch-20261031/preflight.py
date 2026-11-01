"""Read-only asset, RAM, disk, IPv6 and GPU inventory. Never initialize CUDA."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from common import HOSTS, RUN, remote, write_json

script = r'''
import fcntl,hashlib,json,pathlib,shutil,socket,subprocess
root=pathlib.Path('/home/param/runs/pose-runs')
assets={
 'student':'/home/param/models/imported-weights/ppm-aloe-poseduplex-raw-final/export-211',
 'pose_teacher':'/home/param/models/imported-weights/ppm-aloe-monoplex-pose-raw-bandfix-355/weights/export-355',
 'audio_teacher':'/home/param/models/imported-weights/ppm-aloe-zoom-cotrain/export-253',
 'mimi':'/home/param/models/mimi',
 'voxtral':'/home/param/models/imported-weights/aloe-ft-vox17127-nopersona/assets/voxtral-realtime',
 'qwen32':'/home/param/models/imported-weights/ppm-811-17850-base-bf16/configs/Qwen3-32B',
 'qwen4':'/home/param/models/imported-weights/ppm-811-17850-base-bf16/configs/Qwen3-4B',
 'dataset':'/home/param/datasets/aloe-duplex'}
result={'host':socket.gethostname().split('.')[0],'assets':{}}
for role,name in assets.items():
 p=pathlib.Path(name)
 rec={'path':str(p),'exists':p.is_dir()}
 if role in ('student','pose_teacher','audio_teacher'):
  rec['complete']=(p/'COMPLETE').is_file()
  rec['metadata']=any((p/x).is_file() for x in ['meta.json','.metadata'])
  shards=[x for x in p.iterdir() if x.suffix in ('.pt','.distcp','.safetensors')]
  rec['shards']=len(shards);rec['weight_bytes']=sum(x.stat().st_size for x in shards)
 elif role=='dataset':
  rec['ready']=(p/'DATASET_READY.json').is_file();rec['raw_gain']=(p/'original/provenance/RAWGAIN.json').is_file()
  rec['pose']=(p/'pose').is_dir();rec['raw']=(p/'original/stores').is_dir()
 else:rec['config']=(p/'config.json').is_file()
 result['assets'][role]=rec
silence=pathlib.Path('/home/param/models/imported-weights/ppm-aloe-poseduplex-raw-final/code/live/pose_cfg_silence.npy')
result['silence_sha256']=hashlib.sha256(silence.read_bytes()).hexdigest()
result['environment']={p:pathlib.Path(p).is_file() for p in ['/home/param/pose-repo/.venv/bin/python','/home/param/pose-repo/finetune/transcript_pose/environment/activate.sh']}
mem={line.split(':')[0]:int(line.split()[1])*1024 for line in pathlib.Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:','MemAvailable:','SwapFree:'))}
result['memory_bytes']=mem;result['disk_bytes']=shutil.disk_usage(root)._asdict()
result['gpu_processes']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True).strip().splitlines()
result['gpus']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().splitlines()
result['ib_ipv6']=subprocess.check_output(['ip','-6','-o','addr','show','dev','ibs14'],text=True).strip().splitlines()
lock=root/'.duplex-distill-gpus.lock'
result['gpu_lock_available']=True
if lock.exists():
 with lock.open('r') as f:
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:result['gpu_lock_available']=False
old=pathlib.Path('/home/param/models/trained-weights/pose-opd-credit-2min-b320-attempt2/checkpoints/step-000089/resume')
result['old_step89_resume_complete']=(old/'COMPLETE').is_file()
result['new_run_exists']=pathlib.Path('/home/param/runs/pose-runs/pose-opd-teacher-partner-b336-general-20261031').exists()
print(json.dumps(result))
'''


if __name__ == '__main__':
    def inspect(host):
        return json.loads(remote(host, ['python3', '-c', script]).stdout)
    with ThreadPoolExecutor(max_workers=7) as pool:
        results = list(pool.map(inspect, HOSTS))
    report = {'read_only': True, 'new_run': RUN, 'started': False, 'nodes': results}
    write_json(Path(__file__).with_name('preflight.json'), report)
    print(json.dumps(report, indent=2))
