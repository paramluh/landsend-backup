"""Read-only status of this production run's owned supervisors, ranks, and GPUs."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

from common import HOSTS, remote, write_json

script = r'''
import json,pathlib,socket,subprocess
root=pathlib.Path('/home/param/runs/pose-runs/pose-opd-symmetric-b336-website-pad0-20261031');job=root/'supervisor'
result={'host':socket.gethostname().split('.')[0]}
for name in ['supervisor','torchrun']:
 record=json.loads((job/(name+'.json')).read_text());p=pathlib.Path('/proc/%d/stat'%record['pid'])
 fields=p.read_text().rsplit(')',1)[1].split() if p.exists() else None
 record['running']=fields is not None and fields[0]!='Z' and int(fields[19])==record['start_ticks'];result[name]=record
if (job/'finished.json').exists():result['finished']=json.loads((job/'finished.json').read_text())
def tail(path,size=65536):
 with path.open('rb') as stream:
  stream.seek(max(0,path.stat().st_size-size));return stream.read().decode(errors='replace').splitlines()
result['log_tail']=tail(job/'training.log')[-12:]
result['rank_last_events']={}
for path in root.glob('events-rank*.jsonl'):
 lines=tail(path)
 for line in reversed(lines):
  try:result['rank_last_events'][path.name]=json.loads(line);break
  except ValueError:continue
metric=root/'metrics.jsonl'
result['metrics']=[]
if metric.exists():
 for line in tail(metric,262144):
  try: result['metrics'].append(json.loads(line))
  except ValueError: pass
result['metrics']=result['metrics'][-2:]
result['raw_sample_updates']={}
for path in (root/'raw-rollouts').glob('update-*/manifest.json'):
 try:
  value=json.loads(path.read_text());result['raw_sample_updates'][value['update']]=len(value['samples'])
 except ValueError:pass
result['gpu_usage']=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().splitlines()
result['gpu_compute_processes']=[]
for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader,nounits'],text=True).strip().splitlines():
 pid,uuid,memory=[value.strip() for value in line.split(',')]
 current=int(pid);seen=set();owned=False
 while current>1 and current not in seen:
  if current in (result['torchrun']['pid'],result['supervisor']['pid']):owned=True;break
  seen.add(current);path=pathlib.Path('/proc/%d/stat'%current)
  if not path.exists():break
  current=int(path.read_text().rsplit(')',1)[1].split()[1])
 result['gpu_compute_processes'].append({'pid':int(pid),'gpu_uuid':uuid,'used_memory_mib':int(memory) if memory.isdigit() else memory,'owned':owned})
print(json.dumps(result))
'''


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=7) as pool:
        results = list(pool.map(lambda host: json.loads(remote(host, ['python3', '-B', '-c', script]).stdout), HOSTS))
    doc = {'time': time.time(), 'nodes': results}
    write_json(Path(__file__).with_name('LATEST-STATUS.json'), doc)
    print(json.dumps(doc, indent=2))
