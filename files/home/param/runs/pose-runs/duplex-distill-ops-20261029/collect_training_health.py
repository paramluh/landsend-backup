import concurrent.futures
import datetime
import json
from pathlib import Path
import subprocess
import sys

run, launch = sys.argv[1:]
hosts = json.loads((Path(run) / 'cluster' / launch / 'launch.json').read_text())['hosts']
assert hosts and len(set(hosts)) == len(hosts) and set(hosts) <= {f'landsend-{i}' for i in (0, 1, 2, 3, 4, 5, 12)}
probe = r'''
import json,pathlib,subprocess,os,socket,re,sys
run,launch=json.loads(sys.argv[1]);root=pathlib.Path(run);job=root/'cluster'/launch
out={'host':socket.gethostname(),'gpus':subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().splitlines(),'gpu_pids':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip().splitlines(),'loadavg':os.getloadavg(),'progress':[],'errors':[]}
mem={}
for l in pathlib.Path('/proc/meminfo').read_text().splitlines():
 a=l.split();mem[a[0].rstrip(':')]=int(a[1])
out['mem_available_GiB']=mem['MemAvailable']/2**20
out['finished']=json.loads((job/'finished.json').read_text()) if (job/'finished.json').exists() else None
p=job/'node.log'
if p.exists():
 lines=p.read_text(errors='replace').splitlines();out['log_bytes']=p.stat().st_size
 for line in lines:
  if line.startswith('{'):
   try:
    e=json.loads(line)
    if e.get('event')=='step_complete':out['progress'].append(e)
   except json.JSONDecodeError:pass
  if re.search(r'(?:Error|Exception):',line):out['errors'].append(line)
 out['last_lines']=lines[-3:]
 rank_events=[]
 for event_file in root.glob('events-rank*.jsonl'):
  for line in event_file.read_text(errors='replace').splitlines():
   try:
    e=json.loads(line)
    if e.get('event')=='step_complete':rank_events.append(e)
   except json.JSONDecodeError:pass
 if rank_events:out['progress']=rank_events
print(json.dumps(out))
'''

def collect(host):
    import shlex
    command = 'python3 - ' + shlex.quote(json.dumps([run, launch]))
    result = subprocess.run(['ssh', '-i', '/home/param/.ssh/id_ed25519_cluster',
                             '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host,
                             command], input=probe, capture_output=True, text=True,
                            timeout=25, check=True)
    return host, json.loads(result.stdout)

with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
    nodes = dict(pool.map(collect, hosts))
report = {'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'run': run, 'launch': launch, 'nodes': nodes}
path = Path('/home/param/pose-runs/duplex-distill-ops-20261029') / (Path(run).name + '-health.json')
path.write_text(json.dumps(report, indent=2) + '\n')
archive = Path(run)/'verification'
archive.mkdir(parents=True, exist_ok=True)
stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
(archive/f'cluster-health-{stamp}.json').write_text(json.dumps(report, indent=2) + '\n')
for host, node in nodes.items():
    by_step = {}
    for event in node['progress']:
        by_step.setdefault(event['step'], []).append(event['rank'])
    print(host, json.dumps({'GPU_processes': len(node['gpu_pids']), 'gpus': node['gpus'],
          'steps': {k: sorted(set(v)) for k, v in by_step.items()}, 'errors': node['errors'][:2],
          'finished': node['finished'], 'host_RAM_available_GiB': round(node['mem_available_GiB'], 1),
          'last_lines': [line[:400] for line in node['last_lines'][-1:]] if not node['progress'] else None}))
