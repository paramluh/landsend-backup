"""Collect the two-update benchmark's small evidence files; no checkpoint writes."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shlex
import subprocess
from common import HOSTS, RUN, SSH, remote, write_json
BASE=Path(__file__).resolve().parent
EVIDENCE=BASE/'evidence'
script=r'''
import json,pathlib,socket
root=pathlib.Path('/home/param/runs/pose-runs/teacher-partner-b336-full-bench-v1');job=root/'supervisor'
r={'host':socket.gethostname().split('.')[0], 'finished':json.loads((job/'finished.json').read_text())}
for name in ['supervisor','torchrun']:
 rec=json.loads((job/(name+'.json')).read_text());p=pathlib.Path('/proc/%d/stat'%rec['pid']);r[name+'_alive']=p.exists() and int(p.read_text().rsplit(')',1)[1].split()[19])==rec['start_ticks']
if (root/'RESULT.json').exists():r['result']=json.loads((root/'RESULT.json').read_text())
print(json.dumps(r))
'''
def collect(host):
 verified=json.loads(remote(host,['python3','-B','-c',script]).stdout)
 dst=EVIDENCE/host;dst.mkdir(exist_ok=True)
 includes=['events-rank*.jsonl','metrics.jsonl','RESULT.json','config.json','plan.json','data-selection.json','supervisor/','supervisor/*.json','supervisor/training.log','supervisor/worker.log','raw-rollouts/','raw-rollouts/update-*/','raw-rollouts/update-*/manifest.json']
 command=['rsync','-a','-e',shlex.join(SSH)]
 for pattern in includes:command+=['--include','/'+pattern]
 command+=['--exclude','*',host+':'+RUN+'/',str(dst)+'/']
 subprocess.run(command,check=True,capture_output=True)
 write_json(dst/'verification.json',verified)
 return verified
if __name__=='__main__':
 with ThreadPoolExecutor(max_workers=7) as pool:verified=list(pool.map(collect,HOSTS))
 profiles=[]
 for host in HOSTS:
  for path in (EVIDENCE/host).glob('events-rank*.jsonl'):
   for line in path.read_text().splitlines():
    e=json.loads(line)
    if e['event'] in ('profile','actor_memory','optimizer_parked','optimizer_restored','rollout_samples_saved','update'):profiles.append({'host':host,**e})
 summary={'hosts':verified,'profiles':profiles,'scope':{'gpus':56,'nodes':7,'teams':14,'conversations_per_team':24,'global_batch':336,'production_requested_gpus':56,'production_global_batch':336}}
 write_json(EVIDENCE/'FINAL-EVIDENCE.json',summary)
 print(json.dumps({'hosts':len(verified),'profiles':len(profiles),'evidence':str(EVIDENCE/'FINAL-EVIDENCE.json')}))
