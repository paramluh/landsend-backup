from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import json,os,shlex,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parent;SSH=['ssh','-i','/home/param/.ssh/id_ed25519_cluster','-o','BatchMode=yes','-o','ConnectTimeout=10'];NODES=[f'landsend-{i}' for i in range(1,5)]
probe=(R/'probe_training.py').read_text();audit=(R/'audit_checkpoint.py').read_text();audited=set();last_progress=0

def stamp():return datetime.now(timezone.utc).isoformat()
def write(path,data):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)
def remote(node,code,*args,python='/usr/bin/python3',timeout=25):
 cmd=['env','CUDA_VISIBLE_DEVICES=',python,'-c',code,str(R),*map(str,args)]
 p=subprocess.run([*SSH,node,shlex.join(cmd)],capture_output=True,text=True,timeout=timeout,check=True)
 return json.loads(p.stdout)
while True:
 try:
  with ThreadPoolExecutor(max_workers=4) as pool:nodes=list(pool.map(lambda node:remote(node,probe),NODES))
  main=nodes[0];steps=main['steps'];step=steps[-1]['step'] if steps else 0
  bad=[n for n in nodes if n['errors'] or n['exit_code'] not in (None,'0')]
  status='failed' if bad else 'running';done=all(n['exit_code']=='0' and not n['torchrun']['live'] for n in nodes)
  report={'updated_at':stamp(),'status':status,'step':step,'nodes':nodes,'audited_checkpoints':sorted(audited)}
  if len(steps)>1:
   mean=sum(s['seconds'] for s in steps[1:])/len(steps[1:]);report['mean_update_seconds_excluding_first']=mean;report['remaining_update_seconds']=(200-step)*mean
  write(R/'monitor-status.json',report)
  common=set.intersection(*[set(n['full_checkpoints']) for n in nodes]);latest=[n['latest'] for n in nodes]
  for cp in sorted(common-audited):
   if all(x==f'saves/step-{cp:05d}' for x in latest):
    with ThreadPoolExecutor(max_workers=4) as pool:checks=list(pool.map(lambda node:remote(node,audit,cp,python='/home/param/pose-repo/.venv/bin/python',timeout=180),NODES))
    assert len({x['plan'] for x in checks})==1
    write(R/f'checkpoint-{cp:05d}-audit.json',{'status':'pass','step':cp,'checked_at':stamp(),'nodes':checks});audited.add(cp)
    print(f'CHECKPOINT PASS step{cp}: all32 ranks/main+decoder optimizers/scheduler/BF16 exports/LATEST verified',flush=True)
  if bad:
   print('TRAINING ERROR '+json.dumps(bad),flush=True);break
  if done:
   assert main['training_complete']=='200' and step==200 and 200 in audited
   report.update(status='complete',audited_checkpoints=sorted(audited));write(R/'monitor-status.json',report);print('TRAINING COMPLETE all4nodes exit0; finalcheckpoint audited',flush=True);break
  if time.monotonic()-last_progress>=600:
   print(f'PROGRESS step{step}/200 ranks='+str([n['training_ranks'] for n in nodes])+f' audited={sorted(audited)}',flush=True);last_progress=time.monotonic()
 except Exception as error:
  write(R/'monitor-warning.json',{'updated_at':stamp(),'error':repr(error)});print('MONITOR WARNING '+repr(error),flush=True)
 time.sleep(60)
