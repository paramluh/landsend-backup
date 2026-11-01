from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import json,shlex,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent;SSH=['ssh','-i','/home/param/.ssh/id_ed25519_cluster','-o','BatchMode=yes','-o','ConnectTimeout=10'];nodes=[f'landsend-{i}' for i in range(1,5)]
def remote(node,script,*args,python='/usr/bin/python3',timeout=30):
 cmd=['env','CUDA_VISIBLE_DEVICES=',python,'-c',(R/script).read_text(),str(R),*map(str,args)]
 return json.loads(subprocess.check_output([*SSH,node,shlex.join(cmd)],text=True,timeout=timeout))
def write(path,data):
 tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)
with ThreadPoolExecutor(max_workers=4) as p:states=list(p.map(lambda n:remote(n,'probe_training.py'),nodes))
main=states[0];steps=main['steps'];step=steps[-1]['step'] if steps else 0
common=set.intersection(*[set(s['full_checkpoints']) for s in states])
for cp in sorted(common):
 target=R/f'checkpoint-{cp:05d}-audit.json'
 if target.is_file() or not all(s['latest']==f'saves/step-{cp:05d}' for s in states):continue
 with ThreadPoolExecutor(max_workers=4) as p:checks=list(p.map(lambda n:remote(n,'audit_checkpoint.py',cp,python='/home/param/pose-repo/.venv/bin/python',timeout=180),nodes))
 assert len({s['plan'] for s in checks})==1
 write(target,{'status':'pass','step':cp,'checked_at':datetime.now(timezone.utc).isoformat(),'nodes':checks})
 print(f'CHECKPOINT {cp} PASS all32mainoptimizer states; decoder outersteps{cp}/AdamWsteps{cp*8}; BF16exports all4nodes')
audited=sorted(int(p.name.split('-')[1]) for p in R.glob('checkpoint-*-audit.json'))
bad=any(s['errors'] or s['exit_code'] not in (None,'0') for s in states)
done=all(s['exit_code']=='0' and not s['torchrun']['live'] for s in states)
status='failed' if bad else 'complete' if done and main['training_complete']=='200' and 200 in audited else 'running'
report={'status':status,'updated_at':datetime.now(timezone.utc).isoformat(),'step':step,'nodes':states,'audited_checkpoints':audited,'actual_training_ranks':[s['training_ranks'] for s in states]}
if len(steps)>1:report['mean_update_seconds_excluding_first']=sum(s['seconds'] for s in steps[1:])/len(steps[1:])
write(R/'independent-monitor-status.json',report)
print(json.dumps({k:v for k,v in report.items() if k!='nodes'}))
