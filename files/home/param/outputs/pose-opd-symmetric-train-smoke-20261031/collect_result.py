import concurrent.futures,json,shlex,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;latest=json.loads((HERE/'LATEST.json').read_text());local=Path(latest['local']);root=latest['remote'];spec=json.loads((local/'spec.json').read_text())
script='''import json,subprocess
from pathlib import Path
root=Path(ROOT)
events=[]
for p in sorted((root/'run').glob('events-rank*.jsonl')):
 events.extend(json.loads(line) for line in p.read_text().splitlines() if line)
keep={'teacher_partner_rollout_timing','profile','policy_agreement','teacher_stage_profile','replay_stage_profile','actor_memory','optimizer_groups'}
out={'finished':json.loads((root/'finished.json').read_text()),'events':[e for e in events if e['event'] in keep],
'phases':[e for e in events if e['event']=='phase' and e.get('phase') in ('optimizer_step_begin','optimizer_step_end','actor_microbatch_end')],
'checkpoints': [str(p.relative_to(root)) for p in (root/'run').rglob('*.pt')],
'gpu_processes':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()}
p=root/'run/metrics.jsonl'
if p.exists():out['metrics']=[json.loads(x) for x in p.read_text().splitlines()if x]
print(json.dumps(out))
'''.replace('ROOT',repr(root))
ssh=['ssh','-i','/home/param/.ssh/id_ed25519_cluster','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10']
def one(host):
 p=subprocess.run(ssh+[host,shlex.join([spec['python'],'-c',script])],capture_output=True,text=True,timeout=30);p.check_returncode();v=json.loads(p.stdout);(local/(host+'-result.json')).write_text(json.dumps(v,indent=2)+'\n');return {'host':host,**v}
with concurrent.futures.ThreadPoolExecutor(7) as pool:out=list(pool.map(one,spec['hosts']))
assert all(v['finished']['exit_code']==0 and not v['gpu_processes'] and not v['checkpoints'] for v in out)
(local/'ALL-RESULTS.json').write_text(json.dumps({'collected_at':time.time(),'spec':spec,'nodes':out},indent=2)+'\n')
for v in out:
 print(v['host'],v['finished'],'GPUs idle',not v['gpu_processes'])
 for e in v['events']:
  if e['event']=='profile' and e['rank']%8==0:print(json.dumps(e))
