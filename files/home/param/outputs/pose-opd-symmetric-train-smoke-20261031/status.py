import concurrent.futures,json,shlex,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
latest=json.loads((HERE/'LATEST.json').read_text()); root=latest['remote'];local=Path(latest['local'])
spec=json.loads((local/'spec.json').read_text())
script='''import json,os
from pathlib import Path
root=Path(ROOT)
def read(name):
 p=root/name
 return json.loads(p.read_text()) if p.exists() else None
proc=read('torchrun.json')
if proc:
 p=Path('/proc')/str(proc['pid'])/'stat'
 if p.exists():
  fields=p.read_text().rsplit(')',1)[1].split();proc['live']=fields[0]!='Z' and int(fields[19])==proc['start_ticks']
 else:proc['live']=False
out={'finished':read('finished.json'),'torchrun':proc,'ready':read('READY.json')}
for name in ('supervisor.log','training.log'):
 p=root/name
 if p.exists():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-7000));out[name]=f.read().decode(errors='replace')
out['last_events']={}
for event in sorted((root/'run').glob('events-rank*.jsonl')):
 try:out['last_events'][event.name]=[json.loads(line) for line in event.read_text().splitlines() if line][-2:]
 except Exception:pass
p=root/'run/metrics.jsonl'
if p.exists():out['metrics']=[json.loads(x) for x in p.read_text().splitlines() if x][-1:]
print(json.dumps(out))
'''.replace('ROOT',repr(root))
ssh=['ssh','-i','/home/param/.ssh/id_ed25519_cluster','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10']
def one(host):
 p=subprocess.run(ssh+[host,shlex.join([spec['python'],'-c',script])],capture_output=True,text=True,timeout=30)
 if p.returncode:return {'host':host,'error':p.stderr,'exit_code':p.returncode}
 return {'host':host,**json.loads(p.stdout)}
with concurrent.futures.ThreadPoolExecutor(7) as pool:out=list(pool.map(one,spec['hosts']))
(local/'STATUS.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
