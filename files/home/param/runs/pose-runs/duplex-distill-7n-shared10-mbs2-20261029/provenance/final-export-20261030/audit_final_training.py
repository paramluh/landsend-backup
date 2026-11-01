import concurrent.futures,datetime,hashlib,json,pathlib,shlex,subprocess
RUN='/home/param/pose-runs/duplex-distill-7n-shared10-mbs2-20261029'
PROBE=r'''
import datetime,hashlib,json,pathlib,socket
root=pathlib.Path('/home/param/pose-runs/duplex-distill-7n-shared10-mbs2-20261029');meta=json.loads((root/'checkpoints/step-000284/export/meta.json').read_text());events=[];errors=[]
for f in root.glob('events-rank*.jsonl'):
 es=[]
 for l in f.read_text().splitlines():
  try:e=json.loads(l)
  except ValueError:continue
  if e.get('event') in ('step_complete','complete','save_complete'):es.append(e)
 if es:
  steps=[e for e in es if e['event']=='step_complete'];events.append({'file':f.name,'final_step':steps[-1] if steps else None,'terminal':es[-2:]})
finished=json.loads((root/'cluster/latest/finished.json').read_text())
shared={k:meta[k] for k in ('format','step','save_attempt','world','hosts','run_dir','run_tag','cfg','files','pose','decp_bb','decp_bb_alpha','vgain_theta_fp32')}
print(json.dumps({'host':socket.gethostname().split('.')[0],'finished':finished,'events':events,'shared_export_metadata_sha256':hashlib.sha256(json.dumps(shared,sort_keys=True).encode()).hexdigest(),'source_meta_sha256':hashlib.sha256((root/'checkpoints/step-000284/export/meta.json').read_bytes()).hexdigest()}))
'''
def get(host):
 r=subprocess.run(['ssh','-i','/home/param/.ssh/id_ed25519_cluster','-o','BatchMode=yes',host,'python3 -'],input=PROBE,text=True,capture_output=True,check=True,timeout=30)
 return host,json.loads(r.stdout)
with concurrent.futures.ThreadPoolExecutor(max_workers=7) as p:nodes=dict(p.map(get,[f'landsend-{i}' for i in (0,1,2,3,4,5,12)]))
steps=[e['final_step'] for n in nodes.values() for e in n['events'] if e['final_step']]
checks={'all56_rank_step284':len(steps)==56 and sorted(e['rank'] for e in steps)==list(range(56)) and all(e['step']==284 for e in steps),'all7_exit0':all(n['finished']['exit_code']==0 for n in nodes.values()),'all7_same_shared_export_metadata':len({n['shared_export_metadata_sha256'] for n in nodes.values()})==1}
doc={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'run':RUN,'all_checks_pass':all(checks.values()),'checks':checks,'nodes':nodes}
p=pathlib.Path('/home/param/pose-runs/duplex-distill-demo-deploy-20261030/final-training-audit.json');p.write_text(json.dumps(doc,indent=2)+'\n');print(json.dumps({k:v for k,v in doc.items() if k!='nodes'},indent=2))
