"""Read-only verification of the old run's save-and-stop result on every host."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shlex
import subprocess

HOSTS = ['landsend-2', 'landsend-3', 'landsend-4', 'landsend-5', 'landsend-12']
SCRIPT = r'''
import gc,hashlib,json,pathlib,socket,subprocess
root=pathlib.Path('/home/param/runs/pose-runs/pose-opd-credit-2min-b320-attempt2')
host=socket.gethostname();finished=root/'supervisor/finished.json'
if not finished.is_file():
 print(json.dumps({'host':host,'pending':True}));raise SystemExit(0)
terminal=json.loads(finished.read_text());assert terminal['exit_code']==0,terminal
marker=json.loads((root/'checkpoints/LATEST_RESUME.json').read_text());resume=pathlib.Path(marker['resume_path'])
assert (resume/'COMPLETE').read_text().strip()==str(marker['step'])
manifest=json.loads((resume/'train_state.json').read_text())
assert hashlib.sha256((resume/'train_state.json').read_bytes()).hexdigest()==marker['manifest_sha256']
assert manifest['host']==host and manifest['step']==marker['step'] and manifest['world']==40
entries=[e for e in manifest['files'] if e['host']==host and e['section']=='resume']
assert len(entries)==9,len(entries)
for e in entries:assert (resume/e['file']).stat().st_size==e['bytes'],e['file']
import torch
optimizer=[]
for e in entries:
 payload=torch.load(resume/e['file'],map_location='cpu',weights_only=True,mmap=True)
 assert set(payload['optimizers'])=={'main'},e['file']
 saved=payload['optimizers']['main'];assert saved['param_groups'],e['file']
 steps=[]
 for name,state in saved['state'].items():
  assert {'step','exp_avg','exp_avg_sq'}<=set(state),(e['file'],name)
  a,b=state['exp_avg']['data'],state['exp_avg_sq']['data']
  assert a.shape==b.shape and a.dtype==b.dtype,(name,a.shape,b.shape)
  step=state['step'];steps.append(int(step['data'].item()) if isinstance(step,dict) else int(step))
 if e['file'].startswith('rank'):
  assert payload.get('rng') is not None and steps,e['file']
  assert max(steps)==marker['step'],(e['file'],max(steps),marker['step'])
 optimizer.append({'file':e['file'],'state_parameters':len(steps),'minimum_step':min(steps) if steps else None,'maximum_step':max(steps) if steps else None})
 del payload,saved;gc.collect()
owned=[]
for key in ['torchrun','supervisor']:
 receipt=json.loads((root/('supervisor/'+key+'.json')).read_text());p=pathlib.Path('/proc/%s/stat'%receipt['pid'])
 if p.exists():
  current=int(p.read_text().rsplit(')',1)[1].split()[19])
  if current==int(receipt['start_ticks']):owned.append(receipt['pid'])
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
resultpath=root/'RESULT.json';result=json.loads(resultpath.read_text()) if resultpath.exists() else None
if host=='landsend-2':assert result['status']=='bounded_or_stopped' and result['next_update']==marker['step']
assert not owned,owned
print(json.dumps({'host':host,'pending':False,'exit_code':terminal['exit_code'],'step':marker['step'],'resume':str(resume),'local_ranks':manifest['local_ranks'],'manifest_verified':True,'optimizer_files':optimizer,'remaining_compute_processes':apps,'owned_processes':owned,'result_status':None if result is None else result['status']}))
'''

def check(host):
    cmd=['ssh','-i','/home/param/.ssh/id_ed25519_cluster','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',host,
         shlex.join(['/home/param/pose-repo/.venv/bin/python','-c',SCRIPT])]
    done=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
    if done.returncode:raise RuntimeError(host+': '+done.stderr+done.stdout)
    return json.loads(done.stdout)

with ThreadPoolExecutor(max_workers=5) as pool:
    results=list(pool.map(check,HOSTS))
Path('/home/param/outputs/pose-opd-teacher-partner-bench/stop-verification.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
