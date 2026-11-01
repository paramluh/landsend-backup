"""Read-only authoritative live-process/rank progress audit for the seven-node run."""
import concurrent.futures,datetime,json,math,os,shlex,subprocess,sys
from pathlib import Path
RUN=Path('/home/param/pose-runs/duplex-distill-7n-shared10-mbs2-20261029')
DEST=RUN/'verification/monitor-30min';DEST.mkdir(parents=True,exist_ok=True)
HOSTS=[f'landsend-{i}' for i in (0,1,2,3,4,5,12)]
PROBE=r'''
import datetime,json,os,pathlib,shutil,socket,subprocess
root=pathlib.Path('/home/param/pose-runs/duplex-distill-7n-shared10-mbs2-20261029');host=socket.gethostname().split('.')[0]
def process(pid,record=None):
 out={'pid':pid,'live':False}
 try:
  p=pathlib.Path('/proc')/str(pid);fields=p.joinpath('stat').read_text().rsplit(')',1)[1].split();cmd=p.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace');uid=p.stat().st_uid
  out.update(state=fields[0],ppid=int(fields[1]),start_ticks=fields[19],uid=uid,command=cmd,live=fields[0]!='Z' and (record is None or fields[19]==record['start_ticks']))
 except FileNotFoundError:pass
 return out
out={'host':host,'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rank_progress':[]}
job=root/'cluster/latest'
for name in ('supervisor','torchrun'):
 p=job/(name+'.json');record=json.loads(p.read_text()) if p.exists() else None;out[name]=process(record['pid'],record) if record else None
out['launch_id']=job.resolve().name
out['finished']=json.loads((job/'finished.json').read_text()) if (job/'finished.json').exists() else None
for path in root.glob('progress-rank*.json'):
 progress=json.loads(path.read_text());progress['process']=process(progress['pid']);progress['belongs_to_run']=str(root) in progress['process'].get('command','') and 'finetune/duplex_distill/train.py' in progress['process'].get('command','');out['rank_progress'].append(progress)
out['rank_progress'].sort(key=lambda x:x['rank'])
out['gpu_rows']=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu','--format=csv,noheader,nounits'],text=True).strip().splitlines()
out['gpu_pids']=[int(v) for v in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip().splitlines()]
def belongs_to_current_worker(pid):
 seen=set();current=pid;parent=out['torchrun']['pid'] if out['torchrun'] else -1
 while current not in (0,1) and current not in seen:
  if current==parent:return True
  seen.add(current);info=process(current)
  if not info['live']:return False
  current=info.get('ppid',0)
 return False
out['gpu_processes']=[process(pid) for pid in out['gpu_pids']]
for info in out['gpu_processes']:
 info['belongs_to_current_worker']=belongs_to_current_worker(info['pid']) and str(root) in info.get('command','') and 'finetune/duplex_distill/train.py' in info.get('command','')
out['extra_gpu_processes']=[info for info in out['gpu_processes'] if not info['belongs_to_current_worker']]
out['extra_gpu_pids']=[info['pid'] for info in out['extra_gpu_processes']]
out['disk']={str(p):{'total_bytes':shutil.disk_usage(p).total,'free_bytes':shutil.disk_usage(p).free,'owner_uid':p.stat().st_uid,'writable':os.access(p,os.W_OK)} for p in (root,root.parent)}
mem={line.split(':',1)[0]:int(line.split()[1]) for line in pathlib.Path('/proc/meminfo').read_text().splitlines() if len(line.split())>=2 and line.split()[1].isdigit()};out['memory_available_GiB']=mem['MemAvailable']/2**20;out['loadavg']=os.getloadavg()
log=(job/'node.log').read_text(errors='replace');out['error_lines']=[line[:1500] for line in log.splitlines() if any(token in line for token in ('OutOfMemoryError','Traceback (most recent call last)','AssertionError:','RuntimeError:','Nonfinite metric','Negative KL','NCCL WARN'))][-20:]
out['checkpoint_markers']=[str(p.relative_to(root)) for p in root.glob('checkpoints/**/COMPLETE')]
print(json.dumps(out))
'''
def collect(host):
 result=subprocess.run(['ssh','-i','/home/param/.ssh/id_ed25519_cluster','-o','BatchMode=yes','-o','ConnectTimeout=10',host,'python3 -'],input=PROBE,text=True,capture_output=True,timeout=35)
 if result.returncode:raise RuntimeError(result.stderr[-2000:])
 return json.loads(result.stdout)
nodes={};errors={}
with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool:
 tasks={pool.submit(collect,h):h for h in HOSTS}
 for task in concurrent.futures.as_completed(tasks):
  host=tasks[task]
  try:nodes[host]=task.result()
  except Exception as e:errors[host]=str(e)
previous_path=DEST/'latest.json';previous=json.loads(previous_path.read_text()) if previous_path.exists() else None
current=datetime.datetime.now(datetime.timezone.utc);alerts=[];ranks=[];progress_advanced={}
transition_path=DEST/'planned-transition.json';transition=json.loads(transition_path.read_text()) if transition_path.exists() else None
planned=bool(transition and transition.get('active') and transition.get('phase') in ('saving','restarting') and datetime.datetime.fromisoformat(transition['expires_utc'])>current)
if transition and transition.get('active') and not planned:alerts.append('Declared transition is expired or invalid; live failures are not suppressed')
for host in HOSTS:
 if host not in nodes:alerts.append(host+': query failed: '+errors[host]);continue
 n=nodes[host]
 if not planned and not all(n.get(name) and n[name]['live'] for name in ('supervisor','torchrun')):alerts.append(host+': supervisor/torchrun not live')
 if n['finished'] is not None and (not planned or n['finished']['exit_code']!=0):alerts.append(host+': process finished '+str(n['finished']))
 if len(n['rank_progress'])!=8:alerts.append(host+': expected8 progress records')
 if (not planned and len(n['gpu_pids'])!=8) or n['extra_gpu_pids']:alerts.append(host+': unexpected GPU process set '+str(n['extra_gpu_pids']))
 for r in n['rank_progress']:
  ranks.append(r['rank'])
  if not planned and (not r['process']['live'] or not r['belongs_to_run']):alerts.append(host+': rank'+str(r['rank'])+' no live owned training process')
  age=(current-datetime.datetime.fromisoformat(r['utc'])).total_seconds();r['progress_age_seconds']=age
  if not planned and age>180:alerts.append(host+': rank'+str(r['rank'])+' progress stale '+str(round(age,1))+'s')
  if not math.isfinite(r['kl']) or not math.isfinite(r['data_loss']):alerts.append(host+': rank'+str(r['rank'])+' nonfinite metric')
 if n['error_lines']:alerts.append(host+': error/warning log evidence')
 if any(d['free_bytes']<700*10**9 or not d['writable'] for d in n['disk'].values()):alerts.append(host+': low disk or not writable')
 if previous and host in previous.get('nodes',{}):
  old={r['rank']:r['step'] for r in previous['nodes'][host]['rank_progress']};progress_advanced[host]=all(r['step']>old.get(r['rank'],-1) for r in n['rank_progress'])
  if not planned and not progress_advanced[host]:alerts.append(host+': not every rank advanced since previous scheduled snapshot')
if set(ranks)!=set(range(56)):alerts.append('Rank coverage is not exactly0..55')
metrics=[json.loads(line) for line in (RUN/'metrics.jsonl').read_text().splitlines()]
latest_metrics=metrics[-1] if metrics else None
if latest_metrics and not all(math.isfinite(v) for v in latest_metrics.values() if isinstance(v,(int,float))):alerts.append('Latest rank0 metric contains nonfinite values')
report={'checked_utc':current.isoformat(),'scheduled_for_utc':sys.argv[1] if len(sys.argv)>1 else 'initial','run':str(RUN),'planned_transition':transition,'transition_active':planned,'all_checks_pass':not alerts,'alerts':alerts,'query_errors':errors,'all_rank_step_min_max':[min([r['step'] for n in nodes.values() for r in n['rank_progress']],default=0),max([r['step'] for n in nodes.values() for r in n['rank_progress']],default=0)],'every_rank_advanced_since_previous':progress_advanced,'rank0_latest_metrics':latest_metrics,'nodes':{h:nodes[h] for h in HOSTS if h in nodes},'checkpoint_readiness_note':'All rank process identities and ownership checked live; run paths writable and free disk recorded. Full checkpoint integrity/hang validation is handled by separate CPU tests.'}
name=current.strftime('%Y%m%dT%H%M%S.%fZ')+'.json';(DEST/name).write_text(json.dumps(report,indent=2)+'\n');tmp=DEST/'latest.tmp';tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(previous_path)
summary={'checked_utc':report['checked_utc'],'scheduled_for_utc':report['scheduled_for_utc'],'transition_active':planned,'all_checks_pass':report['all_checks_pass'],'alerts':alerts,'rank_steps':report['all_rank_step_min_max'],'nodes':{h:{'launch_id':n['launch_id'],'live_ranks':sum(r['process']['live'] for r in n['rank_progress']),'gpu_pids':len(n['gpu_pids']),'disk_free_TB':min(d['free_bytes'] for d in n['disk'].values())/1e12,'RAM_available_GiB':n['memory_available_GiB'],'uid':sorted({r['process'].get('uid') for r in n['rank_progress']})} for h,n in nodes.items()}}
print(json.dumps(summary),flush=True)
