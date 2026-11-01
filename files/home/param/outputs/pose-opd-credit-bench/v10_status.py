import concurrent.futures,json,pathlib,subprocess,shlex,collections
base=pathlib.Path('/home/param/outputs/pose-opd-credit-bench');spec=json.loads((base/'b320-cache-handoff-v10-launch.json').read_text())
code='''import pathlib,json,time
root=pathlib.Path(%r);rows=[]
for p in root.glob('events-rank*.jsonl'):
 for line in p.read_text().splitlines():
  try:rows.append(json.loads(line))
  except json.JSONDecodeError:pass
latest={}
for e in rows:latest[e['rank']]=e
f=root/'supervisor/finished.json';log=(root/'supervisor/worker.log').read_text(errors='replace')
errors=[line[:400] for line in log.splitlines() if any(x in line for x in ['Traceback (most recent call last)','Error:','Exception:','CUDA out of memory'])]
print(json.dumps({'time':time.time(),'finished':json.loads(f.read_text()) if f.exists() else None,'latest':list(latest.values()),'errors':errors[-3:],'cache':[e for e in rows if e['event']=='actor_preallocation_cache'],'updates':[e for e in rows if e['event']=='update'],'profiles':[e for e in rows if e['event']=='profile'],'prefetch':[e for e in rows if e['event']=='row_prefetch_ready'],'capture':[e for e in rows if e['event']=='phase' and e.get('phase') in ['actor_capture_begin','actor_capture_end','actor_microbatch_end']]}))
'''%spec['run_dir']
def get(host):
 p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',host,shlex.join(['python3','-c',code])],capture_output=True,text=True,check=True,timeout=30)
 d=json.loads(p.stdout);(base/('v10-status-'+host+'.json')).write_text(json.dumps(d,indent=2))
 phases=collections.Counter((e.get('update'),e['event'],e.get('phase')) for e in d['latest'])
 print(json.dumps({'host':host,'finished':d['finished'],'phases':[(list(k),v) for k,v in phases.items()],'errors':d['errors'],'cache_releases':len(d['cache']),'captures_complete':sum(e['phase']=='actor_capture_end' for e in d['capture']),'updates':{str(step):{'ranks':sum(e['update']==step for e in d['updates']),'gradient_norm':next(e['gradient_norm'] for e in d['updates'] if e['update']==step),'learning_rate':next(e['learning_rate'] for e in d['updates'] if e['update']==step)} for step in sorted({e['update'] for e in d['updates']})}}),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:list(pool.map(get,spec['hosts']))
