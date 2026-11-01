import concurrent.futures,json,pathlib,shlex,subprocess,sys,collections
specpath=pathlib.Path(sys.argv[1]);spec=json.loads(specpath.read_text());outdir=specpath.parent/'monitor';outdir.mkdir(exist_ok=True)
code='''import pathlib,json,time
root=pathlib.Path(%r);rows=[]
for p in root.glob('events-rank*.jsonl'):
 for line in p.read_text().splitlines():
  try:rows.append(json.loads(line))
  except json.JSONDecodeError:pass
latest={}
for e in rows:latest[e['rank']]=e
f=root/'supervisor/finished.json';c=root/'config.json';p=root/'plan.json';log=(root/'supervisor/training.log').read_text(errors='replace')
errors=[line[:400] for line in log.splitlines() if any(x in line for x in ['Traceback (most recent call last)','Error:','Exception:','CUDA out of memory'])]
print(json.dumps({'time':time.time(),'finished':json.loads(f.read_text()) if f.exists() else None,'latest':list(latest.values()),'errors':errors[-3:],'cache':[e for e in rows if e['event'] in ['actor_preallocation_cache','learner_preallocation_cache']],'updates':[e for e in rows if e['event']=='update'],'profiles':[e for e in rows if e['event']=='profile'],'prefetch':[e for e in rows if e['event']=='row_prefetch_ready'],'capture':[e for e in rows if e['event']=='phase' and e.get('phase') in ['actor_capture_begin','actor_capture_end','actor_microbatch_end']],'samples':[e for e in rows if e['event']=='rollout_samples_saved'],'config':json.loads(c.read_text()) if c.exists() else None,'plan_updates':len(json.loads(p.read_text())['updates']) if p.exists() else None}))
'''%spec['run_dir']
def get(host):
 result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',host,shlex.join(['python3','-c',code])],capture_output=True,text=True,check=True,timeout=30)
 d=json.loads(result.stdout);(outdir/('status-'+host+'.json')).write_text(json.dumps(d,indent=2))
 phases=collections.Counter((e.get('update'),e['event'],e.get('phase')) for e in d['latest'])
 print(json.dumps({'host':host,'finished':d['finished'],'phases':[(list(k),v) for k,v in phases.items()],'errors':d['errors'],'cache_releases':len(d['cache']),'captures_complete':sum(e['phase']=='actor_capture_end' for e in d['capture']),'updates':{str(step):{'ranks':sum(e['update']==step for e in d['updates']),'gradient_norm':next(e['gradient_norm'] for e in d['updates'] if e['update']==step),'learning_rate':next(e['learning_rate'] for e in d['updates'] if e['update']==step)} for step in sorted({e['update'] for e in d['updates']})},'samples':[(e['update'],e['count']) for e in d['samples']],'plan_updates':d['plan_updates']}),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:list(pool.map(get,spec['hosts']))
