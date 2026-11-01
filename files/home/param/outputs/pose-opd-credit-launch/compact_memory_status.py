import concurrent.futures,json,pathlib,shlex,subprocess,sys
specpath=pathlib.Path(sys.argv[1]);spec=json.loads(specpath.read_text());out=specpath.parent/'monitor'
code=r'''import csv,datetime as dt,json,pathlib,time,os
root=pathlib.Path(RUN);ev=[]
for p in root.glob('events-rank*.jsonl'):
 for line in p.read_text().splitlines():
  try:ev.append(json.loads(line))
  except json.JSONDecodeError:pass
ranks=sorted({e['rank'] for e in ev});ss=[]
for row in csv.reader((root/'gpu-samples.csv').read_text().splitlines()):
 if len(row)!=4:continue
 try:ss.append((dt.datetime.strptime(row[0],'%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=dt.timezone.utc).timestamp(),ranks[int(row[1])],int(row[2])))
 except(ValueError,TypeError,IndexError):pass
result={}
for u in [1,2]:
 phase={};updates=[e for e in ev if e.get('update')==u and e['event']=='update'];profiles=[e for e in ev if e.get('update')==u and e['event']=='profile']
 for rank in ranks:
  rows=[e for e in ev if e['rank']==rank and e.get('update')==u]
  def tm(name):return next((e['time'] for e in rows if e['event']==name or e.get('phase')==name),None)
  for name,start,end in [('actor','actor_build_begin','teacher_update_begin'),('teacher','teacher_update_begin','teacher_update_end'),('learner','replay_microbatch_begin','replay_microbatch_end'),('optimizer','optimizer_begin','update')]:
   lo=tm(start);hi=tm(end) or time.time()
   if lo is not None:
    mm=[m for t,r,m in ss if r==rank and lo<=t<=hi]
    if mm:phase[name]=max(phase.get(name,0),max(mm))
 result[u]={'completed_ranks':len(updates),'updates':[{k:e[k] for k in ['rank','loss/direct','gradient_norm','learning_rate','seconds']} for e in updates],'phase_nvml_peak_mib':phase,'phase_max_seconds':{k:max(e[k] for e in profiles) for k in ['preparation_seconds','rollout_seconds','teacher_seconds','student_replay_seconds','optimizer_seconds']} if profiles else {}}
latest={}
for e in ev:latest[e['rank']]=e
sampler=json.loads((root/'gpu-sampler.json').read_text());pid=sampler['pid'];alive=pathlib.Path('/proc/'+str(pid)).exists()
log=(root/'supervisor/training.log').read_text(errors='replace');err=[s[:300] for s in log.splitlines() if any(x in s for x in ['Traceback (most recent call last)','CUDA out of memory','Error:'])]
print(json.dumps({'time':time.time(),'steps':result,'sampler_pid':pid,'sampler_alive':alive,'errors':err[-3:],'latest':[{k:e.get(k) for k in ['rank','update','event','phase','frame']} for e in latest.values()]}))
'''.replace('RUN',repr(spec['run_dir']))
def get(host):
 p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',host,shlex.join(['python3','-c',code])],capture_output=True,text=True,check=True,timeout=30)
 d=json.loads(p.stdout);d['host']=host;(out/('memory-'+host+'.json')).write_text(json.dumps(d,indent=2));return d
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:rows=list(pool.map(get,spec['hosts']))
result={'source_commit':spec['source_commit'],'run_dir':spec['run_dir'],'samplers_alive':all(x['sampler_alive'] for x in rows),'errors':{d['host']:d['errors'] for d in rows if d['errors']},'steps':{}}
for u in ['1','2']:
 updates=[v for d in rows for v in d['steps'][u]['updates']];ph={};sec={}
 for d in rows:
  for k,v in d['steps'][u]['phase_nvml_peak_mib'].items():ph[k]=max(ph.get(k,0),v)
  for k,v in d['steps'][u]['phase_max_seconds'].items():sec[k]=max(sec.get(k,0),v)
 result['steps'][u]={'completed_ranks':len(updates),'update':updates[0] if updates else None,'max_wall_seconds':max(v['seconds'] for v in updates) if updates else None,'phase_nvml_peak_mib':ph,'phase_max_seconds':sec}
result['latest']={d['host']:d['latest'][0] for d in rows};(out/'two-update-memory-summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
