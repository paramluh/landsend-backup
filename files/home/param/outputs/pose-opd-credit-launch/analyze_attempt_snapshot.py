from pathlib import Path
import collections,csv,datetime as dt,json,math,sys
specpath=Path(sys.argv[1]);spec=json.loads(specpath.read_text());root=specpath.parent/'monitor'/'snapshot';events=[];samples=[]
for hi,h in enumerate(spec['hosts']):
 p=root/h
 for f in p.glob('events-rank*.jsonl'):
  for line in f.read_text().splitlines():
   try:events.append(json.loads(line))
   except json.JSONDecodeError:pass
 for row in csv.reader((p/'gpu-samples.csv').read_text().splitlines()):
  if len(row)!=4:continue
  try:t=dt.datetime.strptime(row[0],'%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=dt.timezone.utc).timestamp();samples.append((t,hi*8+int(row[1]),int(row[2])))
  except (ValueError,TypeError):pass
updates=[e for e in events if e['event']=='update' and e['update']==1];profiles=[e for e in events if e['event']=='profile' and e['update']==1]
assert len(updates)==len(profiles)==40
assert all(math.isfinite(e['gradient_norm']) and abs(e['learning_rate']-3e-7)<1e-20 for e in updates)
cache=[e for e in events if e['event'] in ['actor_preallocation_cache','learner_preallocation_cache']]
summary={}
for name,step in [('learner_preallocation_cache',1),('actor_preallocation_cache',2)]:
 rows=[d for e in cache if e['event']==name and e['update']==step for d in e['devices']]
 assert len(rows)==40
 summary[name]={'ranks':len(rows),'live_allocated_unchanged':all(e['allocated_before_gib']==e['allocated_after_gib'] for e in rows),'released_unused_gib_min':min(e['released_reserved_gib'] for e in rows),'released_unused_gib_max':max(e['released_reserved_gib'] for e in rows),'live_after_gib_max':max(e['allocated_after_gib'] for e in rows),'reserved_after_gib_max':max(e['reserved_after_gib'] for e in rows)}
captures=[e for e in events if e['event']=='phase' and e.get('phase')=='actor_capture_end' and e['update']==2];assert len(captures)==20
phases=collections.defaultdict(list)
for rank in range(40):
 rows=[e for e in events if e['rank']==rank and e.get('update')==1]
 def time(name):return next(e['time'] for e in rows if e['event']==name or e.get('phase')==name)
 for key,lo,hi in [('actor',time('actor_build_begin'),time('teacher_update_begin')),('teacher',time('teacher_update_begin'),time('teacher_update_end')),('learner',time('replay_microbatch_begin'),time('replay_microbatch_end')),('optimizer',time('optimizer_begin'),time('profile'))]:
  phases[key].extend(m for t,r,m in samples if r==rank and lo<=t<=hi)
identity=json.loads((root/spec['hosts'][0]/'config.json').read_text());plan=json.loads((root/spec['hosts'][0]/'plan.json').read_text());assert len(plan['updates'])==129
sample_events=[e for e in events if e['event']=='rollout_samples_saved' and e['update']==1];assert len(sample_events)==1 and sample_events[0]['count']==4
result={'run_dir':spec['run_dir'],'source_commit':spec['source_commit'],'status':'first_actual_update_finite_and_second_actor_captured_all20pairs','ranks':40,'plan_updates':129,'first_update':{k:updates[0][k] for k in ['loss/direct','gradient_norm','learning_rate']},'first_update_wall_max_seconds':max(e['seconds'] for e in updates),'first_update_phase_max_seconds':{k:max(e[k] for e in profiles) for k in ['preparation_seconds','rollout_seconds','teacher_seconds','student_replay_seconds','optimizer_seconds']},'sampled_first_update_phase_nvml_peak_mib':{k:max(v) for k,v in phases.items()},'cache_boundaries':summary,'second_actor_captures':20,'second_prefetch_max_wait_seconds':max(e['wait_seconds'] for e in events if e['event']=='row_prefetch_ready' and e['update']==2),'raw_samples_update1':4,'method':identity['method'],'production_continues':True}
(root/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='method'},indent=2))
