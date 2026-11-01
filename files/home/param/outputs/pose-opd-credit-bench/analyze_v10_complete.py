from pathlib import Path
import csv,datetime as dt,json,math,statistics as st
root=Path('/home/param/outputs/pose-opd-credit-bench/v10-logs')
hosts=['landsend-2','landsend-3','landsend-4','landsend-5','landsend-12']
rows=[];peak=0
for host in hosts:
 p=root/host
 assert json.loads((p/'supervisor/finished.json').read_text())['exit_code']==0
 for f in p.glob('events-rank*.jsonl'):rows.extend(json.loads(x) for x in f.read_text().splitlines())
 peak=max(peak,max(int(x[2]) for x in csv.reader((p/'gpu-samples.csv').read_text().splitlines())))
updates=[x for x in rows if x['event']=='update'];assert len(updates)==80
assert all(math.isfinite(x['gradient_norm']) and math.isfinite(x['loss/direct']) for x in updates)
assert all(not x['synthetic_shape_only'] for x in updates)
profiles=[x for x in rows if x['event']=='profile'];assert len(profiles)==80
warm=[x for x in rows if x['event']=='optimizer_warmup'];assert len(warm)==40
cache=[x for x in rows if x['event']=='actor_preallocation_cache'];assert len(cache)==80
assert all(d['allocated_before_gib']==d['allocated_after_gib'] for x in cache for d in x['devices'])
assert all(d['reserved_after_gib']<=d['reserved_before_gib'] for x in cache for d in x['devices'])
captures=[x for x in rows if x['event']=='phase' and x.get('phase')=='actor_capture_end'];assert len(captures)==40
for rank in range(40):
 for step in (1,2):
  c=next(x for x in cache if x['rank']==rank and x['update']==step)
  build=next(x for x in rows if x['rank']==rank and x['event']=='phase' and x.get('phase')=='actor_build_begin' and x['update']==step)
  assert c['time']<build['time']
steps={}
for step in (1,2):
 ps=[x for x in profiles if x['update']==step];us=[x for x in updates if x['update']==step];cs=[d for x in cache if x['update']==step for d in x['devices']]
 steps[str(step)]={'ranks':len(us),'wall_seconds_max':max(x['seconds'] for x in us),'loss':us[0]['loss/direct'],'gradient_norm':us[0]['gradient_norm'],'learning_rate':us[0]['learning_rate'],'phase_seconds_max':{k:max(x[k] for x in ps) for k in ['preparation_seconds','rollout_seconds','teacher_seconds','student_replay_seconds','optimizer_seconds']},'live_allocated_unchanged_all_ranks':True,'preallocation_live_gib_min':min(x['allocated_before_gib'] for x in cs),'preallocation_live_gib_max':max(x['allocated_before_gib'] for x in cs),'released_unused_reserved_gib_min':min(x['released_reserved_gib'] for x in cs),'released_unused_reserved_gib_max':max(x['released_reserved_gib'] for x in cs),'reserved_after_gib_max':max(x['reserved_after_gib'] for x in cs),'prefetch_wait_seconds_max':max(x['wait_seconds'] for x in rows if x['event']=='row_prefetch_ready' and x['update']==step)}
identity=json.loads((root/hosts[0]/'config.json').read_text())
report={'status':'two_actual_full_updates_passed_all40ranks_exit0','source_commit':'c41886c9fe22b1698359b8182870c04aac28cb2d','source_sha256':identity['source_sha256'],'method':identity['method'],'steps':steps,'warm_adam_ranks':40,'actual_actor_captures_completed':40,'sampled_nvml_peak_mib':peak,'bounded_unsaved':True,'limitation':'Warm Adam zero-gradient allocation precedes update1 for capacity stress; this is not a saved production run.'}
(root/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='method'},indent=2))
