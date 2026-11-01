from pathlib import Path
import collections,csv,datetime as dt,hashlib,json,statistics as st
import numpy as np
root=Path('/home/param/outputs/pose-opd-credit-bench/v8-logs')
hosts=['landsend-2','landsend-3','landsend-4','landsend-5','landsend-12']
def stats(x):
 a=np.asarray(list(x),dtype=np.float64)
 return dict(count=len(a),mean=float(a.mean()),median=float(np.median(a)),max=float(a.max()),min=float(a.min()))
events=[];samples=[];manifest=[]
for hi,host in enumerate(hosts):
 p=root/host
 assert json.loads((p/'supervisor/finished.json').read_text())['exit_code']==0
 for f in p.glob('events-rank*.jsonl'):events.extend(json.loads(line) for line in f.read_text().splitlines())
 for r in csv.reader((p/'gpu-samples.csv').read_text().splitlines()):
  stamp=dt.datetime.strptime(r[0],'%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=dt.timezone.utc).timestamp()
  samples.append(dict(time=stamp,rank=hi*8+int(r[1]),mib=int(r[2]),util=int(r[3])))
 for f in p.rglob('*'):
  if f.is_file():manifest.append(dict(path=str(f.relative_to(root)),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
identity=json.loads((root/hosts[0]/'config.json').read_text());plan=json.loads((root/hosts[0]/'plan.json').read_text())
profiles=[e for e in events if e['event']=='profile'];warm=[e for e in events if e['event']=='optimizer_warmup'];synthetic=[e for e in events if e['event']=='synthetic_shape_inputs']
assert len(profiles)==len(warm)==len(synthetic)==40
assert all(e['synthetic_shape_only'] and e['actor_instances']==0 for e in profiles)
assert all(e['zero_gradient_warmup'] for e in warm)
assert all(e['rows']==8 and e['actor_executed']==False and e['on_policy']==False for e in synthetic)
assert not [e for e in events if e['event']=='phase' and e.get('phase')=='actor_build_begin']
assert not [e for e in events if e['event']=='policy_agreement']
assert plan['shape_spec']['conversation_frames']==1500 and plan['shape_spec']['prefix_frames']==134 and plan['shape_spec']['total_frames']==1636
bench=[e for e in events if e['event']=='replay_benchmark'];assert len(bench)==40 and all(e['trial']==0 for e in bench)
assert all(e['physical_microbatch']==8 and e['decoder_chunk_frames']==256 and e['text_chunk_frames']==128 and e['decoder_gradient_sync']=='per_replay' and not e['checkpoint_depth_decoder'] for e in bench)
assert all(np.isfinite(e['gradient']['l2']) for e in bench)
phase_samples=collections.defaultdict(list)
for rank in range(40):
 er=[e for e in events if e['rank']==rank]
 def when(phase):return next(e['time'] for e in er if e['event']=='phase' and e.get('phase')==phase)
 windows={'preparation':(when('data_audio_begin'),when('teacher_update_begin')),'teacher':(when('teacher_update_begin'),when('teacher_update_end')),'student_replay':(when('replay_microbatch_begin'),when('replay_microbatch_end')),'optimizer':(when('optimizer_begin'),next(e['time'] for e in er if e['event']=='profile'))}
 for key,(lo,hi) in windows.items():phase_samples[key].extend(s['mib'] for s in samples if s['rank']==rank and lo<=s['time']<=hi)
def stages(event):
 rows=[e for e in events if e['event']==event];assert len(rows)==40
 return {k:stats(e['stages'][k]['cuda_current_stream_ms']/1000 for e in rows) for k in rows[0]['stages']}
metric=json.loads((root/hosts[0]/'metrics.jsonl').read_text().splitlines()[-1])
report=dict(run='pose-opd-bench-b320-static-adversarial-v8',status='one_synthetic_worstshape_update_all40ranks_exit0',source_sha256=identity['source_sha256'],method=identity['method'],shape=plan['shape_spec'],independent_maxima=plan['independent_maxima'],warm_adam_ranks=len(warm),actual_actor_instances=0,on_policy=False,teacher_segments=dict(audio=8,pose_cfg=16,positions_each=1636),teacher_depth_chunk=256,pose_cfg_depth_chunk=512,phase_seconds={k:stats(e[k] for e in profiles) for k in ['preparation_seconds','rollout_seconds','teacher_seconds','student_replay_seconds','optimizer_seconds']},sampled_nvml_peak_mib=max(s['mib'] for s in samples),sampled_nvml_phase_peak_mib={k:max(v) for k,v in phase_samples.items()},student_peak_allocated_gib=max(e['peak_allocated_gib'] for e in bench),student_peak_reserved_gib=max(e['peak_reserved_gib'] for e in bench),teacher_stages=stages('teacher_stage_profile'),student_stages=stages('replay_stage_profile'),gradient=bench[0]['gradient'],metrics=metric,limitations=['Synthetic prefill-cycle B continuation, real A waveform and conditioning: validates maximum shape/capacity/finite gradients only, not on-policy quality.','NVML sampled every1s and can miss subsecond peaks; PyTorch allocation peaks supplement it.','One unsaved benchmark update; no production checkpoint.'],provenance=manifest)
(root/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['method','provenance','teacher_stages','student_stages','metrics']},indent=2))
