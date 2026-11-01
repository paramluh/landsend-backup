from pathlib import Path
import collections,csv,datetime as dt,hashlib,json,statistics as st
import numpy as np
root=Path('/home/param/outputs/pose-opd-credit-bench/v7-logs')
hosts=['landsend-2','landsend-3','landsend-4','landsend-5','landsend-12']
def stats(vals):
 a=np.asarray(list(vals),dtype=np.float64)
 return {'count':len(a),'mean':float(a.mean()),'median':float(np.median(a)),'p95':float(np.quantile(a,.95)),'max':float(a.max()),'min':float(a.min())}
events=[];manifest=[];gpupeak=0;hostpeaks={}
for host in hosts:
 p=root/host
 if json.loads((p/'supervisor/finished.json').read_text())['exit_code']!=0:raise ValueError(host)
 for f in p.glob('events-rank*.jsonl'):events.extend(json.loads(line) for line in f.read_text().splitlines())
 samples=list(csv.reader((p/'gpu-samples.csv').read_text().splitlines()))
 hostpeaks[host]={str(i):max(int(r[2]) for r in samples if int(r[1])==i) for i in range(8)}
 gpupeak=max(gpupeak,max(hostpeaks[host].values()))
 for f in p.rglob('*'):
  if f.is_file():manifest.append({'path':str(f.relative_to(root)),'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
identity=json.loads((root/hosts[0]/'config.json').read_text());metrics=json.loads((root/hosts[0]/'metrics.jsonl').read_text().splitlines()[-1]);plan=json.loads((root/hosts[0]/'plan.json').read_text())
profiles={r['rank']:r for r in events if r['event']=='profile'}
trials={i:[r for r in events if r['event']=='replay_benchmark' and r['trial']==i] for i in range(4)}
assert len(profiles)==40 and all(len(rows)==40 for rows in trials.values())
warm=[r for r in events if r['event']=='optimizer_warmup'];assert len(warm)==40 and all(r['zero_gradient_warmup'] for r in warm)
trial_summary={}
base=trials[0][0]['gradient']
for i,rows in trials.items():
 keys=['physical_microbatch','decoder_chunk_frames','text_chunk_frames','decoder_gradient_sync','checkpoint_depth_decoder']
 trial_summary[str(i)]={**{k:rows[0][k] for k in keys},'replay_seconds':stats(r['replay_seconds'] for r in rows),'peak_allocated_gib':max(r['peak_allocated_gib'] for r in rows),'peak_reserved_gib':max(r['peak_reserved_gib'] for r in rows),'gradient':rows[0]['gradient'],'mean_rank_loss':st.mean(r['metrics']['loss/direct'] for r in rows),'gradient_l2_relative_difference_from_trial0':abs(rows[0]['gradient']['l2']-base['l2'])/base['l2']}
stages={}
for i in range(4):
 byrank=collections.defaultdict(lambda:collections.defaultdict(float))
 for r in events:
  if r['event']=='replay_stage_profile' and r['trial']==i:
   for k,v in r['stages'].items():byrank[r['rank']][k]+=v['cuda_current_stream_ms']/1000
 stages[str(i)]={'rank0_seconds':dict(byrank[0]),'across_ranks_seconds':{k:stats(v[k] for v in byrank.values()) for k in byrank[0]}}
teacher_stages={r['rank']:r for r in events if r['event']=='teacher_stage_profile'}
teacher_summary={k:stats(r['stages'][k]['cuda_current_stream_ms']/1000 for r in teacher_stages.values()) for k in teacher_stages[0]['stages']}
same_root=[r for r in events if r['event']=='actor_depth_policy_probe'];assert len(same_root)==20
same_root_summary={}
for stream in ['audio','pose']:
 same_root_summary[stream]={}
 for section in ['all31','1','14','31']:
  values=[r['streams'][stream]['all31'] if section=='all31' else r['streams'][stream]['by_codebook'][section] for r in same_root]
  same_root_summary[stream][section]={k:stats(v[k] for v in values) for k in values[0]}
 same_root_summary[stream]['root_equal_all']=all(r['streams'][stream]['root_equal'] for r in same_root)
policy=[r for r in events if r['event']=='policy_agreement' and r['trial']==3];assert len(policy)==40
probes=[dict(p,rank=r['rank']) for r in policy for c in r['categorical'] for p in c['probes']]
assert all(c['requested']==c['captured'] and not c['missing'] for r in policy for c in r['categorical'])
cat={}
for stream in ['text','audio','pose']:
 rows=[p for p in probes if p['stream']==stream]
 cat[stream]={group:{'count':len(v),'kl_actor_to_replay':stats(p['kl_actor_to_replay'] for p in v),'logit_mean_abs':stats(p['logit_mean_abs_difference'] for p in v),'logit_max_abs':max(p['logit_max_abs_difference'] for p in v)} for group,v in [('all',rows),('early',[p for p in rows if p['model_frame']<200]),('tail',[p for p in rows if p['model_frame']>=200])]}
 cat[stream]['by_codebook']={str(cb):stats(p['kl_actor_to_replay'] for p in rows if p['codebook']==cb) for cb in sorted({p['codebook'] for p in rows})}
 cat[stream]['worst_kl']=max(rows,key=lambda p:p['kl_actor_to_replay'])
generation=[]
for rank in range(0,40,2):
 rs=[r for r in events if r['rank']==rank];p=[r for r in rs if r['event']=='rollout_progress'][-1];end=next(r for r in rs if r['event']=='phase' and r.get('phase')=='actor_microbatch_end')
 generation.append({'rank':rank,'seconds':p['seconds_elapsed']+end['time']-p['time']})
phasekeys=['preparation_seconds','rollout_seconds','teacher_seconds','student_replay_seconds','optimizer_seconds']
selected_sums={str(rank):sum(row[k] for k in phasekeys) for rank,row in profiles.items()}
rollout=max(r['rollout_seconds'] for r in profiles.values());genmax=max(r['seconds'] for r in generation)
report={'run':'pose-opd-bench-b320-static-v7','status':'one_actual_update_four_fixedtrajectory_replay_variants_all40ranks_exit0','source_sha256':identity['source_sha256'],'method':identity['method'],'plan_digest':identity['plan_digest'],'benchmark_actual_total_seconds_rank0':metrics['seconds'],'selected_variant':3,'selected_variant_counterfactual_phase_sum_seconds':stats(selected_sums.values()),'counterfactual_definition':'Preparation+actor(publication/capture/probes)+sharedteacherpreparation+chosenstudentreplay+optimizer. Excludes the three other replay trials, gradient-checksum/profiler/cached-rollout-write overhead and process/model initialization. A phase-sum estimate, not a separately measured production iteration.','phases_across40ranks':{k:stats(r[k] for r in profiles.values()) for k in phasekeys},'replay_trials':trial_summary,'replay_stages':stages,'teacher_stages_across40ranks_seconds':teacher_summary,'teacher_temporal_expected_segments':{'audio':8,'pose_cfg':16,'fixed_positions_per_segment':1636},'total_rows':320,'total_generated_physical_frames':320*1475,'useful_generated_fps_per_pair_including_actor_setup':320*1475/(20*rollout),'generation_plus_readback_seconds_derived':stats(r['seconds'] for r in generation),'useful_generated_fps_per_pair_excluding_actor_setup':16*1475/genmax,'derived_generation_definition':'Finalsubmissionelapsed plus timestampgap to synchronizedactor_microbatch_end; excludesengine.start/prefill/build/publication/capture. Approximate mixedmonotonic/wall delta.','amu_reference_useful_fps_per_pair':129.294036220435,'sampled_nvml_peak_mib':gpupeak,'sampled_nvml_host_peaks_mib':hostpeaks,'warm_adam_ranks':len(warm),'full_categorical_probe_count':len(probes),'full_categorical_probes_all_captured':True,'full_actor_replay_categorical':cat,'identical_root_depth_probe':same_root_summary,'identical_root_depth_raw':same_root,'metrics':metrics,'limitations':['Same-root depth probe uses finalframeonly; fullcategoricalprobes sample early/final frames and codebooks0/14/31.','Gradient summaries are not elementwise parity and BF16batch/accumulation changes can round differently.','Four variants run in fixed order; first-use compilation/allocation can disadvantage trial0.','Oneunsaved update establishes thistestedconfiguration, not repeatedproductionthroughput orquality.'],'provenance':manifest}
(root/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['benchmark_actual_total_seconds_rank0','selected_variant_counterfactual_phase_sum_seconds','phases_across40ranks','replay_trials','sampled_nvml_peak_mib','useful_generated_fps_per_pair_including_actor_setup','useful_generated_fps_per_pair_excluding_actor_setup']},indent=2))
for stream in ['text','audio','pose']:print('categorical',stream,cat[stream]['all']['kl_actor_to_replay'])
for stream in ['audio','pose']:print('same_root',stream,same_root_summary[stream]['all31']['actor_to_replay_kl_mean'])
