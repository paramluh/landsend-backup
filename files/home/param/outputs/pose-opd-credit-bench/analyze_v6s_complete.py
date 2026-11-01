from pathlib import Path
import collections,csv,datetime as dt,hashlib,json,math,statistics as st
import numpy as np
root=Path('/home/param/outputs/pose-opd-credit-bench/v6s-logs')
hosts=['landsend-2','landsend-3','landsend-4','landsend-5','landsend-12']
def stats(vals):
 a=np.asarray(list(vals),dtype=np.float64)
 return dict(count=len(a),mean=float(a.mean()),median=float(np.median(a)),p95=float(np.quantile(a,.95)),p99=float(np.quantile(a,.99)),max=float(a.max()),min=float(a.min()))
events=[];manifest=[];gpupeak=0
for h in hosts:
 p=root/h
 if json.loads((p/'supervisor/finished.json').read_text())['exit_code']!=0:raise ValueError(h)
 for f in p.glob('events-rank*.jsonl'):events += [json.loads(l) for l in f.read_text().splitlines()]
 for row in csv.reader((p/'gpu-samples.csv').read_text().splitlines()):gpupeak=max(gpupeak,int(row[2]))
 for f in p.rglob('*'):
  if f.is_file():manifest.append(dict(path=str(f.relative_to(root)),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
identity=json.loads((root/hosts[0]/'config.json').read_text())
metrics=json.loads((root/hosts[0]/'metrics.jsonl').read_text().splitlines()[-1])
profiles=[r for r in events if r['event']=='profile']
trials={n:[r for r in events if r['event']=='replay_benchmark' and r['trial']==n] for n in [0,1]}
policy={n:[r for r in events if r['event']=='policy_agreement' and r['trial']==n] for n in [0,1]}
assert len(profiles)==40 and all(len(v)==40 for v in [*trials.values(),*policy.values()])
probes={n:[dict(p,rank=r['rank']) for r in policy[n] for c in r['categorical'] for p in c['probes']] for n in policy}
assert all(c['requested']==c['captured'] and not c['missing'] for rows in policy.values() for r in rows for c in r['categorical'])
fields=['kl_actor_to_replay','kl_replay_to_actor','logit_mean_abs_difference','logit_max_abs_difference']
def probestat(rows):
 return {key:stats(r[key] for r in rows) for key in fields}
probe_groups={}
for stream in ['text','audio','pose']:
 rows=[r for r in probes[1] if r['stream']==stream]
 probe_groups[stream]={'all':probestat(rows),'early':probestat([r for r in rows if r['model_frame']<200]),'tail':probestat([r for r in rows if r['model_frame']>=200]),'codebooks':{str(cb):probestat([r for r in rows if r['codebook']==cb]) for cb in sorted({r['codebook'] for r in rows})},'worst_kl':max(rows,key=lambda r:r['kl_actor_to_replay'])}
key=lambda p:(p['rank'],p['row_slot'],p['stream'],p['model_frame'],p['codebook'])
a={key(p):p for p in probes[0]};b={key(p):p for p in probes[1]}
assert a.keys()==b.keys()
probe_repacking={f:max(abs(a[k][f]-b[k][f]) for k in a) for f in [*fields,'sampled_logprob_signed_delta']}
sampled={}
for stream in ['text','audio','pose']:
 rows=[r['streams'][stream] for r in policy[1]];cnt=sum(r['count'] for r in rows)
 sampled[stream]={'count':cnt,'mean_abs':sum(r['mean_abs']*r['count'] for r in rows)/cnt,'mean_signed':sum(r['mean_signed']*r['count'] for r in rows)/cnt,'max_abs':max(r['max_abs'] for r in rows),'worst':max(rows,key=lambda r:r['max_abs'])['worst'],'codebooks':{}}
 for cb in rows[0].get('codebooks',{}):
  c=[r['codebooks'][cb] for r in rows];nc=sum(r['count'] for r in c)
  sampled[stream]['codebooks'][cb]={'count':nc,'mean_abs':sum(r['mean_abs']*r['count'] for r in c)/nc,'max_abs':max(r['max_abs'] for r in c)}
stages={}
for trial in [0,1]:
 byrank=collections.defaultdict(lambda:collections.defaultdict(float))
 for row in events:
  if row['event']=='replay_stage_profile' and row['trial']==trial:
   for k,v in row['stages'].items():byrank[row['rank']][k]+=v['cuda_current_stream_ms']/1000
 names=byrank[0].keys()
 stages[str(trial)]={'physical_microbatch':2 if trial==0 else 4,'rank0_seconds':dict(byrank[0]),'across40ranks_seconds':{name:stats(vals[name] for vals in byrank.values()) for name in names}}
trial_summary={str(n):{'physical_microbatch':2 if n==0 else 4,'replay_seconds':stats(r['replay_seconds'] for r in values),'peak_allocated_gib':max(r['peak_allocated_gib'] for r in values),'peak_reserved_gib':max(r['peak_reserved_gib'] for r in values),'global_gradient':values[0]['gradient'],'mean_rank_loss':st.mean(r['metrics']['loss/direct'] for r in values)} for n,values in trials.items()}
g0=trial_summary['0']['global_gradient'];g1=trial_summary['1']['global_gradient']
actor_stages=next(r for r in events if r['event']=='actor_graph_profile' and r['rank']==0)
report=dict(run='pose-opd-bench-actor8-fp32-60s-v6s',status='all40ranks_exit0_one_update_two_replay_candidates_same_trajectories',source_sha256=identity['source_sha256'],method=identity['method'],benchmark_prefix_frames=identity['benchmark_prefix_frames'],rank0_total_seconds=metrics['seconds'],rollout_seconds=stats(r['rollout_seconds'] for r in profiles),replay_trials=trial_summary,stage_profiles=stages,actor_stages_rank0=actor_stages,gradient_comparison={'l2_relative_difference':abs(g1['l2']-g0['l2'])/g0['l2'],'max_abs_relative_difference':abs(g1['max_abs']-g0['max_abs'])/g0['max_abs'],'sum_absolute_difference':abs(g1['sum']-g0['sum']),'interpretation':'Same loss to roundoff and norm agreement are checksums, not elementwise gradient equality; signed sums can cancel.'},categorical_probe_count=len(probes[1]),categorical_probe_all_captured=True,categorical_probe_groups=probe_groups,categorical_probe_changes_between_student_batch2_and4=probe_repacking,sampled_logprob_first128_frames=sampled,gpu_peak_sampled_mib=gpupeak,metrics=metrics,limitations=['750-frame diagnostic prefixes only; no fullrow long-history claim.','FP32 depthattention avoids P rounding but fullactor/replay still uses different BF16operationgeometry.','Probes cover frames27/56/120 and finaltext/media frames749/751, codebooks0/14/31; not everyfullcategoricaldistribution.','Replaycandidate2runsfirst and mayinclude first-use compilation/allocation overhead; candidate4runssecond.','CUDAevent spans include dependencies and overlappingFSDP communication, not isolated kernelcosts.','Thisrun predates independentteacherbatch4/CFG8 implementation.'],provenance=manifest)
(root/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['rank0_total_seconds','rollout_seconds','replay_trials','gradient_comparison','categorical_probe_count','categorical_probe_changes_between_student_batch2_and4','gpu_peak_sampled_mib']},indent=2))
for stream in ['text','audio','pose']:
 print(stream,json.dumps({k:probe_groups[stream][k]['kl_actor_to_replay'] for k in ['all','early','tail']}))
print('stage rank0bs4',json.dumps(stages['1']['rank0_seconds']))
