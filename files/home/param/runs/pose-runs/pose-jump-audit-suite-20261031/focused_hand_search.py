import json,hashlib,collections,concurrent.futures
from pathlib import Path
import numpy as np
BASE=Path('/home/param/runs/pose-runs');SUITE=BASE/'pose-jump-audit-suite-20261031';OUT=SUITE/'hand-focused-review';OUT.mkdir(exist_ok=True)
EDGES=np.array([(a,b) for k in [1,5,9,13,17] for a,b in [(0,k),(k,k+1),(k+1,k+2),(k+2,k+3)]])
def geom(x):
 e=np.linalg.norm(x[:,EDGES[:,1]]-x[:,EDGES[:,0]],axis=-1);s=np.linalg.svd(x-x.mean(1,keepdims=True),compute_uv=False)
 return {'median_bone_px':float(np.median(e)),'bone_p90_px':float(np.percentile(e,90)),'max_to_median_bone':float(np.max(e)/max(np.median(e),.01)),'minor_major_axis_ratio':float(np.median(s[:,1]/np.maximum(s[:,0],.01))),'inside_fraction':float(np.mean(np.all((x>=[0,0])&(x<=[256,480]),axis=-1))),'edge_fraction':float(np.mean((x[:,:,0]<4)|(x[:,:,0]>252)|(x[:,:,1]<4)|(x[:,:,1]>476))),'extent_px':float(np.median(np.linalg.norm(np.ptp(x,axis=1),axis=-1)))}
def work(job):
 root,rid,events=job;pp=Path(root)/'runs'/rid/'pose_overlay.json';raw=pp.read_bytes();p=np.asarray(json.loads(raw)['frames'],float);xy=p[:,:,:2]*[256,480];rows=[]
 comp=Path(root)/'rendered'/rid/'complete.json';av=json.loads(comp.read_text()) if comp.exists() else None
 for event in events:
  f=event['frame']
  for g in event['groups']:
   if not g['passes'] or 'hand' not in g['group']:continue
   ids=g['metrics']['anchor_indices'];hids=np.arange(13,34) if g['group']=='left_hand' else np.arange(34,55);a=xy[:,ids];h=xy[:,hids];step=g['metrics']['single_frame_step_px'];v=np.sqrt(np.mean(np.sum(np.diff(a,axis=0)**2,axis=-1),axis=-1));hv=np.sqrt(np.mean(np.sum(np.diff(h,axis=0)**2,axis=-1),axis=-1));delta=h[f]-h[f-1];norm=np.linalg.norm(delta,axis=-1);mean=delta.mean(0);coher=float(np.linalg.norm(mean)/max(np.sqrt(np.mean(norm**2)),.01));pre=geom(h[f-4:f]);post=geom(h[f:f+4]);
   m={'pre':pre,'post':post,'next3_step_ratio':float(max(v[f:f+3])/max(step,.01)),'previous3_step_ratio':float(max(v[f-4:f-1])/max(step,.01)),'whole_hand_next3_step_ratio':float(max(hv[f:f+3])/max(hv[f-1],.01)),'whole_hand_previous3_step_ratio':float(max(hv[f-4:f-1])/max(hv[f-1],.01)),'whole_hand_jump_px':float(hv[f-1]),'whole_hand_median_joint_step_px':float(np.median(norm)),'whole_hand_translation_coherence':coher,'whole_hand_before_conf_median':float(np.median(p[f-4:f,hids,2])),'whole_hand_destination_conf_median':float(np.median(p[f,hids,2])),'chosen_before_conf_median':float(np.median(p[f-4:f,ids,2])),'chosen_destination_conf_median':float(np.median(p[f,ids,2])),'wrist_step_px':float(norm[0]),'moving_hand_joint_count_half_rms':int(np.sum(norm>=hv[f-1]/2)),'chosen_inframe_before_fraction':float(np.mean(np.all((p[f-8:f,ids,:2]>=0)&(p[f-8:f,ids,:2]<=1),axis=-1)))}
   shape=min(pre['minor_major_axis_ratio'],post['minor_major_axis_ratio'])**.5*min(1,pre['median_bone_px']/4,post['median_bone_px']/4)*max(.001,1-max(pre['edge_fraction'],post['edge_fraction']))**3*min(pre['inside_fraction'],post['inside_fraction'])**3/max(1,pre['max_to_median_bone']/4,post['max_to_median_bone']/4)
   quiet=min(1,.2/max(m['next3_step_ratio'],.001))*min(1,.2/max(m['whole_hand_next3_step_ratio'],.001))*min(1,.4/max(m['whole_hand_previous3_step_ratio'],.001))
   rank=shape*quiet*coher*min(1,len(ids)/9)*min(1,step/20)*(.5+.5*min(1,m['chosen_before_conf_median']/.4))
   rows.append({'root':root,'run_id':rid,'frame':f,'time_s':f/25,'group':g['group'],'anchor_indices':ids,'raw_score':g['score'],'shape_review_score':rank,'metrics':m,'original_group_metrics':g['metrics'],'source_pose_sha256':hashlib.sha256(raw).hexdigest(),'has_any_existing_avatar_windows':av is not None})
 return rows
if __name__=='__main__':
 jobs=[]
 for suffix in ['-nocfg','-xabi-nocfg','-aloe-raw-nocfg']:
  for c in ['', '-cohort2']:
   root=BASE/('pose-jump-audit-20261031'+suffix+c);d=json.loads((root/'hand-jump-review-r5.json').read_text());rs=collections.defaultdict(list)
   for e in d['events']:rs[e['run_id']].append(e)
   jobs.extend((str(root),rid,es) for rid,es in rs.items())
 rows=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=6) as pool:
  for i,r in enumerate(pool.map(work,jobs,chunksize=5)):
   rows.extend(r)
   if i%200==0:print(json.dumps({'source_runs_done':i,'total_source_runs':len(jobs),'groups':len(rows)}),flush=True)
 rows.sort(key=lambda x:x['shape_review_score'],reverse=True)
 old=json.loads((SUITE/'hand-independent-review/visual-assessment.json').read_text());seen=set()
 for n in old['images_inspected']:
  if '-rollout-' in n:
   stem=Path(n).stem;parts=stem.split('-');co='-cohort2' if 'cohort2' in parts else '';rid='rollout-'+parts[parts.index('rollout')+1];f=int(parts[-1][1:]);seen.add((str(BASE/('pose-jump-audit-20261031-aloe-raw-nocfg'+co)),rid,f))
 selected=[]
 # Bounded review list, rank only; detector outputs remain unchanged.
 for r in rows:
  key=(r['root'],r['run_id'],r['frame'])
  if key in seen:continue
  seen.add(key);selected.append(r)
  if len(selected)==28:break
 for i,r in enumerate(selected,1):
  full=json.loads((Path(r['root'])/'runs'/r['run_id']/'pose_overlay.json').read_text());f=r['frame'];s=f-30;e=f+31;model='aloe357' if 'aloe-raw' in r['root'] else 'xabi200' if 'xabi' in r['root'] else 'mono355';co='cohort2' if r['root'].endswith('cohort2') else 'cohort1';path=OUT/f'{i:02d}-{model}-{co}-{r["run_id"]}-f{f}.json';path.write_text(json.dumps({'frames':full['frames'][s:e],'start_frame':s,'event_frame':f,'fps':25,'assessment':r})+'\n');r['window_json']=str(path)
 (OUT/'ranking.json').write_text(json.dumps({'scope':'Review ranking only; shape/quietness factors do not change detector labels. Bounded manual review28cases across allmodels.','source_runs_examined':len(jobs),'groups_examined':len(rows),'selected':selected,'all_groups':rows},indent=2)+'\n');print(json.dumps({'done':True,'groups':len(rows),'selected':[{k:r[k] for k in ['root','run_id','frame','group','shape_review_score','metrics']} for r in selected[:5]]}),flush=True)
