import concurrent.futures,collections,json,hashlib
from pathlib import Path
import numpy as np
BASE=Path('/home/param/runs/pose-runs');SUITE=BASE/'pose-jump-audit-suite-20261031';OUT=SUITE/'hand-independent-review';OUT.mkdir(exist_ok=True)
def inspect(job):
 root,rid,events=job;p=Path(root)/'runs'/rid/'pose_overlay.json';src=p.read_bytes();pose=np.asarray(json.loads(src)['frames'],float);out=[]
 for e in events:
  f=e['frame']
  for g in e['groups']:
   if not g['passes'] or 'hand' not in g['group']:continue
   ids=g['metrics']['anchor_indices'];pre=pose[f-8:f,ids];post=pose[f:f+13,ids];xy=pose[:,ids,:2]*[256,480]
   inside=lambda a:np.all((a[:,:,:2]>=0)&(a[:,:,:2]<=1),axis=-1)
   preinside=inside(pre);postinside=inside(post)
   step=g['metrics']['single_frame_step_px'];v=np.sqrt(np.mean(np.sum(np.diff(xy,axis=0)**2,axis=-1),axis=-1))
   m={'source_frame_all_anchors_inframe':bool(preinside[-1].all()),'pre8_inside_fraction':float(preinside.mean()),'post13_inside_fraction':float(postinside.mean()),'pre_conf_median':float(np.median(pre[:,:,2])),'destination_conf_median':float(np.median(post[0,:,2])),'post13_conf_median':float(np.median(post[:,:,2])),'post13_medium_high_fraction':float(np.mean(post[:,:,2]>=.4)),'previous3_max_step_fraction':float(max(v[f-4:f-1])/step),'next3_max_step_fraction':float(max(v[f:f+3])/step),'all_hand_pre8_inside_fraction':float(inside(pose[f-8:f,13:34 if g['group']=='left_hand' else 55] if g['group']=='left_hand' else pose[f-8:f,34:55]).mean())}
   m['review_priority']=g['score']*(m['pre8_inside_fraction']**6)*m['post13_inside_fraction']*min(1.,.4/max(m['next3_max_step_fraction'],1e-9))
   out.append({'root':root,'run_id':rid,'frame':f,'time_s':f/25,'group':g['group'],'anchor_indices':ids,'raw_score':g['score'],'metrics':m,'original_group_metrics':g['metrics'],'trajectory_fits':g['trajectory_fits'],'pose_source_sha256':hashlib.sha256(src).hexdigest()})
 return out
def main():
 jobs=[]
 for suffix in ['-aloe-raw-nocfg','-aloe-raw-nocfg-cohort2']:
  root=BASE/('pose-jump-audit-20261031'+suffix);report=json.loads((root/'hand-jump-review-r5.json').read_text());runs=collections.defaultdict(list)
  for e in report['events']:runs[e['run_id']].append(e)
  jobs.extend((str(root),rid,events) for rid,events in runs.items())
 rows=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:
  for r in pool.map(inspect,jobs,chunksize=5):rows.extend(r)
 eligible=[r for r in rows if r['metrics']['source_frame_all_anchors_inframe'] and r['metrics']['pre8_inside_fraction']>=.95 and r['metrics']['post13_inside_fraction']>=.95]
 ranked=sorted(eligible,key=lambda r:r['metrics']['review_priority'],reverse=True)
 selected=[];seen=set()
 for r in ranked:
  key=(r['root'],r['run_id'],r['frame'])
  if key in seen:continue
  seen.add(key);selected.append(r)
  if len(selected)==20:break
 for rank,r in enumerate(selected):
  full=json.loads((Path(r['root'])/'runs'/r['run_id']/'pose_overlay.json').read_text());f=r['frame'];start=max(0,f-30);stop=min(len(full['frames']),f+31)
  window={'frames':full['frames'][start:stop],'start_frame':start,'event_frame':f,'fps':25,'assessment':r}
  name=f'{rank+1:02d}-'+('cohort2-' if r['root'].endswith('cohort2') else 'cohort1-')+r['run_id']+f'-f{f}.json'
  (OUT/name).write_text(json.dumps(window)+'\n');r['window_json']=str(OUT/name)
 result={'scope':'Independent visual review ranking of existing accepted oldAloe hand candidates; no detector result altered. Source coordinates must stay inframe for reviewpriority; low sourceconfidence is allowed. Rankings are not visible-defect labels.','hand_groups_examined':len(rows),'inframe_review_groups':len(eligible),'top20':selected,'all_group_measurements':rows}
 (OUT/'ranked-inframe.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'groups':len(rows),'eligible':len(eligible),'top':[{k:r[k] for k in ['root','run_id','frame','group','metrics']} for r in selected[:8]]}),flush=True)
if __name__=='__main__':main()
