import json,os,time,traceback
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
import audit_native as A
W,S=A.W,A.S
R=Path(__file__).resolve().parent
plan=W.read(R/'execution-plan.json');assert len(plan)==16
expected={(c,r) for c in S.CLIP_IDS for r in ('compact-all','xabi-original')}
assert {(j['clip_id'],j['recipe']) for j in plan}==expected
original=W.read(W.SOURCE/'selection.json');assert W.read(R/'selection.json')['clips']==original['clips']
rows={f"{j['clip_id']}/{j['recipe']}":{'status':'pending','clip_id':j['clip_id'],'recipe':j['recipe']} for j in plan}
while True:
 for job in plan:
  key=f"{job['clip_id']}/{job['recipe']}";directory=Path(job['directory'])
  if rows[key]['status']!='pending' or not (directory/'video_state.json').is_file():continue
  if W.read(directory/'video_state.json').get('status')!='complete':continue
  req=S.normalize_request({'recipe':job['recipe'],'clip_id':job['clip_id'],'text_mode':'native','duration_seconds':60,
      'seed':20261028+int(job['clip']['gallery_id'])-1,'video_seed':2026})
  clip,_=W.source_clip(job['clip_id'])
  try:
   result=A.audit(directory,req,clip);W.write(directory/'independent-audit.json',result);rows[key]=result
   print(f"PASS {key} audio corr={result['audio_vs_ppm']['same_time_correlation']:.6f}",flush=True)
  except Exception as error:
   rows[key]={'status':'failed','clip_id':job['clip_id'],'recipe':job['recipe'],'error':str(error),'traceback':traceback.format_exc()}
   print(f'FAIL {key}: {error}',flush=True)
 complete=sum(r['status']=='pass' for r in rows.values());failed=sum(r['status']=='failed' for r in rows.values())
 status='complete' if complete==16 else ('failed' if failed else 'running')
 report={'status':status,'updated_at':W.stamp(),'gpu_work_started':False,'expected':16,'passed':complete,'failed':failed,
         'source_selection_exactly_controlled_comparison':True,'rows':list(rows.values())}
 W.write(R/'independent-audit.json',report)
 if complete+failed==16:break
 time.sleep(20)
if failed:raise SystemExit(1)
