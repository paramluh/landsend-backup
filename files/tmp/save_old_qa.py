import json,time
from pathlib import Path
p=Path('/home/param/runs/pose-runs/pose-jump-audit-20261031-aloe-raw-nocfg/positive-qa')
notes={
'rollout-000380':'High head/shoulder speed peak with a broader evolving pose burst and fairly irregular movement afterward. Central pose changes in the contact pair, but continuous motion cannot be excluded; v4 rejects both coincident torso onset and trajectory fit.',
'rollout-000228':'Large head/shoulder peak, but head reverses the preceding motion and nearby preceding peaks are substantial. Continuous nod/body gesture remains plausible; v4 rejects the multi-scale step criterion.',
'rollout-000139':'Abrupt face change in a comparatively quiet clip; shoulders are already moving in preceding frames and their position continues to evolve. Looks stronger in head than torso; v4 rejects its step-model preference. Avatar review needed.',
'rollout-000329':'Abrupt onset after a very quiet interval, followed by several moving frames and increased ordinary motion. Broad pose changes are present but a continuing gesture remains possible; v4 step-model criterion fails.',
'rollout-000187':'Shared face/shoulder position shift followed by low motion; moderate activity is present before the event. Passes the stricter v4 coincidence and trajectory checks, making it a useful rendered-review priority. Still not a confirmed visual defect.'}
rows=json.loads((p/'selection.json').read_text());td=json.loads((p.parent/'token-diagnosis.json').read_text());v4=json.loads((p.parent/'trajectory-review-v4.json').read_text())
for r in rows:
 d=json.loads((p/'runs'/r['run_id']/'detection.json').read_text());e=next(x for x in d['events'] if x['frame']==r['frame']);t=next(x for x in td['events'] if x['run_id']==r['run_id'] and x['frame']==r['frame']);a=next(x for x in v4['assessments'] if x['run_id']==r['run_id'] and x['frame']==r['frame'])
 r.update(review_note=notes[r['run_id']],metrics=e['metrics'],isolation=e['isolation'],abrupt_onset=e['abrupt_onset'],token_transition=t,contact_sheet=f"{r['run_id']}-{r['frame']}.png",confirmed_avatar_defect=None,trajectory_review_v4=a)
out={'created':time.time(),'detector_sha256':td['detector_sha256'],'review_scope':'Five highest-score distinct runs inspected through six pose snapshots at actual detected timestamps, per-frame motion within2s, and full clip motion traces. No avatar playback and no representative precision estimate.','conclusion':'High v3 score does not distinguish all rapid continuous gestures from visual jumps. One of these five passes the separate conservative v4 check. All original v3 windows remain in requested render pool.','events':rows}
(p/'review-findings.json').write_text(json.dumps(out,indent=2)+'\n')
