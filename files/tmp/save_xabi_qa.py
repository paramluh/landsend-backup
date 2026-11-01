import json,time
from pathlib import Path
p=Path('/home/param/runs/pose-runs/pose-jump-audit-20261031-xabi-nocfg/positive-qa')
notes={
'rollout-000686':'Large isolated head/shoulder and arm/hand pose rearrangement. Actor is already far right and partially off canvas, so visible-actor quality complicates perceptual interpretation; not the clearest representative example.',
'rollout-000968':'Strong isolated head/shoulder step with hand appearance. Head12.50px,shoulders5.09px in one40ms interval; immediate motion before/after much lower. Clearest centrally framed strong pose example of these five.',
'rollout-000642':'Abrupt head/shoulder onset with hand appearance and several settling frames, followed by low motion. Main head displacement6.92px; shoulder peak follows one frame later.',
'rollout-000992':'Isolated head/shoulder step with hand appearance and low nearby motion. Head6.58px,shoulders2.61px; torso displacement remains smaller than head.',
'rollout-000562':'Abrupt head/shoulder step with hand appearance, preceded by a short ramp and followed by brief settling pulses. Quiet broader context; torso movement smaller than head.'}
rows=json.loads((p/'selection.json').read_text());td=json.loads((p.parent/'token-diagnosis.json').read_text())
for r in rows:
 d=json.loads((p/'runs'/r['run_id']/'detection.json').read_text());e=next(x for x in d['events'] if x['frame']==r['frame']);t=next(x for x in td['events'] if x['run_id']==r['run_id'] and x['frame']==r['frame'])
 r.update(review_note=notes[r['run_id']],metrics=e['metrics'],isolation=e['isolation'],abrupt_onset=e['abrupt_onset'],token_transition=t,contact_sheet=f"{r['run_id']}-{r['frame']}.png",confirmed_avatar_defect=None)
out={'created':time.time(),'detector_sha256':td['detector_sha256'],'review_scope':'Five highest-score distinct runs inspected through six pose snapshots at actual detected timestamps, per-frame motion within2s, and full clip motion traces. No avatar playback and no representative precision estimate.','conclusion':'Four centrally framed examples show isolated broad pose changes with hand appearance; highest-score example is partly off canvas. Do not interpret selected five as a cohort-wide perceptual defect rate. Retain frozen v3 candidates for rendered review.','events':rows}
(p/'review-findings.json').write_text(json.dumps(out,indent=2)+'\n')
