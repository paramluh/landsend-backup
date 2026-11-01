import json,time
from pathlib import Path
p=Path('/home/param/runs/pose-runs/pose-jump-audit-20261031-nocfg/positive-qa')
notes=[
('000145',474,'strong','Abrupt head/shoulder change followed by settling. Real rendered frames473/474 and overlay inspected: image follows the pose step. Visible severity is milder than original reference; one static pair is not a full perceptual video review.'),
('000375',1347,'strong','Abrupt coherent onset, then a second head/core pulse about0.12s later. Prior second quiet and later motion settles. Looks like a short multi-frame transition, not a single rigid translation.'),
('000584',359,'strong','Larger head+shoulder spike amid moderate ordinary motion and smaller nearby peaks. Clear local peak, but clean smooth-neighborhood semantics are less compelling than132/300.'),
('000132',170,'strong','Clear abrupt decoded head/shoulder step with immediate quiet before and rapid settling after; hand enters too but is not the only changing region. Strong isolated-pose candidate.'),
('000300',157,'strong','Clear isolated decoded head/shoulder step; surrounding lower-rate head motion. Strong pose-discontinuity candidate, torso movement remains smaller than head.'),
('000033',440,'random_positive_control','Small coherent abrupt head+shoulder pose step, two nearby spikes about0.2s apart. Severity compared with original whole-body skip uncertain.'),
('000076',416,'random_positive_control','Small brief head-dominant pulse with shoulder motion. Meets numeric isolation/onset criteria but whole-body perceptual severity uncertain.'),
('000261',406,'random_positive_control','Small coherent abrupt pose step. Whole clip later erratic, but actual16.24s local context is quieter. Global irregularity alone does not establish a locally false positive.'),
('000322',226,'random_positive_control','Small coherent step over about2–3frames with shoulder confidence change; severity uncertain.'),
('000322',990,'random_positive_control','Rapid brief head turn/pulse over a few frames; shoulders move less. Whole-body severity uncertain.')]
rows=[]
for rid,f,group,note in notes:
 rid='rollout-'+rid
 d=json.loads((p/'runs'/rid/'detection.json').read_text())
 e=next(x for x in d['events'] if x['frame']==f)
 rows.append({'run_id':rid,'frame':f,'time_s':e['time_s'],'selection_group':group,'review_note':note,'contact_sheet':f'{rid}-{f}.png','metrics':e['metrics'],'isolation':e['isolation'],'abrupt_onset':e.get('abrupt_onset'),'confirmed_avatar_defect':None})
out={'created':time.time(),'detector_sha256':'a9b2eb0bf2458fa0428e0ecd424a6be3b5dd298c475a8a5ae2bd1a69e51d81a6','review_scope':'Ten accepted events at their actual detected timestamps: five highest-score distinct runs plus all five events in four randomly drawn positive-control runs. Six snapshots/event, per-frame motion within2s, and full clip motion trace inspected. Actual rendered before/after pair inspected for145. Original v1 remains sanity reference. This is not a full avatar playback or precision estimate.','conclusion':'Retain v3 candidate set and review rendered windows. Strong and mild decoded discontinuities are both present. No demonstrated new implementation bug justifies changing frozen thresholds; severity of many accepted candidates remains uncertain.','events':rows,'negative_qa':'../negative-review/screening-results.json','negative_qa_interpretation':'Twenty random nonflagged runs were screened through full motion traces and strongest raw-innovation pose contacts while blinded, plus four positive controls. Three negatives remain uncertain. This did not establish a clear large missed whole-body jump, but it is not an exhaustive playback review and cannot establish recall. Random controls were followed up at exact accepted timestamps after unblinding; initial trace-only impressions should not be interpreted as proof of false positives.'}
(p/'review-findings.json').write_text(json.dumps(out,indent=2)+'\n')
print(str(p/'review-findings.json'))
