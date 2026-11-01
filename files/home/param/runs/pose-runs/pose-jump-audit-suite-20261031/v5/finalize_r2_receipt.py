import json,hashlib
from collections import Counter
from pathlib import Path
p=Path('/home/param/runs/pose-runs/pose-jump-audit-suite-20261031/v5');out=[]
for suffix in ['nocfg','xabi-nocfg','aloe-raw-nocfg']:
 root=Path('/home/param/runs/pose-runs/pose-jump-audit-20261031-'+suffix);src=root/'strict-review-v5.json';j=json.loads(src.read_text())
 assert j['analyzed_rollouts']==1000
 assert {s['run_id'] for s in j['sources']}=={f'rollout-{i:06d}' for i in range(1000)}
 assert j['checker_sha256']=='e4512ced382604247e86d9bc3b0ade779c7354e78b0407197c04e91005800b2a'
 (root/'strict-review-v5-r2.json').write_bytes(src.read_bytes())
 old=j['old_pool_review'];reasons=Counter(reason for row in old['rows'] if not row['retained'] for reason in row['rejection_reasons'])
 out.append({'cohort':suffix,'version':j['version'],'checker_sha256':j['checker_sha256'],'summary_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'analyzed_rollouts':j['analyzed_rollouts'],'eligible_frames':j['eligible_frames'],'candidates':[{k:e[k] for k in ['run_id','frame','time_s','score']} for e in j['events']],'old_pool_events':old['events'],'old_pool_retained':old['retained'],'old_pool_rejected':old['rejected'],'old_pool_rejection_reason_counts':dict(reasons),'source_manifest_unique_runs_verified':1000})
(p/'r2-cohort-receipt.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
