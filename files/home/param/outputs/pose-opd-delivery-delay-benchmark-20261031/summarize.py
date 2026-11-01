import json
from pathlib import Path
p=Path(__file__).parent
summary={}
for name in ('baseline','delayed'):
 doc=json.loads((p/name/'result.json').read_text())
 r=doc['runs'][0]
 t=r['runtime_timings']
 summary[name]={'wall_seconds':r['wall_seconds'],'generation_seconds':t['generation_seconds'],'bootstrap_seconds':t['bootstrap_seconds'],'runtime_total_seconds':t['total_seconds'],'student_frames':t['student_frames'],'teacher_backbone_per_refresh':t['teacher_backbone_per_refresh'],'teacher_forced_tower_per_refresh':t['teacher_forced_tower_per_refresh'],'delivery_delay_samples':r['delivery_delay_samples'],'delay_applies_from_sample':t['delay_applies_from_sample'],'memory':r['memory'],'audit':r['audit'],'failures':doc['failures']}
for name in ('baseline','delayed'):
 entry=summary[name]
 prefill=min(entry['audit']['prefill_frames'])
 factor=(1500-prefill)/(750-prefill)
 entry['two_minute_linear_estimate']={'generated_frame_ratio':factor,'generation_seconds':entry['generation_seconds']*factor,'rollout_seconds':entry['generation_seconds']*factor+entry['wall_seconds']-entry['generation_seconds']}
for field in ('wall_seconds','generation_seconds'):
 a,b=(summary[n][field] for n in ('baseline','delayed'))
 summary[field+'_comparison']={'seconds_saved':a-b,'percent_reduction':100*(1-b/a),'speedup':a/b}
print(json.dumps(summary,indent=2))
(p/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
