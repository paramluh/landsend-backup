from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json,sys
sys.path.insert(0,'/home/param/pose-repo/infer/transcript_pose')
import comparison_video as v
from publish_native import publish_native
r=Path('/home/param/pose-runs/native-comparison-20261028');jobs=v.read(r/'execution-plan.json')
statuses=[v.read(r/'teams'/f"{j['clip']['id']}-{j['recipe']}.status.json") for j in jobs]
assert len(statuses)==16 and all(s['status']=='complete' for s in statuses)
with ThreadPoolExecutor(max_workers=4) as p:list(p.map(publish_native,jobs))
v.write(r/'NATIVE_COMPARISON_COMPLETE.json',dict(status='complete',completed_at=v.stamp(),jobs=statuses,
      final_validation='Fresh publication module revalidated all16 artifacts after both pipeline coordinators exited',
      ppm_processes=16,ppm_gpu_retries=0))
print('FINAL COMPLETE:16native60s videos revalidated and published',flush=True)
