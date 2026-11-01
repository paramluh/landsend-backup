from pathlib import Path
import json,time,urllib.request
BASE='https://awarded-slope-decimal-easier.trycloudflare.com/'
OUT=Path('/home/param/websites/pose-mono-dcp227-deploy/qa')
def read(path):
 with urllib.request.urlopen(BASE+path,timeout=15) as r:return json.load(r)
for i in range(120):
 d=read('health')
 if d.get('monoplex_ready'):break
 if i%10==0:print('WAITING_FINAL_GPU_WORKER',d.get('monoplex_ready'),flush=True)
 time.sleep(2)
else:raise RuntimeError('Final worker not ready')
req=json.loads((OUT/'generation-request.json').read_text())['request'];req.update(seed=2272027,clip_s=2,label='clip_s=2 pose_temp=0.8');req['ui'].update(seed=2272027,clip_s=2)
post=urllib.request.Request(BASE+'api/gen',data=json.dumps(req).encode(),headers={'Content-Type':'application/json'},method='POST')
with urllib.request.urlopen(post,timeout=20) as r:answer=json.load(r)
rid=answer['run_id'];record={'run_id':rid,'request':req,'worker_at_request':d};(OUT/'generation-final-gpu.json').write_text(json.dumps(record,indent=2)+'\n');print('FINAL_GPU_GENERATION',rid,flush=True)
for i in range(120):
 status=read('api/run/'+rid)
 if status.get('state') in ('done','error','failed','cancelled'):
  record['result']=status;break
 time.sleep(2)
else:raise RuntimeError('Generation timed out')
(OUT/'generation-final-gpu.json').write_text(json.dumps(record,indent=2)+'\n')
print('FINAL_GPU_RESULT',json.dumps({k:status.get(k) for k in ('run_id','state','model_tag','seed','ms_per_frame','headline','error','render')}),flush=True)
assert status['state']=='done',status
assert status['summary']['frames']==51 and status['summary']['params']['pose_temp']==.8
