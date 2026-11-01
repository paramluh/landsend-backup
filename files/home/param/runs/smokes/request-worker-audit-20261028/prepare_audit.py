import json,os,sys,subprocess,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,'/home/param/pose-repo/infer/transcript_pose')
import request_worker as W
import request_schema as S
root=Path('/home/param/pose-runs/request-worker-audit-20261028');rows=[]
for recipe in S.RECIPES:
 job=root/f'prepare-{recipe}';job.mkdir(exist_ok=True)
 req=S.normalize_request({'recipe':recipe,'clip_id':'clip-01','duration_seconds':4,'audio_temp':0,'pose_temp':3,'audio_topk':2048,'pose_topk':17})
 clip,_=W.source_clip(req['clip_id']);cmd=W.generation_command(job,req,clip)+['--prepare-only'];env=W.environment();env['CUDA_VISIBLE_DEVICES']=''
 with (job/'prepare.log').open('w') as log:subprocess.run(list(map(str,cmd)),cwd=W.REPO,env=env,check=True,stdout=log,stderr=subprocess.STDOUT)
 meta=W.read(job/'ppm/input.json');P=meta['P'];n=S.media_frames(4)
 with np.load(job/'ppm/inputs.npz') as z:
  assert meta['frames']==52 and meta['force_text'] and meta['prefill_frames']==0 and meta['voice_clip_frames']==100
  assert np.array_equal(z['text_force'][P:P+n],np.load(clip['lane'])[:n])
  assert np.all(z['text_force'][P+n:]==151670) and np.array_equal(z['ids'][1],z['text_force'])
  assert np.all(z['depth'][:,P:]==-1) and np.all(z['pdepth'][P:]==-1)
  assert np.array_equal(np.load(job/'ppm/voice_clip_mimi.npy'),np.load(clip['voice_clip']))
  prefix={k:hashlib.sha256(np.ascontiguousarray(z[k][:,:109]).tobytes()).hexdigest() for k in ['ids','depth','present']}
 rows.append({'recipe':recipe,'P':P,'frames':n,'conditioning_P':meta['conditioning_P'],'prefix':prefix,'status':'passed'})
assert len({json.dumps(r['prefix'],sort_keys=True) for r in rows})==1
# Validate a real completed B1 output from the ongoing comparison, CPU only.
p=W.SOURCE/'outputs/01/xabi-original';req=S.normalize_request({'recipe':'xabi-original','clip_id':'clip-01'})
validation=W.validate_ppm(p,req,W.source_clip('clip-01')[0])
result={'status':'complete','gpu_work_started':False,'prepare_only':rows,'existing_B1_validation':{'path':str(p),'voice':validation['voice_codes_sha256'],'lane':validation['framewise_lane_sha256']}}
W.write(root/'prepare-audit.json',result);print(json.dumps(result,indent=2))
