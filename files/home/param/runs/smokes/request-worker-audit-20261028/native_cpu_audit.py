import json,sys,tempfile,shutil
from pathlib import Path
import numpy as np
sys.path.insert(0,'/home/param/pose-repo/infer/transcript_pose')
import request_schema as S
import request_worker as W
checks=[]
def reject(fn):
 try:fn()
 except ValueError:return
 raise AssertionError('Invalid input was accepted')
base={'recipe':'compact-all','clip_id':'clip-01'}
assert S.normalize_request(base)['text_mode']=='forced'
for recipe in ('compact-all','xabi-original'):
 r=S.normalize_request({**base,'recipe':recipe,'text_mode':'native','text_temp':0,'text_topk':256});assert r['text_mode']=='native'
for change in [{'recipe':'aligned-media','text_mode':'native'},{'text_mode':'unknown'},{'text_mode':None},{'text_temp':False},{'text_temp':float('nan')},{'text_temp':3.01},{'text_topk':257},{'text_topk':-1},{'text_topk':True}]:reject(lambda:S.normalize_request({**base,**change}))
checks.append('Legacy forced defaults, native allowed recipes, text sampling bounds and aligned-native rejection')
r=Path('/home/param/pose-runs/native-inference-audit-20261028');clip,_=W.source_clip('clip-01');prepared=[]
for recipe,mode in [('aligned-media','forced'),('compact-all','forced'),('compact-all','native'),('xabi-original','forced'),('xabi-original','native')]:
 p=r/f'{recipe}-{mode}-0';req=S.normalize_request({**base,'recipe':recipe,'text_mode':mode,'duration_seconds':4})
 result=W.validate_inputs(p,req,clip,W.read(p/'input.json'))
 assert (result['supplied_transcript'] is None)==(recipe=='xabi-original' and mode=='native')
 assert (result['framewise_lane_sha256'] is None)==(mode=='native')
 prepared.append({'recipe':recipe,'text_mode':mode,'passed':True,'supplied_transcript':result['supplied_transcript']})
checks.append('Actual five prepare-only modes: exact prefixes, correct supplied transcript and no native generation text')
with tempfile.TemporaryDirectory() as temp:
 dst=Path(temp);src=r/'compact-all-native-0';shutil.copy(src/'input.json',dst/'input.json')
 with np.load(src/'inputs.npz') as data:arrays={k:data[k].copy() for k in data.files}
 req=S.normalize_request({**base,'text_mode':'native','duration_seconds':4});meta=W.read(src/'input.json');P=meta['P']
 arrays['ids'][1,P+3]=100;np.savez(dst/'inputs.npz',**arrays)
 reject(lambda:W.validate_inputs(dst,req,clip,meta))
 arrays['ids'][1,P+3]=151670;arrays['ids'][1,110]=100;np.savez(dst/'inputs.npz',**arrays)
 reject(lambda:W.validate_inputs(dst,req,clip,meta))
checks.append('Reject injected timed-word leakage and incorrect compact upfront prefix')
p=Path('/home/param/pose-runs/inference-requests-20261028/jobs/489344dc0d8d422db29a2c295407314c')
req=S.normalize_request(W.read(p/'request.json'));output=W.validate_ppm(p,req,clip)
assert req['text_mode']=='forced' and output['sampling']['cb0_topk']==7 and output['sampling']['pose_topk']==11
checks.append('Existing actual4s forced HTTP request passes upgraded full PPM validator')
v=W.read(W.SOURCE/'outputs/01/xabi-original/diffusion/generation/status.json');schedule=W.validate_video_receipt(v,752,2026)
for key,bad in [('checkpoint','wrong'),('reference_keep',[True]*8+[False]),('stabilizer_total_steps',4),('resolver_steps',2),('renoising',True),('runtime_audio_guidance_scale',2.0)]:reject(lambda:W.validate_video_receipt({**v,key:bad},752,2026))
checks.append('Real step2911 receipt passes; wrong checkpoint/references/schedule/guidance rejected')
report={'status':'complete','gpu_work_started':False,'checks':checks,'prepared':prepared,'existing_forced_job':p.name,'video_schedule':schedule}
W.write(Path('/home/param/pose-runs/request-worker-audit-20261028/native-cpu-audit.json'),report);print(json.dumps(report,indent=2))
