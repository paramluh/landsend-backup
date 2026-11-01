import contextlib,json,os,sys,tempfile,uuid
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,'/home/param/pose-repo/infer/transcript_pose')
import request_schema as S
import request_worker as W
checks=[]
def ok(name):checks.append(name)
def rejects(fn):
 try:fn()
 except ValueError:return
 raise AssertionError('Invalid request accepted')
base={'recipe':'aligned-media','clip_id':'clip-01'}
for recipe in S.RECIPES:
 for i,clip in enumerate(S.CLIP_IDS):
  x=S.normalize_request({'recipe':recipe,'clip_id':clip});assert x['seed']==20261028+i and len(x)==12
ok('24 model/clip combinations and defaults')
for field,values in {'duration_seconds':[3,61,True,4.0,'4',None], 'seed':[-1,2**31,True,1.0], 'video_seed':[-1,2**31,False], 'audio_topk':[-1,2049,False,1.0], 'pose_topk':[-1,2049,True], 'audio_temp':[-.001,3.001,float('nan'),float('inf'),False,None,'1'], 'pose_temp':[float('-inf'),4,True]}.items():
 for value in values:rejects(lambda:S.normalize_request({**base,field:value}))
for bad in [None,[],{'recipe':'bad','clip_id':'clip-01'},{**base,'cmd':'echo hello'},{**base,'transcript':'hello'},{**base,'clip_id':'../../tmp'}]:rejects(lambda:S.normalize_request(bad))
for text in ['{"recipe":"aligned-media","recipe":"compact-all","clip_id":"clip-01"}', '{"recipe":"aligned-media","clip_id":"clip-01","audio_temp":NaN}']:
 rejects(lambda:S.load_request(text))
ok('invalid type, range, path/command injection, duplicate key and nonfinite JSON rejection')
for seconds in range(4,61):
 n=S.media_frames(seconds);assert n%2==0 and 2*n-3>=25*seconds and 2*(n-2)-3<25*seconds
ok('all57 durations produce minimal sufficient even codec frame counts')
@contextlib.contextmanager
def gpu_free():yield True
@contextlib.contextmanager
def gpu_busy():yield False
with tempfile.TemporaryDirectory() as td:
 q=Path(td);(q/'jobs').mkdir()
 def new(payload=base,status='queued'):
  p=q/'jobs'/uuid.uuid4().hex;p.mkdir();W.write(p/'request.json',payload);W.write(p/'state.json',{'status':status,'id':p.name});return p
 a=new();b=new();os.utime(a/'request.json',ns=(1,1));os.utime(b/'request.json',ns=(2,2))
 staged=q/'jobs'/('.new-'+uuid.uuid4().hex);staged.mkdir()
 assert W.jobs(q)==[a,b]
 def fake_run(job,req,clip,assets):W.state(job,'complete','complete',video='video.mp4',metadata='metadata.json')
 with patch.object(W.socket,'gethostname',return_value='landsend-0'),patch.object(W,'run_job',side_effect=AssertionError('GPU work')):
  rejects(lambda:W.serve(q,once=True))
 with patch.object(W.socket,'gethostname',return_value='landsend-1'),patch.object(W,'source_clip',return_value=({},{})),patch.object(W,'comparison_ready',return_value=False),patch.object(W,'gpu_pair',side_effect=AssertionError('GPU query before completion')):
  W.serve(q,once=True);assert W.read(a/'state.json')['stage']=='waiting-for-comparison'
 with patch.object(W.socket,'gethostname',return_value='landsend-1'),patch.object(W,'source_clip',return_value=({},{})),patch.object(W,'comparison_ready',return_value=True),patch.object(W,'gpu_pair',gpu_busy):
  W.serve(q,once=True);assert W.read(a/'state.json')['stage']=='waiting-for-gpus'
 with patch.object(W.socket,'gethostname',return_value='landsend-1'),patch.object(W,'source_clip',return_value=({},{})),patch.object(W,'comparison_ready',return_value=True),patch.object(W,'gpu_pair',gpu_free),patch.object(W,'run_job',side_effect=fake_run):
  W.serve(q,once=True);assert W.read(a/'state.json')['status']=='complete' and W.read(b/'state.json')['status']=='queued'
 c=new(status='running');W.recover_interrupted(q);assert W.read(c/'state.json')['status']=='failed'
 W.state(b,'running','generating');W.state(b,'failed','failed',error='test');assert 'video' not in W.read(b/'state.json')
 d=new({**base,'path':'/tmp/elsewhere'})
 with patch.object(W.socket,'gethostname',return_value='landsend-1'),patch.object(W,'run_job',side_effect=AssertionError('invalid request executed')):
  W.serve(q,once=True);assert W.read(d/'state.json')['status']=='failed'
 e=new();(q/'STOP').touch()
 with patch.object(W.socket,'gethostname',return_value='landsend-1'),patch.object(W,'run_job',side_effect=AssertionError('STOP ignored')):W.serve(q,once=True)
 assert W.read(e/'state.json')['status']=='queued'
 ok('atomic queue FIFO, role/comparison/occupancy gates, crash recovery, invalid request failure and STOP')
 with patch.object(W,'SOURCE',q):
  teams=[]
  for recipe in S.RECIPES:
   width=1 if recipe=='xabi-original' else 2
   for i in range(0,8,width):teams.append({'recipe':recipe,'clips':list(S.CLIP_IDS[i:i+width]),'status':'complete'})
  W.write(q/'CONTROLLED_COMPARISON_COMPLETE.json',{'status':'complete','teams':teams});assert W.comparison_ready()
  teams[-1]['clips']=['clip-01'];W.write(q/'CONTROLLED_COMPARISON_COMPLETE.json',{'status':'complete','teams':teams});assert not W.comparison_ready()
 ok('completion gate rejects duplicated/missing comparison outputs')
 cmd=W.generation_command(e,S.normalize_request({**base,'duration_seconds':4,'audio_temp':0,'pose_temp':3,'audio_topk':2048,'pose_topk':10}),{'clip_index':123,'start_frame':0,'voice_clip':'/fixed.npy'})
 assert '--devs' in cmd and cmd[cmd.index('--devs')+1]=='0,1' and '--audio-temp' in cmd
 batch=W.read(e/'batch.json');assert len(batch)==1 and batch[0]['frames']==52 and batch[0]['voice_clip']=='/fixed.npy'
 ok('parameterized genuine B1 command and explicit voice/frame/seed plumbing')
 assert W.environment()['CUDA_VISIBLE_DEVICES']=='6,7' and W.environment(video=True)['CUDA_VISIBLE_DEVICES']=='6'
 ok('child GPU environments fixed to node1 roles6,7 only')
print(json.dumps({'status':'complete','checks':checks,'gpu_work_started':False},indent=2))
