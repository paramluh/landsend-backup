import contextlib,fcntl,json,os,sys,tempfile,uuid
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,'/home/param/pose-repo/infer/transcript_pose')
import request_worker as W
from request_schema import normalize_request
checks=[]
def reject(fn):
 try:fn()
 except ValueError:return
 raise AssertionError('Expected rejection')
for a in range(8):
 for b in range(8):
  if a==b:reject(lambda:W.parse_gpu_pair(f'{a},{b}'))
  else:
   assert W.parse_gpu_pair(f'{a},{b}')==(a,b)
   assert W.environment(gpus=(a,b))['CUDA_VISIBLE_DEVICES']==f'{a},{b}'
   assert W.environment(video=True,gpus=(a,b))['CUDA_VISIBLE_DEVICES']==str(a)
for invalid in ['0','0,1,2','-1,2','8,2','a,b','1;echo,2']:reject(lambda:W.parse_gpu_pair(invalid))
for node in ['landsend-0','landsend-12','landsend-5']:
 with patch.object(W.socket,'gethostname',return_value=node):reject(lambda:W.execution_guard(node,(0,1)))
with patch.object(W.socket,'gethostname',return_value='landsend-2'):
 reject(lambda:W.execution_guard('landsend-1',(0,1)))
 with patch.object(W.os,'geteuid',return_value=0):reject(lambda:W.execution_guard('landsend-2',(0,1)))
assert W.process_start_ticks(os.getpid())>0
checks.append('All56 ordered GPU pairs, role/UID guards, environment mapping and PID-start identity')
with tempfile.TemporaryDirectory() as temp:
 root=Path(temp);locks=root/'locks';jobs=root/'jobs';jobs.mkdir()
 def probe(gpu):
  with (locks/f'gpu-{gpu}.lock').open('a') as f:
   try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);return True
   except BlockingIOError:return False
 def smi(cmd,**kw):return '' if '--query-compute-apps=pid' in cmd else '0'
 with patch.object(W,'GPU_LOCKS',locks),patch.object(W.subprocess,'check_output',side_effect=smi):
  with W.gpu_pair((1,3)) as lease:
   assert lease and not probe(1) and not probe(3)
   lease.release(3);assert probe(3) and not probe(1)
  assert probe(1) and probe(3)
 checks.append('Real flock exclusion, immediate secondary release, primary retained until context exits')
 def new(request=None):
  job=jobs/uuid.uuid4().hex;job.mkdir()
  W.write(job/'request.json',request or {'recipe':'compact-all','clip_id':'clip-01','text_mode':'native','duration_seconds':4})
  W.write(job/'state.json',{'status':'running','stage':'assigned','id':job.name})
  W.write(job/'assignment.json',{'claim_token':uuid.uuid4().hex,'node':'landsend-2','gpus':[1,3]})
  return job
 def fake_run(job,req,clip,assets,**kwargs):
  assert kwargs['node']=='landsend-2' and kwargs['gpus']==(1,3) and req['text_mode']=='native'
  assert W.read(job/'executor.json')['active_gpus']==[1,3]
  kwargs['release_secondary']()
  assert W.read(job/'executor.json')['active_gpus']==[1] and probe(3) and not probe(1)
  kwargs['release_primary']()
  assert W.read(job/'executor.json')['active_gpus']==[] and probe(1) and probe(3)
  (job/'video.mp4').write_bytes(b'cpu-lifecycle-fixture')
  W.write(job/'metadata.json',{'sha256':W.sha(job/'video.mp4')})
  W.state(job,'complete','complete',video='video.mp4',metadata='metadata.json')
 job=new()
 with patch.object(W.socket,'gethostname',return_value='landsend-2'),patch.object(W,'GPU_LOCKS',locks),patch.object(W.subprocess,'check_output',side_effect=smi),patch.object(W,'source_clip',return_value=({},{})),patch.object(W,'comparison_ready',return_value=True),patch.object(W,'run_job',side_effect=fake_run):
  assert W.execute_job(job,'landsend-2',(1,3))==0
  record=W.read(job/'executor.json');assert record['status']=='complete' and record['active_gpus']==[] and record['exit_code']==0
  assert record['claim_token']==W.read(job/'assignment.json')['claim_token']
  assert W.execute_job(job,'landsend-2',(1,3))==0
  busy=new()
  with (locks/'gpu-1.lock').open('a') as held:
   fcntl.flock(held,fcntl.LOCK_EX)
   assert W.execute_job(busy,'landsend-2',(1,3))==75
  assert W.read(busy/'state.json')['stage']=='waiting-for-gpus'
  assert W.read(busy/'executor.json')['status']=='busy' and not (busy/'ppm').exists()
  dup=new();W.write(dup/'executor.json',{'status':'running','sentinel':'must-not-overwrite'})
  with (dup/'.executor.lock').open('a') as held:
   fcntl.flock(held,fcntl.LOCK_EX)
   assert W.execute_job(dup,'landsend-2',(1,3))==75
  assert W.read(dup/'executor.json')['sentinel']=='must-not-overwrite'
  bad=new({'recipe':'aligned-media','clip_id':'clip-01','text_mode':'native'})
  assert W.execute_job(bad,'landsend-2',(1,3))==1
  assert W.read(bad/'executor.json')['status']=='failed'
  failed=new()
  with patch.object(W,'run_job',side_effect=RuntimeError('simulated subprocess failure')):
   assert W.execute_job(failed,'landsend-2',(1,3))==1
  assert W.read(failed/'state.json')['status']=='failed' and probe(1) and probe(3)
 checks.append('Assignment/claim handoff, durable lifecycle, idempotent complete, busy75, duplicate exclusion and failure cleanup')
result={'status':'complete','gpu_work_started':False,'checks':checks}
W.write(Path('/home/param/pose-runs/request-worker-audit-20261028/executor-cpu-audit.json'),result)
print(json.dumps(result,indent=2))
