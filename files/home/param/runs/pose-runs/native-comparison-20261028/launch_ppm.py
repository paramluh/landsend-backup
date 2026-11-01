from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json,os,shlex,subprocess,time
R=Path('/home/param/pose-runs/native-comparison-20261028');REPO=Path('/home/param/pose-repo');SSH=['ssh','-i','/home/param/.ssh/id_ed25519_cluster','-o','BatchMode=yes'];clips=json.loads((R/'selection.json').read_text())['clips']
(R/'teams').mkdir(exist_ok=True);(R/'logs').mkdir(exist_ok=True)
jobs=[]
for node,recipe,offset in [('landsend-1','compact-all',0),('landsend-2','compact-all',4),('landsend-3','xabi-original',0),('landsend-4','xabi-original',4)]:
 for i in range(4):
  c=clips[offset+i];d=R/'outputs'/c['gallery_id']/recipe;d.mkdir(parents=True,exist_ok=True)
  job=dict(root=str(R),run_root=str(R),clip=c,clip_id=c['id'],recipe=recipe,node=node,devs=[2*i,2*i+1],gpu=2*i,directory=str(d),out=str(d),ppm=str(d/'ppm'),assets=str(Path(c['input_assets'])/'video_assets.json'),selection=str(R/'selection.json'))
  jfile=R/'teams'/f'{c["id"]}-{recipe}.json';jfile.write_text(json.dumps(job,indent=2)+'\n');job['job_file']=str(jfile)
  batch=[dict(clip_index=c['clip_index'],start_frame=0,frames=752,voice_clip=c['voice_clip'],seed=20261028+int(c['gallery_id'])-1,out=str(d/'ppm'))]
  bfile=R/'teams'/f'{c["id"]}-{recipe}.batch.json';bfile.write_text(json.dumps(batch,indent=2)+'\n');job['batch_file']=str(bfile);jobs.append(job)
(R/'execution-plan.json').write_text(json.dumps(jobs,indent=2)+'\n')
def sync(node):
 subprocess.run(SSH+[node,'mkdir -p '+shlex.quote(str(R/'teams'))],check=True)
 subprocess.run(['rsync','-a','-e',shlex.join(SSH),str(REPO/'infer/engine/ppm/fast_ppm.py'),node+':'+str(REPO/'infer/engine/ppm/')+'/'],check=True)
 subprocess.run(['rsync','-a','-e',shlex.join(SSH),*[str(REPO/'infer/transcript_pose'/f) for f in ('infer.py','infer_batch.py','inputs.py')],node+':'+str(REPO/'infer/transcript_pose/')+'/'],check=True)
 subprocess.run(['rsync','-a','-e',shlex.join(SSH),str(R/'teams')+'/',node+':'+str(R/'teams')+'/'],check=True)
with ThreadPoolExecutor(max_workers=4) as p:list(p.map(sync,[f'landsend-{i}' for i in range(1,5)]))
def run(j):
 name=j['clip_id']+'-'+j['recipe'];node=j['node'];pair=','.join(map(str,j['devs']))
 args=[str(REPO/'.venv/bin/python'),'-u',str(REPO/'infer/transcript_pose/infer_batch.py'),'--recipe',j['recipe'],'--text-mode','native','--batch-json',j['batch_file'],'--devs','0,1','--text-temp','1','--text-topk','0','--audio-temp','1','--pose-temp','1','--audio-topk','0','--pose-topk','0']
 command='cd '+shlex.quote(str(REPO))+' && source finetune/transcript_pose/environment/activate.sh && export CUDA_VISIBLE_DEVICES='+pair+' MIMI_DIR=/home/param/models/mimi FA3_DIR=/home/param/pose-repo/.cache/kernels/vllm-flash-attn3 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 HF_HUB_OFFLINE=1 && exec '+shlex.join(args)
 locks=Path('/home/param/.cache/heldout-comparison-gpu-locks')
 remote=['flock',str(locks/f'gpu-{j["devs"][0]}.lock'),'flock',str(locks/f'gpu-{j["devs"][1]}.lock'),'bash','-lc',command]
 with (R/'logs'/f'{name}.ppm.log').open('w') as log:
  proc=subprocess.Popen(SSH+[node,shlex.join(remote)],stdout=log,stderr=subprocess.STDOUT)
  print(json.dumps(dict(event='started',time=time.time(),job=name,node=node,devs=j['devs'],ssh_pid=proc.pid)),flush=True)
  rc=proc.wait()
 receipt=dict(status='exited' if rc==0 else 'failed',exit_code=rc,node=node,devs=j['devs'],time=time.time(),job=name)
 (R/'teams'/f'{name}.ppm-exit.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(receipt),flush=True)
 return rc
with ThreadPoolExecutor(max_workers=16) as p:codes=list(p.map(run,jobs))
raise SystemExit(any(codes))
