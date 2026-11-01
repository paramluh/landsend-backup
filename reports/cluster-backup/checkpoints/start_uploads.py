import concurrent.futures,json,os,pathlib,shlex,subprocess,time
root=pathlib.Path('/home/param/outputs/landsend-backup-20261031/checkpoints')
hosts=['landsend-0','landsend-1','landsend-2','landsend-3','landsend-4','landsend-5','landsend-12']
base=['ssh','-i','/home/param/.ssh/id_ed25519_cluster','-o','BatchMode=yes','-o','ConnectTimeout=8']
script=root/'upload_node.py'
def one(host):
 plan=root/(host+'-plan.json');log=root/(host+'-upload.log');cmd=['/home/param/pose-repo/.venv/bin/python','-u',str(script),'--plan',str(plan)]
 if host=='landsend-0':
  with log.open('ab',buffering=0) as out:p=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','HF_HUB_DISABLE_PROGRESS_BARS':'1','OMP_NUM_THREADS':'2'})
  pid=p.pid
 else:
  subprocess.run(base+[host,'mkdir -p '+shlex.quote(str(root))],check=True)
  for source in [script,plan]:
   with source.open('rb') as inp:subprocess.run(base+[host,'cat > '+shlex.quote(str(source))],stdin=inp,check=True)
  runner="import os,subprocess,json; f=open("+repr(str(log))+",'ab',buffering=0); p=subprocess.Popen("+repr(cmd)+",stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','HF_HUB_DISABLE_PROGRESS_BARS':'1','OMP_NUM_THREADS':'2'});print(p.pid)"
  p=subprocess.run(base+[host,'python3 -c '+shlex.quote(runner)],text=True,capture_output=True,check=True);pid=int(p.stdout.strip())
 return {'host':host,'pid':pid,'started':time.time(),'log':str(log)}
with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool:
 results=list(pool.map(one,hosts))
(root/'upload-processes.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
