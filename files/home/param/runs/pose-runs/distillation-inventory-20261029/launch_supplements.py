import concurrent.futures,json,pathlib,subprocess,socket,time
base=pathlib.Path('/home/param/pose-runs/distillation-inventory-20261029');hosts=[f'landsend-{i}' for i in (0,1,2,3,4,5,12)];local=socket.gethostname().split('.')[0]
opts=['-i','/home/param/.ssh/id_ed25519_cluster','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10']
def stage(host):
 log=base/f'supplemental-stage-{host}.log'
 if host!=local:
  subprocess.run(['ssh',*opts,host,'mkdir -p '+str(base)],check=True,capture_output=True)
  subprocess.run(['scp',*opts,str(base/'stage_supplements.py'),str(base/'supplemental-manifest.json'),host+':'+str(base)+'/'],check=True,capture_output=True)
 cmd=['/home/param/.hf-cli/venv/bin/python',str(base/'stage_supplements.py')]
 if host!=local:cmd=['ssh',*opts,host,*cmd]
 with log.open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 if host!=local and p.returncode==0:subprocess.run(['scp',*opts,host+':'+str(base/f'supplemental-status-{host}.json'),str(base)+'/'],check=True,capture_output=True)
 result={'host':host,'exit_code':p.returncode,'log':str(log)};print(json.dumps(result),flush=True);return result
with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool:results=list(pool.map(stage,hosts))
(base/'supplemental-coverage.json').write_text(json.dumps(results,indent=2)+'\n')
if any(r['exit_code'] for r in results):raise SystemExit(1)
