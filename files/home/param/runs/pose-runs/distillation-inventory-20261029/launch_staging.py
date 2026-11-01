import concurrent.futures,json,pathlib,subprocess,socket
control=pathlib.Path(__file__).resolve().parent
hosts=['landsend-'+str(i) for i in (0,1,2,3,4,5,12)]
ssh=['ssh','-i','/home/param/.ssh/id_ed25519_cluster','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=8']
# Only the pinned manifest and downloader are copied. Each node uses its existing HF login.
payload={'directory':str(control),'files':{n:(control/n).read_text() for n in ['model-manifest.json','stage_models.py']}}
remote='''import json,sys,pathlib,subprocess
p=json.load(sys.stdin); root=pathlib.Path(p['directory']); root.mkdir(parents=True,exist_ok=True)
for name,data in p['files'].items():(root/name).write_text(data)
with (root/'staging.log').open('a') as log:
 child=subprocess.Popen(['/home/param/.hf-cli/venv/bin/python',str(root/'stage_models.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'pid':child.pid,'control':str(root)}))
'''
import shlex
def launch(host):
 if host==socket.gethostname().split('.')[0]:cmd=['python3','-c',remote]
 else:cmd=ssh+[host,'python3 -c '+shlex.quote(remote)]
 r=subprocess.run(cmd,input=json.dumps(payload),text=True,capture_output=True,timeout=30)
 return {'host':host,'returncode':r.returncode,'result':r.stdout.strip(),'error':r.stderr.strip()}
with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool:
 results=list(pool.map(launch,hosts))
for result in results:print(json.dumps(result),flush=True)
(control/'staging-launch.json').write_text(json.dumps(results,indent=2)+'\n')
assert all(r['returncode']==0 for r in results)
