import concurrent.futures,json,pathlib,subprocess,shlex
root=pathlib.Path(__file__).resolve().parent
ssh=['ssh','-i','/home/param/.ssh/id_ed25519_cluster','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10']
payload=json.dumps({'root':str(root),'files':{name:(root/name).read_text() for name in ['replicate.py','source-files.json']}})
remote='''import json,sys,pathlib,subprocess
p=json.load(sys.stdin);r=pathlib.Path(p['root']);r.mkdir(parents=True,exist_ok=True)
for name,body in p['files'].items():(r/name).write_text(body)
with (r/'replicate.log').open('a') as log:
 child=subprocess.Popen(['python3',str(r/'replicate.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'pid':child.pid,'root':str(r)}))
'''
def launch(host):
 p=subprocess.run(ssh+[host,'python3 -c '+shlex.quote(remote)],input=payload,text=True,capture_output=True,timeout=45)
 return {'host':host,'returncode':p.returncode,'output':p.stdout.strip(),'error':p.stderr.strip()}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:r=list(pool.map(launch,['landsend-5','landsend-12']))
(root/'launch.json').write_text(json.dumps(r,indent=2)+'\n')
for x in r:print(json.dumps(x))
assert all(x['returncode']==0 for x in r)
