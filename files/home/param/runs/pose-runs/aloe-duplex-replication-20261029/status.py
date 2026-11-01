import concurrent.futures,json,pathlib,subprocess
root=pathlib.Path(__file__).resolve().parent
probe='''import json,pathlib,socket
r=pathlib.Path('/home/param/pose-runs/aloe-duplex-replication-20261029');h=socket.gethostname().split('.')[0]
p=r/('status-'+h+'.json')
print(p.read_text() if p.exists() else json.dumps({'host':h,'error':'status missing','log':(r/'replicate.log').read_text()[-1500:]}))
'''
def get(h):
 p=subprocess.run(['ssh','-i','/home/param/.ssh/id_ed25519_cluster','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10',h,'python3 -'],input=probe,text=True,capture_output=True,timeout=30)
 return json.loads(p.stdout) if p.returncode==0 else {'host':h,'error':p.stderr}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:out=list(pool.map(get,['landsend-5','landsend-12']))
(root/'status-latest.json').write_text(json.dumps(out,indent=2)+'\n')
for o in out:print(json.dumps(o))
