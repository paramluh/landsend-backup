import concurrent.futures,json,pathlib,subprocess,socket
control=pathlib.Path(__file__).resolve().parent
hosts=['landsend-'+str(i) for i in (0,1,2,3,4,5,12)]
probe='''import json,pathlib,socket
p=pathlib.Path('/home/param/pose-runs/distillation-inventory-20261029');host=socket.gethostname().split('.')[0]
s=json.loads((p/('staging-'+host+'.json')).read_text());s['pid_live']=pathlib.Path('/proc/'+str(s['pid'])).exists();print(json.dumps(s))
'''
def check(host):
 cmd=['python3','-'] if host==socket.gethostname().split('.')[0] else ['ssh','-i','/home/param/.ssh/id_ed25519_cluster','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=8',host,'python3 -']
 r=subprocess.run(cmd,input=probe,text=True,capture_output=True,timeout=25)
 if r.returncode:return {'host':host,'error':r.stderr.strip()}
 return json.loads(r.stdout)
with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool:results=list(pool.map(check,hosts))
(control/'cluster-staging-latest.json').write_text(json.dumps(results,indent=2)+'\n')
for r in results:
 print(json.dumps({k:r.get(k) for k in ('host','state','pid_live','files_verified','files_total','bytes_verified','bytes_total','updated_utc','failures','error')}))
