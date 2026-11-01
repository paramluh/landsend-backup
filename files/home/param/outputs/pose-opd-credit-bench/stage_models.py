import concurrent.futures,subprocess,pathlib,json,time
base=pathlib.Path('/home/param/models/imported-weights')
models=['ppm-aloe-poseduplex-raw-final/export-424','ppm-aloe-v2vox-raw-w14503/export-253','ppm-aloe-monoplex-pose-raw-bandfix-355/weights/export-355']
hosts=['landsend-'+str(i) for i in [2,3,4,5,12]]
ssh='ssh -i /home/param/.ssh/id_ed25519_cluster -o BatchMode=yes -o StrictHostKeyChecking=yes -o Compression=no'
def run(h):
 t=time.time(); results=[]
 for rel in models:
  src=base/rel
  p=subprocess.run(['rsync','-a','--whole-file','--stats','--info=progress0','-e',ssh,str(src)+'/',h+':'+str(src)+'/'],capture_output=True,text=True)
  results.append({'model':rel,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
  if p.returncode:break
 r={'host':h,'seconds':time.time()-t,'results':results}
 pathlib.Path('/home/param/outputs/pose-opd-credit-bench/stage-'+h+'.json').write_text(json.dumps(r,indent=2))
 print(json.dumps(r),flush=True)
 return r
# Parent dirs are created before transfer, no existing exports are deleted.
for h in hosts:
 subprocess.run(ssh.split()+[h,'mkdir -p '+' '.join(str(base/rel) for rel in models)],check=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:list(ex.map(run,hosts))
