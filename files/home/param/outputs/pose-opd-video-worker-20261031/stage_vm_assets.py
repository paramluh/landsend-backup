from pathlib import Path
import concurrent.futures,hashlib,json,os,shlex,subprocess,time
out=Path('/home/param/outputs/pose-opd-video-worker-20261031');host='param@89.169.125.72';ssh=['ssh','-A','-i','/home/param/.ssh/id_ed25519','-o','BatchMode=yes'];target_root=out
files=[Path('/home/param/models/trained-weights/full-gt-nine-ref-endpoint-unguided-20261020/step-002911/model_ema.safetensors'),Path('/home/param/models/electron-vae/encoder-v2-pose.safetensors'),Path('/home/param/models/wan22-vae/Wan2.2_VAE.pth')]
for root in ['/home/param/models/pose-tokenizer-32cb-200k-noshift','/home/param/models/mimi']:
 files += [p for p in Path(root).iterdir() if p.is_file()]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(8*1024**2):h.update(b)
 return h.hexdigest()
manifest=[]
for p in files:manifest.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)})
(out/'ASSET-MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
script='from pathlib import Path\nfor s in '+repr([str(p.parent) for p in files]+[str(target_root)])+':Path(s).mkdir(parents=True,exist_ok=True)'
subprocess.run(ssh+[host,shlex.join(['python3','-c',script])],check=True)
def copy(p):
 log=out/('rsync-'+p.name+'.log')
 with log.open('wb') as f:subprocess.run(['rsync','-aL','--partial','-e',shlex.join(ssh),str(p),host+':'+str(p)],stdout=f,stderr=subprocess.STDOUT,check=True)
 return str(p)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
 for p in pool.map(copy,files):print('copied '+p,flush=True)
subprocess.run(['rsync','-a','-e',shlex.join(ssh),str(out/'ASSET-MANIFEST.json'),host+':'+str(out/'ASSET-MANIFEST.json')],check=True)
verify=r'''
from pathlib import Path
import concurrent.futures,hashlib,json
manifest=json.loads(Path('/home/param/outputs/pose-opd-video-worker-20261031/ASSET-MANIFEST.json').read_text())
def check(d):
 p=Path(d['path']);assert p.stat().st_size==d['bytes'];h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(8*1024**2):h.update(b)
 assert h.hexdigest()==d['sha256'],p
 return {'path':str(p),'sha256':h.hexdigest(),'bytes':p.stat().st_size}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:result=list(pool.map(check,manifest))
print(json.dumps({'verified':result,'gpu_started':False}))
'''
r=subprocess.run(ssh+[host,shlex.join(['python3','-c',verify])],check=True,text=True,capture_output=True);(out/'ASSETS-VERIFIED.json').write_text(r.stdout);print('ALL ASSETS VERIFIED',flush=True)
