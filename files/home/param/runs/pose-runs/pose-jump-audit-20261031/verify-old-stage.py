from pathlib import Path
import concurrent.futures,hashlib,json,socket,time
root=Path('/home/param/models/imported-weights/ppm-pose6-mono17850-final')
s=json.loads((root/'SOURCE.json').read_text());assert s['revision']=='71c9b718e1d927e0b26d89a9b225459fc3270dc9'
c=json.loads((root/'export-357-dcp/train_cfg.json').read_text());assert c['step']==357 and c['run']=='ppm-pose6-mono17850-j31240' and 'pose_band_mask' not in c['cfg']
def check(f):
 p=root/f['path'];assert p.stat().st_size==f['size'],p
 h=hashlib.sha256() if f['sha256'] else hashlib.sha1()
 if not f['sha256']:h.update(f"blob {p.stat().st_size}\0".encode())
 with p.open('rb') as fp:
  for b in iter(lambda:fp.read(8<<20),b''):h.update(b)
 assert h.hexdigest()==(f['sha256'] or f['blob_id']),p
 return f['path']
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:verified=list(ex.map(check,s['files']))
out={'host':socket.gethostname(),'repo':s['repo'],'revision':s['revision'],'files_verified':len(verified),'bytes_verified':sum(f['size'] for f in s['files']),'time':time.time(),'training_run':c['run'],'step':357,'band_mask_present':False,'method':'Every file checked against HF LFS SHA256 or Git blob SHA1'}
(root/('DEPLOYMENT_VERIFIED.'+socket.gethostname()+'.json')).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
