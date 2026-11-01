from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
import hashlib, json, time, concurrent.futures
repo='InternalCan/ppm-pose6-mono17850-final'; revision='71c9b718e1d927e0b26d89a9b225459fc3270dc9'
root=Path('/home/param/models/imported-weights/ppm-pose6-mono17850-final');root.mkdir(parents=True,exist_ok=True)
info=HfApi().model_info(repo,revision=revision,files_metadata=True)
files=[{'path':f.rfilename,'size':f.size,'sha256':f.lfs.sha256 if f.lfs else None,'blob_id':f.blob_id} for f in info.siblings]
(root/'SOURCE.json').write_text(json.dumps({'repo':repo,'revision':revision,'files':files},indent=2)+'\n')
print(json.dumps({'phase':'download','revision':revision,'files':len(files),'bytes':sum(f['size'] for f in files)}),flush=True)
snapshot_download(repo_id=repo,revision=revision,local_dir=root,max_workers=8)
def check(f):
 p=root/f['path'];h=hashlib.sha256() if f['sha256'] else hashlib.sha1()
 if not f['sha256']:h.update(f"blob {p.stat().st_size}\0".encode())
 with p.open('rb') as fp:
  for b in iter(lambda:fp.read(8<<20),b''):h.update(b)
 expected=f['sha256'] or f['blob_id']
 assert p.stat().st_size==f['size'] and h.hexdigest()==expected,(f['path'],h.hexdigest(),expected)
 return f['path']
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:verified=list(ex.map(check,files))
out={'repo':repo,'revision':revision,'files_total':len(files),'files_verified':len(verified),'bytes_verified':sum(f['size'] for f in files),'verified_at':time.time(),'verification':'HF LFS SHA256 or Git blob SHA1 for every file'}
(root/'PRODUCTION_VERIFIED.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
