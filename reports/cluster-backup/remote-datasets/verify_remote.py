"""Hash remote datasets against pinned original-HF and node-zero archive baselines."""
import concurrent.futures,hashlib,json,os,socket,stat,time
from pathlib import Path
BASE=Path('/home/param/outputs/landsend-backup-20261031/remote-datasets');BASE.mkdir(parents=True,exist_ok=True)
ROOT=Path('/home/param/datasets');HOST=socket.gethostname().split('.')[0]
catalog=json.loads((BASE/'node0-catalog.json').read_text());baseline={x['relative']:x for x in catalog['entries']}
by_digest={}
for x in catalog['entries']:by_digest.setdefault((x['bytes'],x['algorithm'],x['digest']),x)
files=[];links=[];skipped=[]
for b,ds,fs in os.walk(ROOT,followlinks=False):
 kept=[]
 for d in ds:
  p=Path(b,d)
  if p.is_symlink():links.append({'path':str(p),'target':os.readlink(p)})
  elif d in {'.cache','.git','.venv','__pycache__'}:skipped.append({'path':str(p),'reason':'regenerable_metadata_or_dependency_cache'})
  else:kept.append(d)
 ds[:]=kept
 for f in fs:
  p=Path(b,f)
  try:s=p.lstat()
  except OSError as exc:skipped.append({'path':str(p),'error':str(exc)});continue
  if stat.S_ISLNK(s.st_mode):links.append({'path':str(p),'target':os.readlink(p)})
  elif stat.S_ISREG(s.st_mode):files.append(str(p))
  else:skipped.append({'path':str(p),'reason':'not_regular_file'})

def digest_file(path,algorithm):
 if algorithm=='sha256':
  with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
 h=hashlib.sha1();h.update(b'blob '+str(path.stat().st_size).encode()+b'\0')
 with path.open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()

def one(name):
 p=Path(name);relative=str(p.relative_to(ROOT));expected=baseline.get(relative)
 for attempt in range(3):
  try:
   before=p.stat();algorithm=expected['algorithm'] if expected and expected['bytes']==before.st_size else 'sha256'
   digest=digest_file(p,algorithm);after=p.stat()
   if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):continue
   match=by_digest.get((before.st_size,algorithm,digest))
   if expected and expected['bytes']==before.st_size and expected['algorithm']==algorithm and expected['digest']==digest:match=expected
   record={'path':str(p),'relative':relative,'archive_path':str(p).lstrip('/'),'bytes':before.st_size,'mtime_ns':before.st_mtime_ns,'mode':stat.S_IMODE(before.st_mode),'inode':[before.st_dev,before.st_ino],'algorithm':algorithm,'digest':digest}
   if match:return {**record,'state':'matched','baseline_relative':match['relative'],'storage':match['storage']}
   if algorithm!='sha256':record.update(algorithm='sha256',digest=digest_file(p,'sha256'))
   return {**record,'state':'upload_required','reason':'content_differs' if expected else 'remote_only_path'}
  except Exception as exc:return {'path':str(p),'relative':relative,'state':'error','error':type(exc).__name__+': '+str(exc)}
 return {'path':str(p),'relative':relative,'state':'error','error':'File kept changing during hashing'}
start=time.time();matched=[];missing=[];errors=[];processed_bytes=0
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 for n,r in enumerate(pool.map(one,files),1):
  processed_bytes+=r.get('bytes',0)
  if r['state']=='matched':matched.append(r)
  elif r['state']=='upload_required':missing.append(r)
  else:errors.append(r)
  if n%500==0:
   progress={'host':HOST,'processed':n,'total':len(files),'matched':len(matched),'upload_required':len(missing),'errors':len(errors),'GiB':processed_bytes/2**30,'seconds':time.time()-start}
   (BASE/'progress.json').write_text(json.dumps(progress));print(json.dumps(progress),flush=True)
report={'format':'landsend-remote-dataset-verification-v1','host':HOST,'completed_unix':time.time(),'catalog_sha256':hashlib.sha256((BASE/'node0-catalog.json').read_bytes()).hexdigest(),'files':len(files),'bytes':processed_bytes,'matched':matched,'upload_required':missing,'errors':errors,'symlinks':links,'excluded':skipped,'seconds':time.time()-start}
(BASE/'verification.json').write_text(json.dumps(report,separators=(',',':'))+'\n')
(BASE/'upload-required.json').write_text(json.dumps(missing,indent=2)+'\n')
summary={'host':HOST,'state':'verified' if not errors else 'errors','files':len(files),'GiB':processed_bytes/2**30,'matched':len(matched),'original_hf_matches':sum(r['storage']['kind']=='original_hf' for r in matched),'node0_archive_matches':sum(r['storage']['kind']=='node0_dataset_archive' for r in matched),'upload_required':len(missing),'upload_GiB':sum(r['bytes'] for r in missing)/2**30,'errors':errors,'symlinks':len(links),'seconds':time.time()-start}
(BASE/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
