import concurrent.futures,hashlib,json,time
from pathlib import Path
root=Path('/home/param/models/imported-weights')
here=Path(__file__).parent
for mf in here.glob('ppm-*.json'):
 d=json.loads(mf.read_text()); p=root/d['repo_id'].split('/')[-1]
 def check(f):
  q=p/f['path']; assert q.stat().st_size==f['size'],q
  h=hashlib.sha256() if f['sha256'] else hashlib.sha1()
  if not f['sha256']: h.update(b'blob '+str(f['size']).encode()+b'\0')
  with q.open('rb') as fi:
   for block in iter(lambda:fi.read(8*1024*1024),b''): h.update(block)
  assert h.hexdigest()==(f['sha256'] or f['git_blob_sha1']),q
  return f['size']
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: total=sum(pool.map(check,d['files']))
 result={'repo':d['repo_id'],'revision':d['revision'],'status':'passed','files':len(d['files']),'bytes':total,'time':time.time()}
 (here/(p.name+'.verified.json')).write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result),flush=True)
