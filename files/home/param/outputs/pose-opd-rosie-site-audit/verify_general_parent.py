import concurrent.futures,hashlib,json,time
from pathlib import Path
root=Path('/home/param/models/imported-weights/ppm-aloe-zoom-cotrain')
out=Path('/home/param/outputs/pose-opd-rosie-site-audit/general-parent')
expected=json.loads((out/'expected_sha256.json').read_text())
def verify(entry):
 p=root/entry['path'];assert p.stat().st_size==entry['bytes'],str(p)
 sha=hashlib.sha256()
 with p.open('rb') as f:
  while chunk:=f.read(8*1024*1024): sha.update(chunk)
 actual=sha.hexdigest()
 if entry['sha256']: assert actual==entry['sha256'],(str(p),actual,entry['sha256'])
 return {**entry,'actual_sha256':actual}
started=time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool: result=list(pool.map(verify,expected))
receipt={'repo':'InternalCan/ppm-aloe-zoom-cotrain','revision':'8cdb31a0cd08d4b7ceeb7e1cc9b9ebbdebdc8ba4','checkpoint':'export-253','sha256_lfs_matches':sum(bool(x['sha256']) for x in result),'bytes':sum(x['bytes'] for x in result),'seconds':time.time()-started,'files':result}
(out/'sha256_verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
(root/'HF_SHA256_VERIFICATION.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='files'}))
