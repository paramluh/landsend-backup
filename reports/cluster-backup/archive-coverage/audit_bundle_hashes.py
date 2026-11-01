"""Identify exact Git bundles in selected files and source containers."""
import hashlib,json,socket,tarfile,zipfile
from pathlib import Path
BASE=Path('/home/param/outputs/landsend-backup-20261031');B=BASE/'orphans';results=[];seen=set()
def digest(f):return hashlib.file_digest(f,'sha256').hexdigest()
for name in ['selection.json','exclusions.json']:
 for r in json.loads((B/name).read_text()):
  p=Path(r['path'])
  if p in seen or not p.is_file():continue
  if p.suffix=='.bundle':
   seen.add(p)
   with p.open('rb') as f:results.append({'source':str(p),'member':None,'sha256':digest(f),'bytes':p.stat().st_size})
  elif str(p).endswith(('.tar','.tar.gz','.tgz','.zip','.tar.bz2','.tar.xz')) and 'datasets' not in p.parts:
   seen.add(p)
   if zipfile.is_zipfile(p):
    with zipfile.ZipFile(p) as archive:
     for m in archive.infolist():
      if m.filename.endswith('.bundle'):
       with archive.open(m) as f:results.append({'source':str(p),'member':m.filename,'sha256':digest(f),'bytes':m.file_size})
   else:
    with tarfile.open(p,'r:*') as archive:
     for m in archive:
      if m.isfile() and m.name.endswith('.bundle'):
       with archive.extractfile(m) as f:results.append({'source':str(p),'member':m.name,'sha256':digest(f),'bytes':m.size})
output={'host':socket.gethostname(),'bundles':results,'unique_hashes':sorted({x['sha256'] for x in results})}
(BASE/'archive-coverage/bundle-hash-audit.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output))
