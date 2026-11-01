import json,hashlib,socket,os
from pathlib import Path
base=Path('/home/param/outputs/landsend-backup-20261031');selection=base/'orphans/selection.json';paths=[Path(r['path']) for r in json.loads(selection.read_text()) if r['path'].lower().endswith('.bundle')]
seen={};rows=[];errors=[]
for p in paths:
 try:
  s=p.stat();key=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
  if key not in seen:
   with p.open('rb') as f:magic=f.read(16);f.seek(0);digest=hashlib.file_digest(f,'sha256').hexdigest()
   seen[key]={'sha256':digest,'bytes':s.st_size,'git_bundle':magic in (b'# v2 git bundle\n',b'# v3 git bundle\n')}
  rows.append({'path':str(p),**seen[key]})
 except Exception as e:errors.append({'path':str(p),'error':str(e)})
report={'host':socket.gethostname().split('.')[0],'files':rows,'errors':errors};out=base/'git/bundle-file-audit.json';out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'host':report['host'],'files':len(rows),'unique_contents':len({r['sha256'] for r in rows}),'errors':errors}))
