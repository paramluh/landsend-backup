"""Read-only provenance/credential inventory for opaque compressed archives."""
import hashlib,json,socket,sys,tarfile,zipfile
from pathlib import Path,PurePosixPath
BASE=Path('/home/param/outputs/landsend-backup-20261031')
sys.path.insert(0,str(BASE/'orphans'))
import orphan_backup as B
rows=[]
for source in ['selection.json','exclusions.json']:
 for r in json.loads((B.BASE/source).read_text()):
  p=Path(r['path'])
  if not str(p).endswith(('.tar.gz','.tgz','.zip','.tar.bz2','.tar.xz')) or not p.is_file():continue
  item={'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest(),'selection_source':source,'members':0,'expanded_bytes':0,'flagged_members':[],'extensions':{}}
  def inspect(name,data):
   item['members']+=1;item['expanded_bytes']+=len(data)
   suffix=Path(name).suffix.lower();item['extensions'][suffix]=item['extensions'].get(suffix,0)+1
   _,count=B.sanitize(data)
   secretname=Path(name).name.lower() in B.SECRET_NAMES or Path(name).name.lower().startswith(('.env','id_ed25519','id_rsa')) or Path(name).suffix.lower() in {'.pem','.key','.p12','.pfx'}
   unsafe=PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts
   if count or secretname or unsafe:item['flagged_members'].append({'name':name,'credential_pattern_matches':count,'credential_filename':secretname,'unsafe_path':unsafe})
  try:
   if zipfile.is_zipfile(p):
    with zipfile.ZipFile(p) as z:
     for info in z.infolist():
      if not info.is_dir():inspect(info.filename,z.read(info))
   else:
    with tarfile.open(p,'r:*') as t:
     for info in t:
      if info.isfile():
       with t.extractfile(info) as f:inspect(info.name,f.read())
   item['state']='needs_review' if item['flagged_members'] else 'inspected_no_credentials'
  except Exception as e:item['state']='error';item['error']=f'{type(e).__name__}: {e}'
  rows.append(item)
out=BASE/'archive-coverage';out.mkdir(exist_ok=True)
(out/'opaque-archives-audit.json').write_text(json.dumps({'host':socket.gethostname(),'archives':rows},indent=2)+'\n')
print(json.dumps({'host':socket.gethostname(),'archives':len(rows),'flags':[{'path':r['path'],'state':r['state'],'flagged_members':r['flagged_members']} for r in rows if r['state']!='inspected_no_credentials'],'sha256s':{r['path']:r['sha256'] for r in rows}}))
