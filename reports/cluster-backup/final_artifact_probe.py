"""Read-only per-node evidence for the final artifact refresh controller."""
import argparse,hashlib,json,os,socket
from pathlib import Path
BASE=Path('/home/param/outputs/landsend-backup-20261031')
ORPHANS=BASE/'orphans'
RUN=Path('/home/param/runs/pose-runs/pose-basic-opd-b336-20261031')
def read(path):
 try:return json.loads(path.read_text())
 except (OSError,ValueError):return None
def running():
 rows=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   if p.stat().st_uid!=os.getuid():continue
   args=p.joinpath('cmdline').read_bytes().decode().split('\0')
   if any(Path(a).name in {'remote_orphan_worker.py','final_node_refresh.py','backup_source_archives.py'} for a in args) or (any(Path(a).name=='orphan_backup.py' for a in args) and 'pack' in args and 'datasets' not in args):rows.append({'pid':int(p.name),'args':args})
  except OSError:pass
 return rows
def initial():
 group=ORPHANS/'archives/artifacts';marker=read(group/'COMPLETE.json') or read(group/'PARTIAL.json')
 errors=[]
 if marker and marker.get('errors'):
  for p in group.glob('*.files.json'):
   for row in read(p) or []:
    if row.get('error'):
     ephemeral='/outputs/' in row['path'] and ('/snapshots/' in row['path'] or '/.sample-' in row['path'])
     errors.append({'path':row['path'],'error':row['error'],'classification':'rotating_dashboard_cache' if ephemeral else 'needs_review'})
 if marker and marker.get('errors',0)!=len(errors):errors.append({'path':str(group),'error':'Marker error count does not match manifest errors','classification':'needs_review'})
 return {'marker':marker and {k:v for k,v in marker.items() if k!='archives'},'errors':errors,'workers':running(),'groups':[p.name for p in (ORPHANS/'archives').glob('artifacts-final-*') if p.is_dir()],'source_groups':[p.name for p in (ORPHANS/'archives').glob('source-archives-final-*') if p.is_dir()],'job':read(BASE/'final-artifact-refresh'/f'{socket.gethostname()}.process.json'),'finished':read(BASE/'final-artifact-refresh'/f'{socket.gethostname()}.finished.json')}
def verify(group):
 folder=ORPHANS/'archives'/group;complete=read(folder/'COMPLETE.json');partial=read(folder/'PARTIAL.json')
 if complete is None:return {'state':'partial' if partial else 'missing','marker':partial,'workers':running()}
 assert complete.get('state')=='complete' and not complete.get('errors')
 files=[]
 for path in [folder/'COMPLETE.json',folder/'aliases.json',*folder.glob('*.files.json')]:
  data=path.read_bytes();remote=f'{socket.gethostname()}/{group}/'+(path.name if path.name=='COMPLETE.json' else 'manifests/'+path.name)
  files.append({'path':remote,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'git_blob_sha1':hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest(),'kind':'metadata'})
 for name in (['source-archive-audit.json'] if group.startswith('source-archives') else ['selection.json','symlinks.json','empty-directories.json','exclusions.json']):
  data=(ORPHANS/name).read_bytes();files.append({'path':f'{socket.gethostname()}/{group}/metadata/{name}','bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'git_blob_sha1':hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest(),'kind':'metadata'})
 for a in complete['archives']:
  assert a['state']=='verified' and not a.get('errors')
  receipt=read(folder/(a['archive']+'.uploaded.json'))
  assert receipt and receipt['sha256']==a['sha256'] and receipt['state']=='verified'
  files.append({'path':a['hf_path'],'bytes':a['bytes'],'sha256':a['sha256'],'kind':'archive'})
 critical=[RUN/n for n in ['config.json','plan.json','data-selection.json','RESULT.json','metrics.jsonl']]
 events=list(RUN.glob('events-rank*.jsonl'));assert len(events)>=8,'Missing per-rank canonical event logs'
 critical.extend(events);critical.extend((RUN/'control').glob('*.json'))
 if socket.gethostname()=='landsend-0':
  for p in critical[:5]:assert p.is_file(),f'Missing canonical run file: {p}'
 current={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in critical if p.is_file()}
 covered={};archived={}
 for manifest in (ORPHANS/'archives').glob('artifacts*/part-*.files.json'):
  receipt=read(manifest.with_name(manifest.name.removesuffix('.files.json')+'.uploaded.json'))
  if not receipt or receipt.get('state')!='verified':continue
  for row in read(manifest) or []:
   p=row['path']
   if row.get('error'):continue
   version=(row['bytes'],row['mtime_ns']);proof={'manifest':str(manifest),'archive':receipt['hf_path'],'sha256':receipt['sha256']}
   archived[p]=(version,proof)
   if p in current and version==current[p]:covered[p]=proof
 # Hardlink aliases share the target's stored bytes and metadata. Resolve chains.
 aliases=[item for path in (ORPHANS/'archives').glob('artifacts*/aliases.json') for item in read(path) or []]
 for _ in range(len(aliases)+1):
  changed=False
  for item in aliases:
   source='/'+item['target'].lstrip('/');target='/'+item['path'].lstrip('/')
   if source in archived and target not in archived:
    archived[target]=archived[source];changed=True
    version,proof=archived[target]
    if target in current and version==current[target]:covered[target]={**proof,'hardlink_target':source}
  if not changed:break
 return {'state':'ready','group':group,'repo':complete['hf_repo'],'files':files,'canonical':{'files':len(current),'covered':covered,'missing':[p for p in current if p not in covered]},'archives':len(complete['archives']),'archive_bytes':sum(a['bytes'] for a in complete['archives']),'marker':{k:v for k,v in complete.items() if k!='archives'}}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--group');a=p.parse_args();print(json.dumps(verify(a.group) if a.group else initial()))
