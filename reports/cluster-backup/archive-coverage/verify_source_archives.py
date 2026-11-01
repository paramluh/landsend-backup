"""Verify all-node preserved source-tar sets against private HF file hashes."""
import concurrent.futures,importlib.util,json,time
from pathlib import Path
BASE=Path('/home/param/outputs/landsend-backup-20261031')
spec=importlib.util.spec_from_file_location('refresh',BASE/'final_artifact_refresh.py');M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
CODE='''import hashlib,json,os,socket
from pathlib import Path
b=Path('/home/param/outputs/landsend-backup-20261031/orphans');group=b/'archives/source-archives';running=[]
for p in Path('/proc').iterdir():
 try:
  if p.name.isdigit() and p.stat().st_uid==os.getuid() and any(Path(a).name=='backup_source_archives.py' for a in p.joinpath('cmdline').read_bytes().decode().split('\\0')):running.append(int(p.name))
 except OSError:pass
complete=group/'COMPLETE.json'
if running or not complete.exists():print(json.dumps({'state':'waiting','running':running,'complete':complete.exists()}))
else:
 c=json.loads(complete.read_text());audit=json.loads((b/'source-archive-audit.json').read_text());files=[]
 for p in [complete,group/'aliases.json',*group.glob('*.files.json'),b/'source-archive-audit.json']:
  data=p.read_bytes();remote='metadata/'+p.name if p.name=='source-archive-audit.json' else p.name if p.name=='COMPLETE.json' else 'manifests/'+p.name
  files.append({'path':socket.gethostname()+'/source-archives/'+remote,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'git_blob_sha1':hashlib.sha1(b'blob '+str(len(data)).encode()+b'\\0'+data).hexdigest(),'kind':'metadata'})
 for a in c['archives']:
  assert a['state']=='verified' and not a.get('errors')
  files.append({'path':a['hf_path'],'bytes':a['bytes'],'sha256':a['sha256'],'kind':'archive'})
 assert c['state']=='complete' and not c.get('errors')
 print(json.dumps({'state':'ready','repo':c['hf_repo'],'group':'source-archives','files':files,'canonical':{'missing':[],'files':0},'archives':len(c['archives']),'archive_bytes':sum(a['bytes'] for a in c['archives']),'source_audit':audit}))
'''
def main():
 out=BASE/'archive-coverage';deadline=time.time()+1800;pending=set(M.HOSTS);results={}
 while pending and time.time()<deadline:
  for h in list(pending):
   try:
    evidence=M.remote(h,CODE)
    if evidence['state']=='waiting':continue
    result=M.verify_remote(evidence);result.update(host=h,source_archives=len(evidence['source_audit']),sanitized_archives=sum(x['repacked'] for x in evidence['source_audit']),audit=evidence['source_audit'])
    M.atomic(out/(h+'-source-archives-verified.json'),result);results[h]=result;pending.remove(h)
   except Exception as e:results[h]={'state':'retrying','error':f'{type(e).__name__}: {e}'}
  M.atomic(out/'SOURCE_ARCHIVES_VERIFICATION.json',{'state':'verified' if not pending else 'waiting','pending_hosts':sorted(pending),'hosts':results,'time':time.time()})
  if pending:time.sleep(20)
 if pending:raise RuntimeError('Source archives unfinished: '+','.join(sorted(pending)))
 print(json.dumps({'state':'verified','hosts':len(results),'source_archives':sum(x['source_archives'] for x in results.values())}))
if __name__=='__main__':main()
