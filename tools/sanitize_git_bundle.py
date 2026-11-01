"""Sanitize COPIES of Git bundles. Never modify source repositories or source bundles.

sanitize_bundle_file(path, cache_dir=None) -> (seekable binary stream, redaction_count).
Cache receipt *.json and *.commit-map.csv preserve source/destination identities.
Only the known exposed vendor service-account credential blob is redacted.
Nested Git-bundle blobs are recursively sanitized as well.
"""
import os,json,hashlib,subprocess,tempfile,shutil,fcntl,csv,io
from pathlib import Path
KNOWN_BLOB='90b6f61602f20a1300c37e9dd4a288c93e5b1ff9'
DEFAULT_CACHE=Path('/home/param/outputs/landsend-backup-20261031/git/sanitized-bundles')
MAGICS=(b'# v2 git bundle\n',b'# v3 git bundle\n')
def git(repo,*args,data=None,check=True):
 p=subprocess.run(['git','-C',str(repo),*args],input=data,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 if check and p.returncode:raise RuntimeError('git '+args[0]+': '+p.stderr.decode(errors='replace')[:600])
 return p

def prepare_bundle(path,cache_dir=None):
 path=Path(path);cache=Path(cache_dir or DEFAULT_CACHE);cache.mkdir(parents=True,exist_ok=True)
 with path.open('rb') as f:
  if f.read(16) not in MAGICS:return path,{'state':'not_git_bundle','redactions':0}
  f.seek(0);source_sha=hashlib.file_digest(f,'sha256').hexdigest()
 receipt_path=cache/(source_sha+'.json');dest=cache/(source_sha+'.bundle')
 with (cache/(source_sha+'.lock')).open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if receipt_path.exists():
   receipt=json.loads(receipt_path.read_text())
   return (dest if receipt['redactions'] else path),receipt
  with tempfile.TemporaryDirectory(prefix='bundle-sanitize-') as temp:
   temp=Path(temp);raw=temp/'original.git';clean=temp/'clean.git'
   subprocess.run(['git','clone','--mirror',str(path),str(raw)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
   subprocess.run(['git','init','--bare',str(clean)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
   oldmarks=temp/'old.marks';newmarks=temp/'new.marks'
   exporter=subprocess.Popen(['git','-C',str(raw),'fast-export','--show-original-ids','--signed-tags=strip','--tag-of-filtered-object=rewrite','--export-marks='+str(oldmarks),'--all'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
   importer=subprocess.Popen(['git','-C',str(clean),'fast-import','--quiet','--export-marks='+str(newmarks)],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
   kind=None;oid=None;redactions=0;nested=[];redacted_blobs=[]
   try:
    while True:
     line=exporter.stdout.readline()
     if not line:break
     if line==b'blob\n':kind='blob';oid=None
     elif line.startswith(b'commit '):kind='commit';oid=None
     elif line.startswith(b'tag '):kind='tag';oid=None
     if line.startswith(b'original-oid '):oid=line.split()[1].decode();continue
     if line.startswith(b'data '):
      size=int(line.split()[1]);data=exporter.stdout.read(size)
      if len(data)!=size:raise RuntimeError('Truncated fast-export data')
      if kind=='blob' and oid==KNOWN_BLOB:
       obj=json.loads(data);assert 'private_key' in obj;obj['private_key']='[REDACTED_CREDENTIAL]';data=(json.dumps(obj,indent=2)+'\n').encode();redactions+=1;redacted_blobs.append({'original':oid,'sanitized':hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()})
      elif kind=='blob' and data.startswith(MAGICS):
       nested_path=temp/('nested-'+str(len(nested))+'.bundle');nested_path.write_bytes(data);replacement,record=prepare_bundle(nested_path,cache)
       if record['redactions']:data=replacement.read_bytes();redactions+=record['redactions'];nested.append(record['source_sha256'])
      importer.stdin.write(b'data '+str(len(data)).encode()+b'\n');importer.stdin.write(data)
     else:importer.stdin.write(line)
    importer.stdin.close()
    err_export=exporter.stderr.read();err_import=importer.stderr.read()
    if exporter.wait()!=0:raise RuntimeError('fast-export failed: '+err_export.decode(errors='replace')[:600])
    if importer.wait()!=0:raise RuntimeError('fast-import failed: '+err_import.decode(errors='replace')[:600])
   except BaseException:
    exporter.kill();importer.kill();exporter.wait();importer.wait();raise
   receipt={'state':'sanitized' if redactions else 'unchanged','source_sha256':source_sha,'source_bytes':path.stat().st_size,'redactions':redactions,'redacted_blobs':redacted_blobs,'nested_bundles':nested}
   if redactions:
    assert git(clean,'cat-file','-e',KNOWN_BLOB,check=False).returncode!=0,'Original secret blob remains'
    git(clean,'fsck','--full')
    original_heads=git(raw,'for-each-ref','--format=%(refname) %(objectname)').stdout.decode().splitlines()
    clean_heads=git(clean,'for-each-ref','--format=%(refname) %(objectname)').stdout.decode().splitlines()
    assert {r.split()[0] for r in original_heads}=={r.split()[0] for r in clean_heads},'Lost ref names'
    old=dict(line.split() for line in oldmarks.read_text().splitlines());new=dict(line.split() for line in newmarks.read_text().splitlines());map_path=cache/(source_sha+'.commit-map.csv')
    with map_path.open('w') as f:
     writer=csv.writer(f);writer.writerow(['old','new']);writer.writerows((oid,new[mark]) for mark,oid in sorted(old.items()) if mark in new)
    temp_dest=cache/(source_sha+'.bundle.tmp');git(clean,'bundle','create',str(temp_dest),'--all');os.replace(temp_dest,dest)
    # Independently reopen the resulting bundle and prove the sensitive object is absent.
    verify=temp/'verify.git';subprocess.run(['git','clone','--mirror',str(dest),str(verify)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    assert git(verify,'cat-file','-e',KNOWN_BLOB,check=False).returncode!=0
    receipt.update(sanitized_sha256=hashlib.file_digest(dest.open('rb'),'sha256').hexdigest(),sanitized_bytes=dest.stat().st_size,commit_map=map_path.name,original_refs=original_heads,sanitized_refs=clean_heads,verified_secret_blob_absent=True)
   else:receipt.update(sanitized_sha256=source_sha,sanitized_bytes=path.stat().st_size,verified_secret_blob_absent=True)
   tmp=receipt_path.with_suffix('.json.tmp');tmp.write_text(json.dumps(receipt,indent=2)+'\n');os.replace(tmp,receipt_path)
   return (dest if redactions else path),receipt

def sanitize_bundle_file(path,cache_dir=None):
 replacement,receipt=prepare_bundle(path,cache_dir);return replacement.open('rb'),receipt['redactions']

def sanitize_bundle_stream(stream,cache_dir=None):
 with tempfile.NamedTemporaryFile(suffix='.bundle') as tmp:
  shutil.copyfileobj(stream,tmp);tmp.flush();replacement,receipt=prepare_bundle(tmp.name,cache_dir)
  if receipt['redactions']:return replacement.open('rb'),receipt['redactions']
  result=tempfile.TemporaryFile();tmp.seek(0);shutil.copyfileobj(tmp,result);result.seek(0);return result,0

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('paths',nargs='+');p.add_argument('--cache-dir');a=p.parse_args()
 for source in a.paths:
  replacement,receipt=prepare_bundle(source,a.cache_dir);print(json.dumps({'source':source,'replacement':str(replacement),**receipt}),flush=True)
