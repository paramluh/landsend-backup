import concurrent.futures, datetime, hashlib, json, os, pathlib, shutil, socket, subprocess, time
assert socket.gethostname().split('.')[0] == 'landsend-5'
ROOT=pathlib.Path('/home/param/pose-runs/duplex-distill-demo-deploy-20261030')
SOURCE=pathlib.Path('/home/param/pose-runs/duplex-distill-7n-shared10-mbs2-20261029/checkpoints/step-000284/export')
TARGET=pathlib.Path('/home/param/models/duplex-distill-step-000284-inference')
STAGING=TARGET.with_name(TARGET.name+'.staging')
META=json.loads((ROOT/'source-meta.json').read_text())
SSH=['ssh','-i','/home/param/.ssh/id_ed25519_cluster','-o','BatchMode=yes','-o','ConnectTimeout=10']
HOSTS=[f'landsend-{i}' for i in (0,1,2,3,4,5,12)]
entries=[e for e in META['files'] if e.get('section')=='export']
reps=[e for e in entries if e['file']=='rep.pt']
assert len(reps)==7 and len({(e['bytes'],e['sha256']) for e in reps})==1
selected=[e for e in entries if 'rank' in e]+[next(e for e in reps if e['host']=='landsend-0')]
assert len(selected)==57 and sorted(e['rank'] for e in selected if 'rank' in e)==list(range(56))
assert META['step']==284 and META['world']==56
assert not TARGET.exists(), str(TARGET)
STAGING.mkdir(parents=True,exist_ok=True)
assert shutil.disk_usage(STAGING).free > sum(e['bytes'] for e in selected)+100*1024**3
started=datetime.datetime.now(datetime.timezone.utc).isoformat()
def status(phase,**kw):
 doc={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'started_utc':started,'phase':phase,'target':str(TARGET),'staging':str(STAGING),'expected_files':57,'expected_bytes':sum(e['bytes'] for e in selected),**kw}
 p=ROOT/'gather-status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(doc,indent=2)+'\n');t.replace(p);print(json.dumps(doc),flush=True)
status('copying')
def copy(host):
 local=[e for e in selected if e['host']==host]
 names=[e['file'] for e in local]
 assert all(pathlib.Path(n).name==n for n in names)
 filelist=ROOT/f'files-{host}.txt';filelist.write_text('\n'.join(names)+'\n')
 src=str(SOURCE)+'/' if host=='landsend-5' else f'{host}:{SOURCE}/'
 command=['rsync','-a','--partial','--info=stats2','--files-from',str(filelist),'-e','ssh -i /home/param/.ssh/id_ed25519_cluster -o BatchMode=yes -o ConnectTimeout=10',src,str(STAGING)+'/']
 with (ROOT/f'copy-{host}.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
 print(json.dumps({'host':host,'copied_files':len(local),'bytes':sum(e['bytes'] for e in local)}),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool:list(pool.map(copy,HOSTS))
status('hashing')
def verify(e):
 p=STAGING/e['file'];assert p.is_file() and not p.is_symlink();before=p.stat();assert before.st_size==e['bytes'];h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 after=p.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
 assert h.hexdigest()==e['sha256'],e['file']
 return dict(e,actual_sha256=h.hexdigest())
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:verified=list(pool.map(verify,selected))
# Preserve the checkpoint config and scalars exactly; only gathered storage provenance changes.
meta=dict(META);meta['storage_layout']='gathered-shards';meta['source_storage_layout']=META['storage_layout'];meta['source_host']=META['host'];meta['host']='landsend-5';meta['files']=selected
meta['gathered_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();meta['source_export_dir']=str(SOURCE)
(STAGING/'meta.json').write_text(json.dumps(meta,indent=2)+'\n')
(STAGING/'source-meta.json').write_text((ROOT/'source-meta.json').read_text())
audit={'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'step':284,'world':56,'all_checks_pass':True,'files_verified':len(verified),'bytes_sha256_verified':sum(e['bytes'] for e in verified),'source_export_dir':str(SOURCE),'files':verified,'cfg_unchanged':meta['cfg']==META['cfg'],'vgain_theta_fp32_unchanged':meta.get('vgain_theta_fp32')==META.get('vgain_theta_fp32'),'decp_bb_alpha_unchanged':meta.get('decp_bb_alpha')==META.get('decp_bb_alpha')}
(STAGING/'VERIFIED.json').write_text(json.dumps(audit,indent=2)+'\n')
(STAGING/'COMPLETE').write_text('284\n')
for p in [STAGING/'meta.json',STAGING/'VERIFIED.json',STAGING/'COMPLETE']:
 with p.open('rb') as f:os.fsync(f.fileno())
STAGING.rename(TARGET)
(ROOT/'export-verification.json').write_text(json.dumps(audit,indent=2)+'\n')
status('complete',all_checks_pass=True,files_verified=57,bytes_sha256_verified=audit['bytes_sha256_verified'])
