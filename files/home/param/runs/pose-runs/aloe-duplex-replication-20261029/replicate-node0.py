import concurrent.futures,fcntl,hashlib,json,os,pathlib,shlex,shutil,socket,subprocess,time
CONTROL=pathlib.Path(__file__).resolve().parent
ROOT=pathlib.Path('/home/param/datasets/aloe-duplex')
STAGE=ROOT.with_name('.aloe-duplex-staging-20261029')
HOST=socket.gethostname().split('.')[0]
SSH=['ssh','-i','/home/param/.ssh/id_ed25519_cluster','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10']
assert HOST == 'landsend-0'
lock=(CONTROL/'replicate.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def now():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def atomic(path,value):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
manifest=json.loads((CONTROL/'source-files.json').read_text());files=manifest['files']
assert not ROOT.exists(), 'Final dataset already exists; inspect before changing it'
assert shutil.disk_usage(ROOT.parent).free > sum(e['size'] for e in files)+80*10**9
STAGE.mkdir(exist_ok=True)
state={'host':HOST,'source':'landsend-1','started_utc':now(),'phase':'copying','root':str(ROOT),
 'staging':str(STAGE),'files_total':len(files),'bytes_total':sum(e['size'] for e in files),
 'files_copied':0,'bytes_copied':0,'files_sha256_verified':0,'bytes_sha256_verified':0,'errors':[]}
def report(scan=False):
 if scan:
  present=[e for e in files if (STAGE/e['path']).is_file() and (STAGE/e['path']).stat().st_size==e['size']]
  state['files_copied']=len(present);state['bytes_copied']=sum(e['size'] for e in present)
 state['updated_utc']=now();atomic(CONTROL/('status-'+HOST+'.json'),state)
 print(json.dumps(state),flush=True)
def copy_partition(i,entries):
 listfile=CONTROL/f'copy-files-{i}.nul'
 listfile.write_bytes(b'\0'.join(e['path'].encode() for e in sorted(entries,key=lambda e:e['path']))+b'\0')
 cmd=['rsync','-a','--no-owner','--no-group','--partial','--partial-dir=.rsync-partial',
      '--from0','--files-from='+str(listfile),'--stats','-e',shlex.join(SSH),
      'landsend-1:'+str(ROOT)+'/',str(STAGE)+'/']
 with (CONTROL/f'rsync-{i}.log').open('a') as log:
  proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
 if proc.returncode:raise RuntimeError(f'rsync partition {i} exited {proc.returncode}')
def digest(entry):
 path=STAGE/entry['path']; before=path.stat()
 assert before.st_size==entry['size'],entry['path']
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 after=path.stat()
 assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),entry['path']
 assert h.hexdigest()==entry['sha256'],entry['path']
 return entry['size']
try:
 baskets=[[] for _ in range(8)];sizes=[0]*8
 for entry in sorted(files,key=lambda e:e['size'],reverse=True):
  rel=pathlib.PurePosixPath(entry['path']);assert not rel.is_absolute() and '..' not in rel.parts
  i=min(range(8),key=lambda i:sizes[i]);baskets[i].append(entry);sizes[i]+=entry['size']
 report(True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
  pending={pool.submit(copy_partition,i,entries) for i,entries in enumerate(baskets)}
  while pending:
   done,pending=concurrent.futures.wait(pending,timeout=15,return_when=concurrent.futures.FIRST_COMPLETED)
   for f in done:f.result()
   report(True)
 state['phase']='waiting_for_source_hashes';report()
 # Source hashes are made independently while network copies run.
 hashpath=CONTROL/'source-sha256.json'
 for attempt in range(120):
  result=subprocess.run(SSH+['landsend-1','test -f '+shlex.quote(str(hashpath))],capture_output=True)
  if result.returncode==0:
   subprocess.run(['rsync','-a','-e',shlex.join(SSH),'landsend-1:'+str(hashpath),str(hashpath)],check=True)
   break
  time.sleep(5)
 else:raise RuntimeError('Source checksum manifest not ready')
 hashes=json.loads(hashpath.read_text());entries=hashes['files']
 assert {(e['path'],e['size']) for e in entries}=={(e['path'],e['size']) for e in files}
 state['phase']='verifying_sha256';report()
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  futures=[pool.submit(digest,e) for e in entries]
  last=time.monotonic()
  for future in concurrent.futures.as_completed(futures):
   state['bytes_sha256_verified']+=future.result();state['files_sha256_verified']+=1
   if time.monotonic()-last>=10:report();last=time.monotonic()
  report()
 ready_result=subprocess.run(SSH+['landsend-1','cat '+shlex.quote(str(ROOT/'DATASET_READY.json'))],check=True,capture_output=True,text=True)
 ready=json.loads(ready_result.stdout);assert ready['training_ready']
 ready.update(hostname=HOST,completed_utc=now(),replicated_from='landsend-1',replication_verification='SHA256 of every payload and metadata file')
 atomic(STAGE/'DATASET_READY.json',ready)
 state['phase']='complete';state['completed_utc']=now();state['training_ready']=True
 atomic(STAGE/'REPLICA_VERIFIED.json',state)
 assert not ROOT.exists()
 STAGE.rename(ROOT)
 report()
except Exception as exc:
 state['phase']='failed';state['errors'].append({'type':type(exc).__name__,'message':str(exc)[:500]});report();raise
