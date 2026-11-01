import json,os,pathlib,pwd,signal,subprocess,time
UID=1005
SCRIPTS={'/home/faisal/vq/gen_text_cb.py','/home/faisal/vq/worker.sh','/home/faisal/vq/watchdog.py','/home/faisal/vq/start_watchdog.sh'}
def record(pid):
 p=pathlib.Path('/proc')/str(pid)
 fields=(p/'stat').read_text().rsplit(')',1)[1].split()
 uid=int(next(x for x in (p/'status').read_text().splitlines() if x.startswith('Uid:')).split()[1])
 argv=[x.decode() for x in (p/'cmdline').read_bytes().split(bytes([0])) if x]
 return {'pid':int(pid),'uid':uid,'start_ticks':int(fields[19]),'argv':argv,'state':fields[0],'pgid':int(fields[2])}
def matching(r):return r['uid']==UID and bool(SCRIPTS.intersection(r['argv']))
rows=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:r=record(int(p.name))
 except (OSError,ValueError):continue
 if matching(r):rows.append(r)
if not rows:print(json.dumps({'host':os.uname().nodename,'already_clear':True}));raise SystemExit()
# This is the fleet's documented scheduler handoff marker. Keep an existing one.
release=pathlib.Path('/home/faisal/vq/RELEASED')
reason={'t':time.time(),'reason':'Owner-authorized GPU handoff for pose OPD seven-node production training','requester':'param','run':'pose-opd-teacher-partner-b336-general-20261031'}
if not release.exists():
 tmp=release.with_name('RELEASED.pose-opd-'+str(os.getpid()));tmp.write_text(json.dumps(reason)+'\n');os.chown(tmp,UID,UID);os.replace(tmp,release)
actions=[]
def send(r,sig):
 try:live=record(r['pid'])
 except FileNotFoundError:return
 if live['start_ticks']!=r['start_ticks'] or not matching(live):raise RuntimeError('PID identity changed; refusing signal '+str(r['pid']))
 if live['state']=='Z':return
 os.kill(r['pid'],sig);actions.append({'pid':r['pid'],'start_ticks':r['start_ticks'],'signal':signal.Signals(sig).name})
# Stop watchdogs and batch supervisors first so worker teardown cannot schedule more work.
rows.sort(key=lambda r:0 if any('watchdog' in x for x in r['argv']) else 1 if '/home/faisal/vq/worker.sh' in r['argv'] else 2 if 'torch.distributed.run' in r['argv'] else 3)
for r in rows:send(r,signal.SIGTERM)
def remaining():
 live=[]
 for r in rows:
  try:v=record(r['pid'])
  except FileNotFoundError:continue
  if v['start_ticks']==r['start_ticks'] and v['state']!='Z':live.append(r)
 return live
deadline=time.monotonic()+20
while remaining() and time.monotonic()<deadline:time.sleep(.2)
for r in remaining():send(r,signal.SIGKILL)
print(json.dumps({'host':os.uname().nodename,'release_marker':str(release),'verified_targets':rows,'actions':actions,'remaining':remaining(),'gpu_processes':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader,nounits'],text=True).strip().splitlines()}))
