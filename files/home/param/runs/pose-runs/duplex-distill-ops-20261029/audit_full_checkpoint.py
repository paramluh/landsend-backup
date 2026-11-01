import concurrent.futures,datetime,json,pathlib,shlex,subprocess,sys
RUN=pathlib.Path('/home/param/pose-runs/duplex-distill-7n-shared10-mbs2-20261029');STEP=int(sys.argv[1]);HOSTS=[f'landsend-{i}' for i in (0,1,2,3,4,5,12)]
PROBE=r'''
import concurrent.futures,datetime,hashlib,json,pathlib,socket,sys
root=pathlib.Path('/home/param/pose-runs/duplex-distill-7n-shared10-mbs2-20261029');step=int(sys.argv[1]);host=socket.gethostname().split('.')[0];ck=root/'checkpoints'/f'step-{step:06d}'
meta=json.loads((ck/'resume/train_state.json').read_text());exp=json.loads((ck/'export/meta.json').read_text());complete=(ck/'resume/COMPLETE').read_text().strip();nodecomplete=json.loads((ck/'export/NODE_COMPLETE').read_text());latest=json.loads((root/'checkpoints/LATEST_RESUME.json').read_text())
assert complete==str(step) and meta['step']==exp['step']==nodecomplete['step']==step
assert meta['world']==exp['world']==nodecomplete['world']==56 and meta['host']==exp['host']==nodecomplete['host']==host
entries=[e for e in meta['files'] if e['host']==host];assert len(entries)==18
assert {e['file'] for e in entries if e['section']=='resume'}=={'rep.pt',*[f'rank{r}.pt' for r in meta['local_ranks']]}
assert {e['file'] for e in entries if e['section']=='export'}=={'rep.pt',*[f'rank{r}.pt' for r in meta['local_ranks']]}
def verify(e):
 assert pathlib.Path(e['file']).name==e['file'] and e['section'] in ('resume','export')
 p=ck/e['section']/e['file'];assert p.is_file() and not p.is_symlink();before=p.stat();assert before.st_size==e['bytes'];h=hashlib.sha256()
 with p.open('rb') as f:
  for buf in iter(lambda:f.read(8*1024*1024),b''):h.update(buf)
 after=p.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
 assert h.hexdigest()==e['sha256'],str(p)
 return dict(e,actual_sha256=h.hexdigest(),owner_uid=after.st_uid,mtime_ns=after.st_mtime_ns)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:verified=list(pool.map(verify,entries))
markers={name:{'bytes':(ck/name).stat().st_size,'sha256':hashlib.sha256((ck/name).read_bytes()).hexdigest()} for name in ('resume/COMPLETE','resume/train_state.json','export/NODE_COMPLETE','export/meta.json')}
print(json.dumps({'host':host,'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'step':step,'world':meta['world'],'local_ranks':meta['local_ranks'],'complete':True,'files_verified':verified,'total_bytes':sum(e['bytes'] for e in verified),'markers':markers,'latest_resume':latest,'runtime_state':meta['runtime_state'],'full_manifest_sha256':hashlib.sha256(json.dumps(meta['files'],sort_keys=True).encode()).hexdigest()}))
'''
def get(host):
 r=subprocess.run(['ssh','-i','/home/param/.ssh/id_ed25519_cluster','-o','BatchMode=yes',host,'python3 - '+shlex.quote(str(STEP))],input=PROBE,capture_output=True,text=True,timeout=600)
 if r.returncode:raise RuntimeError(r.stderr[-5000:])
 return json.loads(r.stdout)
nodes={};errors={}
with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool:
 tasks={pool.submit(get,h):h for h in HOSTS}
 for f in concurrent.futures.as_completed(tasks):
  h=tasks[f]
  try:nodes[h]=f.result();print(json.dumps({'host':h,'sha256_verified_files':len(nodes[h]['files_verified']),'bytes':nodes[h]['total_bytes']}),flush=True)
  except Exception as e:errors[h]=str(e);print(json.dumps({'host':h,'error':str(e)}),flush=True)
files=[e for n in nodes.values() for e in n['files_verified']]
checks={'all7nodes_verified':len(nodes)==7 and not errors,'all126payloadfiles_verified':len(files)==126,'all56resume_rank_shards':sorted(e['rank'] for e in files if e['section']=='resume' and 'rank' in e)==list(range(56)),'all56export_rank_shards':sorted(e['rank'] for e in files if e['section']=='export' and 'rank' in e)==list(range(56)),'all_nodes_global_file_manifests_match':len({n['full_manifest_sha256'] for n in nodes.values()})==1 and len(nodes)==7}
report={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'run':str(RUN),'step':STEP,'all_checks_pass':all(checks.values()),'checks':checks,'bytes_sha256_verified':sum(n['total_bytes'] for n in nodes.values()),'files_sha256_verified':len(files),'errors':errors,'nodes':{h:nodes[h] for h in HOSTS if h in nodes}}
p=RUN/'verification'/f'checkpoint-step-{STEP:06d}-audit.json';p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ('all_checks_pass','checks','bytes_sha256_verified','files_sha256_verified','errors')}),flush=True)
