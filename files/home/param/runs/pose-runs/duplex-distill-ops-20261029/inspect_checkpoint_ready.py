import concurrent.futures,datetime,json,pathlib,shlex,subprocess,sys
run=pathlib.Path('/home/param/pose-runs/duplex-distill-7n-shared10-mbs2-20261029');step=int(sys.argv[1]);hosts=[f'landsend-{i}' for i in (0,1,2,3,4,5,12)]
probe=r'''
import hashlib,json,pathlib,socket,sys
root=pathlib.Path('/home/param/pose-runs/duplex-distill-7n-shared10-mbs2-20261029');step=int(sys.argv[1]);host=socket.gethostname().split('.')[0];ck=root/'checkpoints'/f'step-{step:06d}';job=root/'cluster/latest'
meta=json.loads((ck/'resume/train_state.json').read_text());complete=(ck/'resume/COMPLETE').read_text().strip();exp=json.loads((ck/'export/NODE_COMPLETE').read_text());latest=json.loads((root/'checkpoints/LATEST_RESUME.json').read_text());finished=json.loads((job/'finished.json').read_text()) if (job/'finished.json').exists() else None
assert meta['step']==exp['step']==latest['step']==step and complete==str(step)
assert meta['world']==exp['world']==56 and meta['host']==exp['host']==host
local=[e for e in meta['files'] if e['host']==host];assert len(local)==18
for e in local:
 p=ck/e['section']/e['file'];assert p.is_file() and not p.is_symlink() and p.stat().st_size==e['bytes'] and e['bytes']>0
assert len(meta['local_ranks'])==8
print(json.dumps({'host':host,'step':step,'world':56,'local_ranks':meta['local_ranks'],'complete':True,'latest':latest,'finished':finished,'files':local,'total_bytes':sum(e['bytes'] for e in local),'runtime_state':meta['runtime_state'],'global_manifest_sha256':hashlib.sha256(json.dumps(meta['files'],sort_keys=True).encode()).hexdigest()}))
'''
def read(host):
 r=subprocess.run(['ssh','-i','/home/param/.ssh/id_ed25519_cluster','-o','BatchMode=yes',host,'python3 - '+shlex.quote(str(step))],input=probe,text=True,capture_output=True,timeout=25)
 if r.returncode:raise RuntimeError(r.stderr[-3000:])
 return json.loads(r.stdout)
nodes={};errors={}
with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool:
 tasks={pool.submit(read,h):h for h in hosts}
 for f in concurrent.futures.as_completed(tasks):
  h=tasks[f]
  try:nodes[h]=f.result()
  except Exception as e:errors[h]=str(e)
checks={'all7complete_matching_step_latest':len(nodes)==7 and not errors,'all56rankcoverage':sorted(r for n in nodes.values() for r in n['local_ranks'])==list(range(56)),'same_global_manifest_all7':len({n['global_manifest_sha256'] for n in nodes.values()})==1 and len(nodes)==7,'all7exited0':all(n['finished'] and n['finished']['exit_code']==0 for n in nodes.values()) and len(nodes)==7}
a={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'step':step,'all_checks_pass':all(checks.values()),'checks':checks,'errors':errors,'total_payload_bytes':sum(n['total_bytes'] for n in nodes.values()),'nodes':nodes,'note':'Quick size/manifest/marker check only; saved manifests contain writer SHA256 and the running checkpoint code verifies all resume hashes before committing latest pointers. Independent full payload SHA audit follows without delaying restart.'}
p=run/'verification'/f'checkpoint-step-{step:06d}-quick-audit.json';p.write_text(json.dumps(a,indent=2)+'\n');print(json.dumps({k:v for k,v in a.items() if k!='nodes'},indent=2))
