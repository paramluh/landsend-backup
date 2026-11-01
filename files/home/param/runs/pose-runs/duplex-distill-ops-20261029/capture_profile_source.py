import concurrent.futures,datetime,hashlib,json,shlex,subprocess,sys
from pathlib import Path
run=Path(sys.argv[1]);repo=Path('/home/param/pose-repo')
files=['train.py','runtime.py','losses.py','recipe.py','metrics.py','checkpoints.py','data.py','profiling.py','benchmark_text_head.py','profile_packing.py']
paths=[str(repo/'finetune/duplex_distill'/name) for name in files]+[str(run/'config.json'),str(run/'plan.json')]
probe='import hashlib,json,pathlib,sys;print(json.dumps({p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() if pathlib.Path(p).exists() else None for p in json.loads(sys.argv[1])}))'
def get(host):
 out=subprocess.check_output(['ssh','-i','/home/param/.ssh/id_ed25519_cluster','-o','BatchMode=yes',host,'python3 -c '+shlex.quote(probe)+' '+shlex.quote(json.dumps(paths))],text=True,timeout=30)
 return host,json.loads(out)
hosts=json.loads((run/'cluster/latest/launch.json').read_text())['hosts']
with concurrent.futures.ThreadPoolExecutor(max_workers=len(hosts)) as pool:nodes=dict(pool.map(get,hosts))
controller={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths[:-2]}
label=sys.argv[2] if len(sys.argv)>2 else None
snap=run/'verification'/(label if label else 'launch-source');snap.mkdir(parents=True,exist_ok=True)
for name in files:
 subprocess.run(['scp','-q','-i','/home/param/.ssh/id_ed25519_cluster','landsend-1:'+str(repo/'finetune/duplex_distill'/name),str(snap/name)],check=True)
snapshot={str(repo/'finetune/duplex_distill'/name):hashlib.sha256((snap/name).read_bytes()).hexdigest() for name in files}
report={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'snapshot_source':'landsend-1 after launch, before next source sync','snapshot_sha256':snapshot,'controller_source_sha256':controller,'nodes':nodes,
 'all_source_hashes_match_snapshot':all(all(n[p]==sha for p,sha in snapshot.items()) for n in nodes.values()),
 'all_source_hashes_match_controller':all(all(n[p]==sha for p,sha in controller.items()) for n in nodes.values()),
 'all_configs_identical':len({n[paths[-2]] for n in nodes.values()})==1 and all(n[paths[-2]] for n in nodes.values()),
 'all_plans_identical':len({n[paths[-1]] for n in nodes.values()})==1 and all(n[paths[-1]] for n in nodes.values()),
 'mismatches':{host:[p for p,sha in snapshot.items() if n[p]!=sha] for host,n in nodes.items()}}
(run/'verification'/('source-config-audit-'+label+'.json' if label else 'source-config-audit.json')).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('snapshot_sha256','controller_source_sha256','nodes')},indent=2))
