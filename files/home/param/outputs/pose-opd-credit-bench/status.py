import concurrent.futures,json,pathlib,subprocess,shlex,sys
spec=json.loads(pathlib.Path(sys.argv[1]).read_text());run=pathlib.Path(spec['run_dir']);script='''import pathlib,json,subprocess,time
r=pathlib.Path(%r); result={'host':subprocess.check_output(['hostname'],text=True).strip()}
job=r/'supervisor'; result['finished']=json.loads((job/'finished.json').read_text()) if (job/'finished.json').exists() else None
logs=sorted(r.glob('events-rank*.jsonl'))
phases=[]
for p in logs:
 lines=p.read_text().splitlines()
 if lines:
  x=json.loads(lines[-1]);phases.append({k:v for k,v in x.items() if k in ('rank','event','phase','role','update','begin','seconds','time')})
result['last_events']=phases
if (job/'worker.log').exists():result['tail']=(job/'worker.log').read_text(errors='replace').splitlines()[-6:]
result['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().splitlines()
print(json.dumps(result))
''' % str(run)
def f(h):
 p=subprocess.run(['ssh','-i','/home/param/.ssh/id_ed25519_cluster',h,shlex.join(['python3','-c',script])],capture_output=True,text=True);return json.loads(p.stdout) if p.returncode==0 else {'host':h,'error':p.stderr}
with concurrent.futures.ThreadPoolExecutor(max_workers=len(spec['hosts'])) as ex:
 for x in ex.map(f,spec['hosts']):print(json.dumps(x),flush=True)
