import concurrent.futures,json,subprocess,shlex,pathlib
s=json.load(open('/home/param/outputs/pose-opd-credit-bench/b320-resident-prefetch-v9-launch.json'))
r=s['run_dir']
code="""import pathlib,json
r=pathlib.Path(%r)
ev=[]
for p in r.glob('events-rank*.jsonl'):
 lines=p.read_text().splitlines()
 if lines:ev.append(json.loads(lines[-1]))
f=r/'supervisor/finished.json'
a=[json.loads(l) for p in r.glob('events-rank*.jsonl') for l in p.read_text().splitlines()]
print(json.dumps({'finished':json.loads(f.read_text()) if f.exists() else None,'latest':ev,'prefetch':[e for e in a if e['event'].startswith('row_prefetch')],'profiles':[e for e in a if e['event'] in ('profile','replay_benchmark')],'logtail':(r/'supervisor/worker.log').read_text()[-1000:]}))
"""%r
def f(h):
 x=subprocess.run(['ssh','-i','/home/param/.ssh/id_ed25519_cluster','-o','BatchMode=yes',h,shlex.join(['python3','-c',code])],text=True,capture_output=True,check=True)
 d=json.loads(x.stdout)
 pathlib.Path('/home/param/outputs/pose-opd-credit-bench/v9-status-'+h+'.json').write_text(json.dumps(d,indent=2))
 print(h,json.dumps({'finished':d['finished'],'latest':[(e['rank'],e['event'],e.get('phase')) for e in d['latest']],'prefetch_events':len(d['prefetch']),'profiles':len(d['profiles']),'logtail':d['logtail'][-300:]}),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:list(ex.map(f,s['hosts']))
