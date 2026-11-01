import json,pathlib,shlex,subprocess,time,concurrent.futures
root=pathlib.Path(__file__).parent;spec=json.load(open(root/'launch.json'));done=set();deadline=time.monotonic()+900

def state(x):
 code="import pathlib,json; r=pathlib.Path("+repr(x['run'])+"); p=r/'result.json'; print(p.read_text() if p.exists() else 'null')"
 return x,json.loads(subprocess.check_output(['ssh','-i','/home/param/.ssh/id_ed25519_cluster','-o','BatchMode=yes',x['host'],shlex.join(['python3','-c',code])],text=True))
while len(done)<len(spec['nodes']) and time.monotonic()<deadline:
 with concurrent.futures.ThreadPoolExecutor(7) as pool:states=list(pool.map(state,[x for x in spec['nodes'] if x['host'] not in done]))
 for x,result in states:
  if result is None:continue
  if result['codes']!=[0,0]:raise RuntimeError((x['host'],result))
  for job in x['jobs']:
   dst=root/'collected'/x['host']/job['name'];dst.mkdir(parents=True,exist_ok=True)
   subprocess.run(['rsync','-a','-e','ssh -i /home/param/.ssh/id_ed25519_cluster -o BatchMode=yes',f"{x['host']}:{x['run']}/{job['name']}/",str(dst)+'/'],check=True)
  done.add(x['host']);print(json.dumps({'completed_collected':x['host'],'bias':x['bias'],'count':len(done)}),flush=True)
  (root/'collected-status.json').write_text(json.dumps({'completed':sorted(done),'time':time.time()}))
 if len(done)<len(spec['nodes']):time.sleep(10)
if len(done)!=len(spec['nodes']):raise TimeoutError('Sweep still running; inspect exact processes before changing anything')
