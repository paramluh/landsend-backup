import concurrent.futures,subprocess,shlex,pathlib,json,sys
spec=json.loads(pathlib.Path(sys.argv[1]).read_text());run=spec['run_dir']
script="import pathlib,time,subprocess\nr=pathlib.Path("+repr(run)+")\nwith (r/'gpu-samples.csv').open('a',buffering=1) as f:\n while not (r/'supervisor/finished.json').exists():\n  q=subprocess.check_output(['nvidia-smi','--query-gpu=timestamp,index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)\n  f.write(q);time.sleep(1)\n"
launcher="import subprocess,pathlib,json\nr=pathlib.Path("+repr(run)+")\nassert not (r/'gpu-sampler.json').exists()\np=subprocess.Popen(['python3','-c',"+repr(script)+"],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)\n(r/'gpu-sampler.json').write_text(json.dumps({'pid':p.pid}))\nprint(p.pid)"
def f(h):
 out=subprocess.check_output(['ssh','-i','/home/param/.ssh/id_ed25519_cluster',h,shlex.join(['python3','-c',launcher])],text=True);return h,out.strip()
with concurrent.futures.ThreadPoolExecutor(max_workers=len(spec['hosts'])) as ex:
 for x in ex.map(f,spec['hosts']):print(*x)
