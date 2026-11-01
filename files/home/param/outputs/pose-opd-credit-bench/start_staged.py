import concurrent.futures,json,pathlib,subprocess,shlex,sys
spec=json.loads(pathlib.Path(sys.argv[1]).read_text());job=pathlib.Path(spec['run_dir'])/'supervisor'
ssh=['ssh','-i','/home/param/.ssh/id_ed25519_cluster','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes']
script='''import pathlib,subprocess,json
job=pathlib.Path(%r)
with (job/'started-launch').open('x') as marker:marker.write('launched\\n')
with (job/'worker.log').open('ab',buffering=0) as log:
 p=subprocess.Popen(['python3',str(job/'worker.py'),str(job/'spec.json')],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
print(json.dumps({'supervisor_pid':p.pid}))
''' % str(job)
def start(h):
 p=subprocess.run(ssh+[h,shlex.join(['python3','-c',script])],capture_output=True,text=True,check=True);print(h,p.stdout.strip(),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=len(spec['hosts'])) as ex:list(ex.map(start,spec['hosts']))
