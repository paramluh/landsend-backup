import concurrent.futures
import json
from pathlib import Path
import shlex
import subprocess

root = Path('/home/param/outputs/pose-opd-credit-bench')
ssh = ['ssh', '-i', '/home/param/.ssh/id_ed25519_cluster', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes']
def start(pair):
    host, variant = pair
    subprocess.run(['rsync', '-a', '-e', shlex.join(ssh), str(root / 'actor_worker.py'), host + ':' + str(root / 'actor_worker.py')], check=True)
    script = """import pathlib,subprocess,json
root=pathlib.Path(%r)
job=root/%r
job.mkdir(exist_ok=True)
with (job/'started').open('x') as marker:marker.write('started')
with (job/'worker.log').open('ab',buffering=0) as log:
 p=subprocess.Popen(['python3',str(root/'actor_worker.py'),%r],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
print(json.dumps({'pid':p.pid}))
""" % (str(root), 'actor-' + variant, variant)
    output = subprocess.check_output(ssh + [host, shlex.join(['python3', '-c', script])], text=True)
    print(host, variant, output.strip(), flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(start, [('landsend-2', 'linear'), ('landsend-3', 'tuned')]))
