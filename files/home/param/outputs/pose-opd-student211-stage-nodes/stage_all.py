from pathlib import Path
import concurrent.futures,json,subprocess,time
base=Path('/home/param/outputs/pose-opd-student211-stage-nodes')
started=time.time()
def run(node):
 print(json.dumps({'event':'node_begin','node':node,'seconds':time.time()-started}),flush=True)
 with (base/f'node{node}'/'run.log').open('wb') as log:
  p=subprocess.run(['/home/param/pose-repo/.venv/bin/python',str(base/f'node{node}'/'stage.py')],stdout=log,stderr=subprocess.STDOUT)
 print(json.dumps({'event':'node_complete','node':node,'returncode':p.returncode,'seconds':time.time()-started}),flush=True)
 return {'node':node,'returncode':p.returncode}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 results=list(pool.map(run,[1,2,3,4]))
(base/'summary.json').write_text(json.dumps({'seconds':time.time()-started,'results':results},indent=2))
assert all(r['returncode']==0 for r in results),results
