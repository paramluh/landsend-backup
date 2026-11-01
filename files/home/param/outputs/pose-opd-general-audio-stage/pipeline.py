import concurrent.futures,json,subprocess,time
from pathlib import Path
out=Path(__file__).parent
root=Path('/home/param/models/imported-weights/ppm-aloe-zoom-cotrain')
started=time.time()
while not (root/'HF_IMPORT_RECEIPT.json').exists():
 if time.time()-started>600: raise TimeoutError('Waiting for general checkpoint download')
 time.sleep(2)
subprocess.run(['python3','/home/param/outputs/pose-opd-rosie-site-audit/verify_general_parent.py'],check=True)
def stage(node):
 with (out/f'node{node}.log').open('wb') as log:
  result=subprocess.run(['python3',str(out/'stage_node.py'),str(node)],stdout=log,stderr=subprocess.STDOUT)
 if result.returncode: raise RuntimeError(f'Node{node} stage failed: {out}/node{node}.log')
 receipt=json.loads((out/f'node{node}/receipt.json').read_text());print(json.dumps(receipt),flush=True);return receipt
results=[stage(5)]
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:results+=list(pool.map(stage,[1,2,3,4,12]))
summary={'seconds':time.time()-started,'results':results}
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
