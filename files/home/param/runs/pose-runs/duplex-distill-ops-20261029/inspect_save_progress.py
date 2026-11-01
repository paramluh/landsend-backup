import concurrent.futures,datetime,json,pathlib,subprocess
run=pathlib.Path('/home/param/pose-runs/duplex-distill-7n-shared10-mbs2-20261029')
probe=r'''
import json,pathlib,socket,subprocess
root=pathlib.Path('/home/param/pose-runs/duplex-distill-7n-shared10-mbs2-20261029');job=root/'cluster/latest';out={'host':socket.gethostname(),'finished':json.loads((job/'finished.json').read_text()) if (job/'finished.json').exists() else None,'checkpoints':{},'gpu_pids':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip().splitlines()}
for d in (root/'checkpoints').glob('step-*'):
 files=[p for p in d.glob('*/*') if p.is_file()];out['checkpoints'][d.name]={'files':len(files),'bytes':sum(p.stat().st_size for p in files),'complete':(d/'resume/COMPLETE').exists(),'node_complete':(d/'export/NODE_COMPLETE').exists()}
out['latest_resume']=json.loads((root/'checkpoints/LATEST_RESUME.json').read_text()) if (root/'checkpoints/LATEST_RESUME.json').exists() else None
out['errors']=[l[:1200] for l in (job/'node.log').read_text(errors='replace').splitlines() if any(e in l for e in ('Traceback (most recent call last)','RuntimeError:','OutOfMemoryError','AssertionError:'))][-5:]
print(json.dumps(out))
'''
def read(h):
 r=subprocess.run(['ssh','-i','/home/param/.ssh/id_ed25519_cluster',h,'python3 -'],input=probe,text=True,capture_output=True,timeout=25,check=True);return h,json.loads(r.stdout)
with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool:nodes=dict(pool.map(read,[f'landsend-{i}' for i in (0,1,2,3,4,5,12)]))
a={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'nodes':nodes};(run/'verification/checkpoint-save-status.json').write_text(json.dumps(a,indent=2)+'\n')
for h,n in nodes.items():print(h,json.dumps(n))
