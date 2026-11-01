"""Enforce the user's15-minute budget against only these owned synthetic jobs."""
import concurrent.futures,json,pathlib,subprocess,sys,time
sys.path.insert(0,'/home/param/pose-worktrees/pose-basic-opd/finetune/pose_opd_credit/ops/basic_launch')
from common import HOSTS,remote
root=pathlib.Path(__file__).resolve().parent
budget=json.loads((root/'BUDGET.json').read_text())
while time.time()<budget['deadline_unix']:
    if json.loads((root/'BUDGET.json').read_text()).get('ended_unix'):sys.exit(0)
    time.sleep(min(10,max(.01,budget['deadline_unix']-time.time())))
script='''import json,os,pathlib,signal
out=[]
for suffix in ('a','b','c','d'):
 job=pathlib.Path('/home/param/runs/smokes/pose-basic-opd-synthetic-20261031-'+suffix)/'supervisor'
 if not (job/'supervisor.json').exists() or (job/'finished.json').exists():continue
 spec=json.loads((job/'spec.json').read_text());assert '--no-save' in spec['extra_args'] and '--benchmark-shape-only' in spec['extra_args']
 row=json.loads((job/'supervisor.json').read_text());p=pathlib.Path('/proc/%d/stat'%row['pid'])
 if p.exists():
  fields=p.read_text().rsplit(')',1)[1].split()
  if int(fields[19])==row['start_ticks'] and fields[0]!='Z':os.kill(row['pid'],signal.SIGTERM);out.append({'run':suffix,'pid':row['pid']})
print(json.dumps(out))
'''
def stop(host):
    try:return {'host':host,'result':remote(host,['python3','-c',script]).stdout}
    except Exception as error:return {'host':host,'error':str(error)}
with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool: receipts=list(pool.map(stop,HOSTS))
(root/'DEADLINE-ENFORCED.json').write_text(json.dumps({'time':time.time(),'receipts':receipts},indent=2)+'\n')
