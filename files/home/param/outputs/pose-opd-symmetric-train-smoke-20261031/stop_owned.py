"""Stop only exact bounded-smoke torchrun process groups from saved receipts."""
import concurrent.futures,json,shlex,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
latest=json.loads((HERE/'LATEST.json').read_text());local=Path(latest['local']);root=latest['remote'];spec=json.loads((local/'spec.json').read_text())
if not spec['no_save'] or spec['max_updates']!=1 or 'symmetric-train-smoke' not in root:raise RuntimeError('Not bounded smoke')
script='''import json,os,signal,time
from pathlib import Path
root=Path(ROOT)
spec=json.loads((root/'spec.json').read_text())
if not spec['no_save'] or spec['max_updates']!=1:raise RuntimeError('Not bounded smoke')
receipt=json.loads((root/'torchrun.json').read_text());pid=receipt['pid'];p=Path('/proc')/str(pid)
result={'pid':pid,'start_ticks':receipt['start_ticks'],'signalled':False,'time':time.time()}
if (p/'stat').exists():
 fields=(p/'stat').read_text().rsplit(')',1)[1].split();cmd=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode()
 if int(fields[19])!=receipt['start_ticks']:raise RuntimeError('PID reused')
 if str(root) not in cmd or 'torch.distributed.run' not in cmd:raise RuntimeError('Wrong owned command')
 if fields[0]!='Z':os.killpg(pid,signal.SIGTERM);result['signalled']=True
(root/'STOP-REQUESTED.json').write_text(json.dumps(result,indent=2)+'\\n')
print(json.dumps(result))
'''.replace('ROOT',repr(root))
ssh=['ssh','-i','/home/param/.ssh/id_ed25519_cluster','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10']
def one(host):
 p=subprocess.run(ssh+[host,shlex.join([spec['python'],'-c',script])],capture_output=True,text=True,timeout=20)
 return {'host':host,'exit_code':p.returncode,'result':p.stdout,'error':p.stderr}
with concurrent.futures.ThreadPoolExecutor(7) as pool:out=list(pool.map(one,spec['hosts']))
(local/'STOP-REQUESTED.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
