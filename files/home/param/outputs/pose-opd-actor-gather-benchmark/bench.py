import argparse,gc,hashlib,json,os,pickle,time,statistics
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);torch.set_num_threads(1)
dist.init_process_group('gloo');cached=pickle.loads(a.prepared.read_bytes());assert len(cached['rows'])==48 and world==8
payload=[(dict(row,dummy=False),None) for row in cached['rows'][rank*6:(rank+1)*6]];del cached;gc.collect()
def fingerprint(value):
 h=hashlib.sha256()
 def visit(v):
  if isinstance(v,dict):
   for k in sorted(v):h.update(str(k).encode());visit(v[k])
  elif isinstance(v,(list,tuple)):
   h.update(str(len(v)).encode())
   for x in v:visit(x)
  elif isinstance(v,np.ndarray):h.update(str(v.dtype).encode());h.update(str(v.shape).encode());h.update(v.tobytes())
  elif isinstance(v,torch.Tensor):visit(v.numpy())
  else:h.update(repr(v).encode())
 visit(value);return h.hexdigest()
expected=[None]*world;dist.all_gather_object(expected,fingerprint(payload));results=[]
for trial in range(7):
 for mode in (('all_gather','gather') if trial%2==0 else ('gather','all_gather')):
  output=[None]*world if mode=='all_gather' or rank==0 else None
  dist.barrier();beg=time.perf_counter()
  if mode=='all_gather':dist.all_gather_object(output,payload)
  else:dist.gather_object(payload,object_gather_list=output,dst=0)
  elapsed=time.perf_counter()-beg;dt=torch.tensor(elapsed,dtype=torch.float64);dist.all_reduce(dt,op=dist.ReduceOp.MAX)
  if output is not None:assert [fingerprint(x) for x in output]==expected
  if mode=='gather':
   received=[None];dist.scatter_object_list(received,output,src=0);assert fingerprint(received[0])==expected[rank]
  if rank==0:
   rec={'trial':trial,'mode':mode,'max_rank_seconds':dt.item(),'warmup':trial==0,'exact_payloads':True,'scatter_back_exact':mode=='gather'};results.append(rec);print(json.dumps(rec),flush=True)
  del output;gc.collect()
if rank==0:
 summary={mode:{'median_seconds':statistics.median(r['max_rank_seconds'] for r in results if r['mode']==mode and not r['warmup']),'samples':[r['max_rank_seconds'] for r in results if r['mode']==mode and not r['warmup']]} for mode in ['all_gather','gather']}
 a.output.write_text(json.dumps({'world':world,'rows':48,'local_rows':6,'input_pickle_bytes':a.prepared.stat().st_size,'results':results,'summary':summary,'all_payload_fingerprints_equal':True,'all_scatter_ownership_equal':True,'cuda_initialized':torch.cuda.is_initialized()},indent=2));print(json.dumps(summary),flush=True)
dist.destroy_process_group()
