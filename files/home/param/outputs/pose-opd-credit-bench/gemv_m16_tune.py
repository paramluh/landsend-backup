import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

lock = open('/home/param/runs/pose-runs/.duplex-distill-gpus.lock', 'a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
    raise RuntimeError('Another GPU process is present; refusing isolated microbenchmark')
source = Path('/home/param/pose-worktrees/opd-bench-actor8-fp32-60s-v6s/infer/engine/ppm/light_kernels.py')
sys.path.insert(0, str(source.parent))
import torch
import torch.nn.functional as F
import triton
import light_kernels as LK

torch.set_num_threads(1)
torch.manual_seed(193)
started = time.time()
configs = ((16,128,4,3), (16,256,4,3), (32,128,4,3), (32,256,4,3),
           (64,128,4,3), (16,512,4,2), (32,64,4,4), (16,128,2,4), (64,64,4,4))

def explicit(x, w, parameters):
    bn,bk,warps,stages = parameters
    m,k = x.shape
    n = w.shape[0]
    y = torch.empty(m,n,dtype=torch.bfloat16,device=x.device)
    LK._gemv_k.fn[(triton.cdiv(n,bn),)](x,w,y,m,n,k,BM=16,BN=bn,BK=bk,
                                       num_warps=warps,num_stages=stages)
    return y

results = []
for name,n,k in (('qkv',4096,1024), ('head',2048,1024), ('gate_up',6144,1024),
                 ('down',1024,3072), ('output',1024,2048)):
    weights = [torch.randn(n,k,dtype=torch.bfloat16,device='cuda').mul_(.02) for _ in range(32)]
    x = torch.randn(16,k,dtype=torch.bfloat16,device='cuda')
    baseline = [LK.gemv(x,w) for w in weights]
    functions = [('native',LK.gemv,None), ('linear',F.linear,None)]
    functions += [(str(parameters),lambda x,w,p=parameters: explicit(x,w,p),parameters) for parameters in configs]
    choices = []
    for label,function,parameters in functions:
        actual = [function(x,w) for w in weights]
        max_abs = max(float((a.float()-b.float()).abs().max()) for a,b in zip(actual,baseline))
        unequal = sum(int((a!=b).sum()) for a,b in zip(actual,baseline))
        count = sum(a.numel() for a in actual)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = [function(x,w) for w in weights]
        for _ in range(5): graph.replay()
        torch.cuda.synchronize()
        samples = []
        for _ in range(5):
            begin,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(20): graph.replay()
            end.record();end.synchronize()
            samples.append(begin.elapsed_time(end)*1000/(20*len(weights)))
        choices.append({'label':label,'parameters':parameters,'microseconds':statistics.median(samples),
                        'max_abs_vs_native':max_abs,'unequal_vs_native':unequal,'count':count})
        del graph,actual
    result = {'name':name,'shape':[n,k],'batch':16,'distinct_weights':32,
              'weight_working_set_bytes':sum(w.numel()*w.element_size() for w in weights),
              'winner':min(choices,key=lambda row:row['microseconds']), 'candidates':choices}
    results.append(result)
    print(json.dumps({'event':'shape_complete',**result}),flush=True)
    del x,weights,baseline

report = {'host':os.uname().nodename,'device':torch.cuda.get_device_name(),'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
          'pid':os.getpid(),'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
          'torch_version':torch.__version__,'elapsed_seconds':time.time()-started,'results':results,
          'notes':['Isolated M16 microbenchmark;32 distinct BF16 weights per graph exceed H100 L2.',
                   'Same native Triton body with nine explicit launch configs; no environment mutation.',
                   'Not whole-actor or training throughput; first depth step has M32 and already uses F.linear.']}
path=Path('/home/param/outputs/pose-opd-credit-bench/gemv-m16-tune.json')
path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'event':'complete','elapsed_seconds':report['elapsed_seconds'],'output':str(path)}),flush=True)
