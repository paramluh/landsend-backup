import fcntl, importlib.util, json, os, pathlib, statistics, subprocess, time
import torch
import torch.nn.functional as F

root=pathlib.Path(__file__).parent
lock=open('/home/param/runs/pose-runs/.duplex-distill-gpus.lock','a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
if apps:raise RuntimeError('GPU busy: '+apps)
spec=importlib.util.spec_from_file_location('light_kernels',root/'light_kernels.py')
LK=importlib.util.module_from_spec(spec);spec.loader.exec_module(LK)
torch.manual_seed(73418)
torch.set_grad_enabled(False)
shapes=[(4096,1024),(2048,1024),(6144,1024),(1024,2048),(1024,3072)]
configs=['cublaslt']
records=[]
for m in (24,48):
 for n,k in shapes:
  weights=[torch.randn(n,k,device='cuda',dtype=torch.bfloat16) for _ in range(32)]
  xs=[torch.randn(m,k,device='cuda',dtype=torch.bfloat16) for _ in range(32)]
  ys=[torch.empty(m,n,device='cuda',dtype=torch.bfloat16) for _ in weights]
  reference=F.linear(xs[0],weights[0])
  for cfg in [None]+configs:
   torch.backends.cuda.preferred_blas_library('cublas' if cfg is None else cfg)
   def run():
    for x,w,y in zip(xs,weights,ys):
     if cfg is None:F.linear(x,w,out=y)
     else:
      bn,bk,warps,stages=cfg
      LK._gemv_k.fn[((n+bn-1)//bn,)](x,w,y,m,n,k,BM=32 if m==24 else 64,BN=bn,BK=bk,num_warps=warps,num_stages=stages)
   # F.linear does not expose out; use mathematically identical addmm-free mm.
   if True:
    def run():
     for x,w,y in zip(xs,weights,ys):torch.mm(x,w.T,out=y)
   try:
    run();torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):run()
    for _ in range(3):g.replay()
    timings=[]
    for _ in range(10):
     a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
     a.record();g.replay();b.record();b.synchronize();timings.append(a.elapsed_time(b)/32)
    err=(ys[0].float()-reference.float())
    rec={'m':m,'n':n,'k':k,'config':cfg,'milliseconds':statistics.median(timings),'relative_rms':float(err.square().mean().sqrt()/reference.float().square().mean().sqrt()),'max_abs':float(err.abs().max())}
    del g
   except Exception as exc:rec={'m':m,'n':n,'k':k,'config':cfg,'error':repr(exc)}
   records.append(rec);print(json.dumps(rec),flush=True)
   (root/'results-blas.json').write_text(json.dumps(records,indent=2))
  del weights,xs,ys,reference
print('COMPLETE',flush=True)
