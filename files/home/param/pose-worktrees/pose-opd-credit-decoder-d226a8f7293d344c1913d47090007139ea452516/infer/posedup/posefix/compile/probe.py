import os, sys, time
sys.path.insert(0, "/data/home/amartyavarma/ppm/aloe-posedup-live/posefix/sim/engexact")
import torch
import torch._dynamo as dyn
import torch._inductor.config as ic
ic.fallback_random = True
ic.emulate_precision_casts = True
import live_ppm_pose as LP
FO, LK, LX = LP.FO, LP.LK, LP.LX
dev = torch.device("cuda:0")
torch.cuda.set_device(dev)
g = torch.Generator(device=dev); g.manual_seed(1)

def samp(z, temp, topk):
    return LP.t_sample(z, g, temp, topk)
z = torch.randn(1, 2048, device=dev)
temp = torch.tensor(1.0, device=dev); topk = torch.tensor(30, device=dev)
for name, fn, args in [("t_sample", samp, (z, temp, topk))]:
    try:
        ex = dyn.explain(fn)(*args)
        print(name, "graphs", ex.graph_count, "breaks", ex.graph_break_count, [str(b.reason)[:200] for b in ex.break_reasons][:3], flush=True)
    except Exception as e:
        print(name, "EXPLAIN FAIL", repr(e)[:400], flush=True)
x = torch.randn(1, 2, 1024, device=dev, dtype=torch.bfloat16); w = torch.randn(1024, device=dev, dtype=torch.bfloat16)
for name, fn, args in [("FO.rms", lambda a: FO.rms(a, w, 1e-6), (x,)),
                       ("FO.add_rms", lambda a: FO.add_rms(a, a, w, 1e-6), (x,)),
                       ("LK.gemv", lambda a: LK.gemv(a[0], torch.randn(4096, 1024, device=dev, dtype=torch.bfloat16)), (x,)),
                       ("LX.add_rms_f32", lambda a: LX.add_rms_f32(a[0].float(), None, w.float(), 1e-6), (x,))]:
    try:
        ex = dyn.explain(fn)(*args)
        print(name, "graphs", ex.graph_count, "breaks", ex.graph_break_count, [str(b.reason)[:200] for b in ex.break_reasons][:3], flush=True)
    except Exception as e:
        print(name, "EXPLAIN FAIL", repr(e)[:400], flush=True)
import fa3_op
print("fa3_op", [a for a in dir(fa3_op) if not a.startswith("_")][:20])
