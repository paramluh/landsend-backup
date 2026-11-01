"""per-kernel latency of the TP trunk's ops (G=4 shapes, 3 rows), each captured 200x in a graph on one GPU; then the all-reduce round
trip across G GPUs (push of a tiny GEMV + ar_rms, 200 sites per graph)."""
import sys, os, json, time
sys.argv = [sys.argv[0], "/dev/null", "1"]
src = open("/data/home/amartyavarma/ppm/aloe-posedup-live/fastmg/dev/t_tp.py").read()
exec(src[:src.index("# ---- 1. gemv vs F.linear")])
exec(src[src.index("# ---- 4. the TP trunk on a bare engine object"):src.index("# 4a. correctness")])
d = devs[0]
NR = 3
REP = 200


def gtime(fn, dev=d, rep=REP):
    with torch.cuda.device(dev):
        s = torch.cuda.Stream(dev)
        s.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(s):
            fn(); fn()
        torch.cuda.synchronize(dev)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            for _ in range(rep):
                fn()
        torch.cuda.synchronize(dev)
        g.replay(); torch.cuda.synchronize(dev)
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); g.replay(); e1.record(); torch.cuda.synchronize(dev)
        return round(e0.elapsed_time(e1) / rep * 1000, 2)        # us per op


out = {}
G = int(os.environ.get("TPG", 4))
e = bare(G, NR, 1, 9600)
e._mode(True)
e.set_T(100)
s0, ar = e.shards[0], e.ars[0]
Ly = s0["L"][0]
nq, nkv = s0["nq"], s0["nkv"]
with torch.cuda.device(d):
    x = torch.randn(NR, H, device=d).to(BF)
    out["gemv_qkv"] = gtime(lambda: tpk.gemv(x, Ly["wqkv"], s0["qkv"]))
    out["cublas_qkv"] = gtime(lambda: F.linear(x, Ly["wqkv"]))
    out["gemv_gu"] = gtime(lambda: tpk.gemv(x, Ly["wgu"], s0["gu"]))
    out["cublas_gu"] = gtime(lambda: F.linear(x, Ly["wgu"]))
    a = torch.randn(NR, nq * D, device=d).to(BF)
    out["gemv_o_f32"] = gtime(lambda: tpk.gemv(a, Ly["wo"], ar.part[:NR], f32=True))
    out["cublas_o"] = gtime(lambda: F.linear(a, Ly["wo"]))
    out["gemv_d_silu_f32"] = gtime(lambda: tpk.gemv(s0["gu"], Ly["wd"], ar.part[:NR], silu=True, f32=True))
    act = torch.randn(NR, s0["I"], device=d).to(BF)
    out["cublas_d"] = gtime(lambda: F.linear(act, Ly["wd"]))
    out["ar_reduce_local"] = gtime(lambda: ar.reduce(1, s0["res"], Ly["ln2"], 1e-6, s0["x2"], NR))
    T3 = e.T3[d]
    cos, sin = s0["cos"].index_select(0, T3), s0["sin"].index_select(0, T3)
    out["qkv_post"] = gtime(lambda: tpk.qkv_post(s0["qkv"], Ly["qn"], Ly["kn"], 1e-6, cos, sin, T3, s0["q"], s0["K"][0], s0["V"][0], nq, nkv, D))
    for T in (100, 2000, 9000):
        e.set_T(T)
        sl = (T3 + 1).to(torch.int32)
        for ns in (1, 0, 4, 16):
            out[f"fa3_T{T}_s{ns}"] = gtime(lambda: fa.flash_attn_with_kvcache(s0["q"].view(NR, 1, nq, D), s0["K"][0], s0["V"][0], cache_seqlens=sl,
                                                                          softmax_scale=D ** -0.5, causal=True, num_splits=ns))
    out["empty_kernel_like(fill_)"] = gtime(lambda: s0["x2"].fill_(0))
print(json.dumps(out), flush=True)
e._mode(False)

# all-reduce round trip across G GPUs: per rank a graph of R x (push gemv + ar_rms), tiny K and the o_proj shard
R = 100
for SIG in (0,):
    for KK in (128, nq * D):
        wt = [(torch.randn(H, KK, device=dd) * 0.02).to(BF) for dd in e.tpd]
        xt = [torch.randn(NR, KK, device=dd).to(BF) for dd in e.tpd]
        e._mode(True)
        gs = []
        for r, dd in enumerate(e.tpd):
            def fn(r=r):
                for k in range(R):
                    site = 1 + (k % (e.nsite - 1))
                    tpk.gemv(xt[r], wt[r], e.ars[r].part[:NR], f32=True)
                    e.ars[r].reduce(site, e.shards[r]["res"], e.shards[r]["L"][0]["ln2"], 1e-6, e.shards[r]["x2"], NR)
            with torch.cuda.device(dd):
                s_ = torch.cuda.Stream(dd)
                s_.wait_stream(torch.cuda.current_stream(dd))
                with torch.cuda.stream(s_):
                    fn()
                torch.cuda.synchronize(dd)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=s_):
                    fn()
                torch.cuda.synchronize(dd)
            gs.append(g)
        # local-only cost of the same graph (warm tables, flags ready): push + ar_rms with no cross-GPU wait
        t_loc = None
        with torch.cuda.device(e.tpd[0]):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            gs[0].replay(); torch.cuda.synchronize(e.tpd[0])
            e0.record(); gs[0].replay(); e1.record(); torch.cuda.synchronize(e.tpd[0])
            t_loc = e0.elapsed_time(e1) / R * 1000
        e._mode(False)
        ts = []
        for it in range(6):
            e.set_T(100 + it)
            t1 = time.perf_counter()
            for r, dd in enumerate(e.tpd):
                with torch.cuda.device(dd):
                    gs[r].replay()
            for dd in e.tpd:
                torch.cuda.synchronize(dd)
            ts.append((time.perf_counter() - t1) * 1e3)
        e.check_errors()
        print(json.dumps({"G": G, "SIG": SIG, "K": KK, "allreduce_us_per_site": round(min(ts[1:]) / R * 1000, 2), "local_only_us": round(t_loc, 2)}), flush=True)
        del gs
