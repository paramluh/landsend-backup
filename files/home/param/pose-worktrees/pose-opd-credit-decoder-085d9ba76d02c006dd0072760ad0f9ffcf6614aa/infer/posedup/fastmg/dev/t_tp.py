#!/usr/bin/env python3
"""fastmg unit + latency test with RANDOM weights (no checkpoint): tpk kernels vs references, then the tensor-parallel trunk of
live_ppm_pose_mg (its own _init_trunk / _trunk / _prefill / _mode code on a bare engine object) vs an unsharded reference (4 layers),
then the trunk graph latency (64 layers, every visible GPU) at short and long context. usage: t_tp.py <out json> [nlayers_ref]"""
import json, sys, time, types
sys.path.insert(0, "/data/home/amartyavarma/ppm/aloe-posedup-live/fastmg")
import torch
import torch.nn.functional as F
import live_ppm_pose_mg as MG
import tpk
FO, M, LP, LK = MG.FO, MG.M, MG.LP, MG.LK
BF = torch.bfloat16
OUT = sys.argv[1]
res = {}
devs = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
d0 = devs[0]
torch.manual_seed(0)
c32 = M.cfg32(64)
H, D, NQ, NKV, I = c32.hidden_size, c32.head_dim, c32.num_attention_heads, c32.num_key_value_heads, c32.intermediate_size
fa = MG.fa3_op._mod().flash_attn_interface

# ---- 1. gemv vs F.linear (fp32-accumulated: compare against an fp32 reference)
g = []
for (N, K, Mr, silu) in ((2560, 5120, 3, False), (12800, 5120, 3, False), (5120, 2048, 3, False), (5120, 6400, 3, True), (1280, 5120, 2, False)):
    w = (torch.randn(N, K, device=d0) * 0.02).to(BF)
    x = torch.randn(Mr, 2 * K if silu else K, device=d0).to(BF)
    y = tpk.gemv(x, w, silu=silu)
    xin = LK.silu_mul_gu(x) if silu else x
    ref = (xin.float() @ w.float().t())
    cub = F.linear(xin, w)
    g.append({"N": N, "K": K, "M": Mr, "silu": silu, "max_err_vs_fp32": float((y.float() - ref).abs().max()),
              "cublas_max_err_vs_fp32": float((cub.float() - ref).abs().max()), "frac_bitwise_eq_cublas": float((y == cub).float().mean())})
res["gemv"] = g
print("gemv", g, flush=True)

# ---- 2. qkv_post vs fused_triton qk_prep + cache writes (bitwise), 3. FA3 k=None vs append
R, nq, nkv, MP = 3, 16, 2, 4096
qkv = torch.randn(R, (nq + 2 * nkv) * D, device=d0).to(BF)
qn, kn = (torch.rand(D, device=d0) + 0.5).to(BF), (torch.rand(D, device=d0) + 0.5).to(BF)
rot = M.Rot(c32, d0)
cs, sn = rot(torch.zeros(1, MP, 1, dtype=BF, device=d0), torch.arange(MP, device=d0)[None])
cs, sn = cs[0].contiguous(), sn[0].contiguous()
T3 = torch.tensor([1000, 1000, 1000], device=d0)
KC, VC = torch.randn(R, MP, nkv, D, device=d0).to(BF), torch.randn(R, MP, nkv, D, device=d0).to(BF)
KC2, VC2 = KC.clone(), VC.clone()
qb = torch.zeros(R, nq, D, dtype=BF, device=d0)
tpk.qkv_post(qkv, qn, kn, 1e-6, cs.index_select(0, T3), sn.index_select(0, T3), T3, qb, KC, VC, nq, nkv, D)
q_ref, k_ref = FO.qk_prep(qkv[None, :, :nq * D].contiguous(), qkv[None, :, nq * D:(nq + nkv) * D].contiguous(), qn, kn, 1e-6,
                          cs.index_select(0, T3)[None], sn.index_select(0, T3)[None])
v_ref = qkv[:, (nq + nkv) * D:].view(R, nkv, D)
res["qkv_post_bitwise"] = {"q": bool(torch.equal(qb, q_ref[0])), "k": bool(torch.equal(KC[:, 1000], k_ref[0])), "v": bool(torch.equal(VC[:, 1000], v_ref))}
sl = torch.full((R,), 1000, dtype=torch.int32, device=d0)
a_app = fa.flash_attn_with_kvcache(q_ref[0].view(R, 1, nq, D), KC2, VC2, k=k_ref[0].view(R, 1, nkv, D), v=v_ref.reshape(R, 1, nkv, D).contiguous(),
                                   cache_seqlens=sl, softmax_scale=D ** -0.5, causal=True, num_splits=1)
a_nok = fa.flash_attn_with_kvcache(qb.view(R, 1, nq, D), KC, VC, cache_seqlens=sl + 1, softmax_scale=D ** -0.5, causal=True, num_splits=1)
a_s0 = fa.flash_attn_with_kvcache(qb.view(R, 1, nq, D), KC, VC, cache_seqlens=sl + 1, softmax_scale=D ** -0.5, causal=True, num_splits=0)
res["fa3_noappend_bitwise_vs_append"] = bool(torch.equal(a_app, a_nok))
res["fa3_splits0_maxdiff"] = float((a_s0.float() - a_nok.float()).abs().max())
print("post/fa3", res["qkv_post_bitwise"], res["fa3_noappend_bitwise_vs_append"], res["fa3_splits0_maxdiff"], flush=True)


# ---- 4. the TP trunk on a bare engine object
def bare(G, NR, nl, lmax, full=None, splits=1):
    e = object.__new__(MG.LivePPM)
    e.devs = devs[:G]; e.tpd = e.devs; e.G = G; e.NR = NR; e.c32 = c32; e.lmax = lmax; e.pcfg = NR == 3
    e.dS, e.dP, e.dF = MG.roles(e.devs) if G >= 2 else (e.devs[0],) * 3
    e.attn_splits = splits; e.fa = fa
    sh = []
    for r, d in enumerate(e.devs):
        gq, gk, gi = NQ // G, NKV // G, I // G
        Ls = []
        for li in range(nl):
            if full is not None:
                W = full[li]
                sq, sk, si = slice(r * gq * D, (r + 1) * gq * D), slice(r * gk * D, (r + 1) * gk * D), slice(r * gi, (r + 1) * gi)
                Ly = dict(ln1=W["ln1"].to(d), wqkv=torch.cat([W["q"][sq], W["k"][sk], W["v"][sk]]).contiguous().to(d), qn=W["qn"].to(d), kn=W["kn"].to(d),
                          wo=W["o"][:, sq].contiguous().to(d), ln2=W["ln2"].to(d), wgu=torch.cat([W["g"][si], W["u"][si]]).contiguous().to(d),
                          wd=W["d"][:, si].contiguous().to(d))
            else:
                gen = torch.Generator(device=d).manual_seed(100 * r + li)
                rn = lambda *s: (torch.randn(*s, generator=gen, device=d) * 0.02).to(BF)
                Ly = dict(ln1=torch.ones(H, dtype=BF, device=d), wqkv=rn((gq + 2 * gk) * D, H), qn=torch.ones(D, dtype=BF, device=d),
                          kn=torch.ones(D, dtype=BF, device=d), wo=rn(H, gq * D), ln2=torch.ones(H, dtype=BF, device=d), wgu=rn(2 * gi, H), wd=rn(H, gi))
            Ly.update(e1=1e-6, e2=1e-6, eqk=1e-6, sc=D ** -0.5)
            Ls.append(Ly)
        sh.append(dict(dev=d, nq=gq, nkv=gk, I=gi, L=Ls, nw=torch.ones(H, dtype=BF, device=d), ne=1e-6, rot=M.Rot(c32, d)))
    e.shards = sh
    e._init_trunk()
    e.h0 = torch.zeros(NR, H, dtype=BF, device=e.dS)
    e.hn_prev = torch.zeros(2, H, dtype=BF, device=e.dS)
    e.hn_u = torch.zeros(1, H, dtype=BF, device=e.dS)
    e.hn_p = torch.zeros(2 if NR == 3 else 1, H, dtype=BF, device=e.dP)
    e.ho = {}
    e.scratch = {d: torch.zeros(1 << 16, dtype=torch.int64, device=d) for d in e.devs}
    e._trunk_in = lambda: e.ars[0].part[:NR].copy_(e.h0)
    return e


def capture(e):
    e._mode(True)
    e.set_T(64)
    gs = []
    for r, d in enumerate(e.tpd):
        with torch.cuda.device(d):
            s = torch.cuda.Stream(d)
            s.wait_stream(torch.cuda.current_stream(d))
            with torch.cuda.stream(s):
                for _ in range(2):
                    e._trunk(r)
            torch.cuda.current_stream(d).wait_stream(s)
            torch.cuda.synchronize(d)
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr, stream=s):
                e._trunk(r)
            torch.cuda.synchronize(d)
        gs.append(gr)
    e._mode(False)
    return gs


def replay(e, gs, T):
    e.set_T(T)
    for r, d in enumerate(e.tpd):
        with torch.cuda.device(d):
            gs[r].replay()
    torch.cuda.synchronize(e.dS)


# 4a. correctness vs unsharded reference (nl layers, prefill P positions + 3 decode steps)
nl = int(sys.argv[2]) if len(sys.argv) > 2 else 4
dref = devs[-1]
full = []
for li in range(nl):
    gen = torch.Generator(device=dref).manual_seed(7 + li)
    rn = lambda *s: (torch.randn(*s, generator=gen, device=dref) * 0.02).to(BF)
    full.append(dict(ln1=(torch.rand(H, generator=gen, device=dref) + 0.5).to(BF), q=rn(NQ * D, H), k=rn(NKV * D, H), v=rn(NKV * D, H),
                     qn=(torch.rand(D, generator=gen, device=dref) + 0.5).to(BF), kn=(torch.rand(D, generator=gen, device=dref) + 0.5).to(BF),
                     o=rn(H, NQ * D), ln2=(torch.rand(H, generator=gen, device=dref) + 0.5).to(BF), g=rn(I, H), u=rn(I, H), d=rn(H, I)))
P, NR, lmax = 300, 3, 1024


def ref_run(e_in, steps, split=1):
    """unsharded: prefill e_in[:P] then decode e_in[P:P+steps] one position at a time (3 rows: same input) -> final normed hidden of each step."""
    rotr = M.Rot(c32, dref)
    cs, sn = rotr(torch.zeros(1, lmax, 1, dtype=BF, device=dref), torch.arange(lmax, device=dref)[None])
    Kc = [torch.zeros(lmax, NKV, D, dtype=BF, device=dref) for _ in range(nl)]
    Vc = [torch.zeros(lmax, NKV, D, dtype=BF, device=dref) for _ in range(nl)]
    outs = []
    for s0, n in [(0, P)] + [(P + j, 1) for j in range(steps)]:
        h = e_in[s0:s0 + n][None].to(dref)
        for li, W in enumerate(full):
            x = FO.rms(h, W["ln1"], 1e-6)
            q, k = FO.qk_prep(F.linear(x, W["q"]), F.linear(x, W["k"]), W["qn"], W["kn"], 1e-6, cs[:, s0:s0 + n], sn[:, s0:s0 + n])
            v = F.linear(x, W["v"]).view(n, NKV, D)
            Kc[li][s0:s0 + n] = k[0]; Vc[li][s0:s0 + n] = v
            cu_q = torch.tensor([0, n], dtype=torch.int32, device=dref); cu_k = torch.tensor([0, s0 + n], dtype=torch.int32, device=dref)
            a = MG.fa3_op.fa3_varlen_func(q[0], Kc[li][:s0 + n], Vc[li][:s0 + n], cu_q, cu_k, n, s0 + n, D ** -0.5, True)
            def lin(x, w):
                if split == 1:
                    return F.linear(x, w)
                K = w.shape[1] // split
                return sum(x[..., j * K:(j + 1) * K].float() @ w[:, j * K:(j + 1) * K].float().t() for j in range(split)).to(BF)
            h, x2 = FO.add_rms(h, lin(a.reshape(1, n, -1), W["o"]), W["ln2"], 1e-6)
            h = h + lin(FO.silu_mul(F.linear(x2, W["g"]), F.linear(x2, W["u"])), W["d"])
        if n == 1:
            outs.append(FO.rms(h[0], torch.ones(H, dtype=BF, device=dref), 1e-6))
    return outs


steps = 3
e_in = (torch.randn(P + steps, H, device=dref) * 1.0).to(BF)
with torch.cuda.device(dref):
    ref = ref_run(e_in, steps)
    floor = {}
    for sp in (2, 4):
        rs = ref_run(e_in, steps, split=sp)
        floor[f"split{sp}"] = [{"rel_l2": float((a[0].float() - b[0].float()).norm() / a[0].float().norm()), "max_abs": float((a[0].float() - b[0].float()).abs().max()),
                                "frac_bitwise": float((a[0] == b[0]).float().mean())} for a, b in zip(ref, rs)]
res["noise_floor_ref_vs_ref_with_rank_order_fp32_sums"] = floor
print("floor", floor, flush=True)
cmpres = []
for G in [g_ for g_ in (2, 4, 8) if g_ <= len(devs) - 1]:
    e = bare(G, NR, nl, lmax, full=full)
    gs = capture(e)
    hs = e._prefill(e_in[:P].to(e.dS), 0)
    for s_, d in zip(e.shards, e.tpd):
        for t in s_["K"] + s_["V"]:
            t[1, :P].copy_(t[0, :P]); t[2, :P].copy_(t[0, :P])
    errs = []
    for j in range(steps):
        e.h0.copy_(e_in[P + j][None].expand(NR, -1))
        replay(e, gs, P + j)
        out = e.hn_prev[0].float()
        r_ = ref[j][0].float().to(e.dS)
        errs.append({"max_abs": float((out - r_).abs().max()), "rel_l2": float((out - r_).norm() / r_.norm()),
                     "frac_bitwise": float((e.hn_prev[0] == ref[j][0].to(e.dS)).float().mean()),
                     "rows_equal": bool(torch.equal(e.hn_prev[0], e.hn_prev[1]) and torch.equal(e.hn_prev[0], e.hn_u[0]))})
    e.check_errors()
    # bf16 noise floor: the reference itself recomputed with cuBLAS vs the same in fp32 would differ by ~this much; compare to rounding unit
    cmpres.append({"G": G, "steps": errs})
    print("TP vs ref", G, errs, flush=True)
    del e, gs
    torch.cuda.empty_cache()
res["tp_vs_ref"] = cmpres

# 4b. latency of the 64-layer trunk graph
lat = []
for G in [g_ for g_ in (2, 4, 8) if g_ <= len(devs)]:
    for splits in (1, 0):
        e = bare(G, 3, 64, 9600, splits=splits)
        gs = capture(e)
        for T in (100, 2000, 9000):
            for _ in range(5):
                replay(e, gs, T)
            ts = []
            for _ in range(30):
                t1 = time.perf_counter(); replay(e, gs, T); ts.append((time.perf_counter() - t1) * 1e3)
            ts.sort()
            lat.append({"G": G, "splits": splits, "T": T, "ms_p50": round(ts[15], 3), "ms_min": round(ts[0], 3), "ms_max": round(ts[-1], 3)})
            print("lat", lat[-1], flush=True)
        e.check_errors()
        del e, gs
        torch.cuda.empty_cache()
res["trunk_latency"] = lat
json.dump(res, open(OUT, "w"), indent=1)
print("DONE")
