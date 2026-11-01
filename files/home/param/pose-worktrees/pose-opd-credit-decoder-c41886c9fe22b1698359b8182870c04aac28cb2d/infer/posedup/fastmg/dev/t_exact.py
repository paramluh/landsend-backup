#!/usr/bin/env python3
"""exactness probes for an all-gather (column-parallel only) tensor-parallel trunk: is a GEMM on a row slice of the weight bitwise equal to
the matching columns of the full GEMM (cuBLAS, M = 1..3 rows)? is FA3 kvcache attention on a head slice bitwise equal? P2P access?"""
import os
import sys
import json
AC = "/data/home/amartyavarma/ppm/aloe-m1vox-ft/code"
for p in (AC, f"{AC}/lib", f"{AC}/vendor"):
    sys.path.insert(0, p)
import torch
import torch.nn.functional as F
import fa3_op

BF = torch.bfloat16
dev = torch.device("cuda:0")
torch.manual_seed(0)
res = {"gemm": [], "fa3": [], "p2p": {}}
shapes = {"q": (8192, 5120), "k": (1024, 5120), "o": (5120, 8192), "gate": (25600, 5120), "down": (5120, 25600)}
for name, (N, K) in shapes.items():
    W = (torch.randn(N, K, device=dev) * 0.02).to(BF)
    for M in (1, 2, 3):
        x = torch.randn(1, M, K, device=dev).to(BF)
        full = F.linear(x, W)
        for G in (2, 4, 8):
            n = N // G
            ok = True
            for r in range(G):
                y = F.linear(x, W[r * n:(r + 1) * n])
                ok &= bool(torch.equal(y, full[..., r * n:(r + 1) * n]))
            res["gemm"].append({"w": name, "N": N, "K": K, "M": M, "G": G, "equal": ok})
# fused qkv shard vs separate
Wq, Wk, Wv = [(torch.randn(n, 5120, device=dev) * 0.02).to(BF) for n in (8192, 1024, 1024)]
for M in (1, 3):
    x = torch.randn(1, M, 5120, device=dev).to(BF)
    q, k, v = F.linear(x, Wq), F.linear(x, Wk), F.linear(x, Wv)
    for G in (2, 4, 8):
        ok = True
        for r in range(G):
            a, b = 8192 // G, 1024 // G
            Wr = torch.cat([Wq[r * a:(r + 1) * a], Wk[r * b:(r + 1) * b], Wv[r * b:(r + 1) * b]], 0).contiguous()
            y = F.linear(x, Wr)
            ok &= bool(torch.equal(y[..., :a], q[..., r * a:(r + 1) * a]) and torch.equal(y[..., a:a + b], k[..., r * b:(r + 1) * b])
                       and torch.equal(y[..., a + b:], v[..., r * b:(r + 1) * b]))
        res["gemm"].append({"w": "qkv_fused_shard", "M": M, "G": G, "equal": ok})
    # fused gate|up shard
Wg, Wu = [(torch.randn(25600, 5120, device=dev) * 0.02).to(BF) for _ in range(2)]
for M in (1, 3):
    x = torch.randn(1, M, 5120, device=dev).to(BF)
    g, u = F.linear(x, Wg), F.linear(x, Wu)
    for G in (2, 4, 8):
        ok = True
        n = 25600 // G
        for r in range(G):
            y = F.linear(x, torch.cat([Wg[r * n:(r + 1) * n], Wu[r * n:(r + 1) * n]], 0).contiguous())
            ok &= bool(torch.equal(y[..., :n], g[..., r * n:(r + 1) * n]) and torch.equal(y[..., n:], u[..., r * n:(r + 1) * n]))
        res["gemm"].append({"w": "gu_fused_shard", "M": M, "G": G, "equal": ok})
# FA3 head slices
fa = fa3_op._mod().flash_attn_interface
for R in (1, 3):
    for L in (100, 2000, 9000):
        kc = (torch.randn(R, 9600, 8, 128, device=dev)).to(BF)
        vc = (torch.randn(R, 9600, 8, 128, device=dev)).to(BF)
        q = torch.randn(R, 1, 64, 128, device=dev).to(BF)
        k = torch.randn(R, 1, 8, 128, device=dev).to(BF)
        v = torch.randn(R, 1, 8, 128, device=dev).to(BF)
        sl = torch.full((R,), L, dtype=torch.int32, device=dev)
        kc1, vc1 = kc.clone(), vc.clone()
        full = fa.flash_attn_with_kvcache(q, kc1, vc1, k=k, v=v, cache_seqlens=sl, softmax_scale=128 ** -0.5, causal=True, num_splits=1)
        for G in (2, 4, 8):
            ok = True
            a, b = 64 // G, 8 // G
            for r in range(G):
                kcr = kc[:, :, r * b:(r + 1) * b].contiguous(); vcr = vc[:, :, r * b:(r + 1) * b].contiguous()
                y = fa.flash_attn_with_kvcache(q[:, :, r * a:(r + 1) * a].contiguous(), kcr, vcr, k=k[:, :, r * b:(r + 1) * b].contiguous(),
                                               v=v[:, :, r * b:(r + 1) * b].contiguous(), cache_seqlens=sl, softmax_scale=128 ** -0.5,
                                               causal=True, num_splits=1)
                ok &= bool(torch.equal(y, full[:, :, r * a:(r + 1) * a])) and bool(torch.equal(kcr, kc1[:, :, r * b:(r + 1) * b]))
            res["fa3"].append({"R": R, "L": L, "G": G, "equal": ok})
n = torch.cuda.device_count()
res["p2p"] = {f"{i}-{j}": torch.cuda.can_device_access_peer(i, j) for i in range(n) for j in range(n) if i != j}
print(json.dumps(res, indent=0))
json.dump(res, open(sys.argv[1], "w"), indent=1)
