"""GEMV variants for M = 3 rows (bf16 weights streamed once): tl.dot (rows padded to 16) vs elementwise-accumulate FMA vs cuBLAS; and the
row-parallel push signalling variants (per-program sys-scope atomic vs fence + gpu-scope atomic vs separate broadcast kernel)."""
import json, sys
import torch, triton, triton.language as tl
import torch.nn.functional as F
BF = torch.bfloat16
d = torch.device("cuda:0")


@triton.jit
def k_dot(X, W, Y, M, N, K, BN: tl.constexpr, BK: tl.constexpr, F32: tl.constexpr):
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    mi = tl.arange(0, 16)
    acc = tl.zeros([16, BN], dtype=tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + tl.arange(0, BK)
        x = tl.load(X + mi[:, None] * K + kk[None, :], mask=mi[:, None] < M, other=0.0)
        w = tl.load(W + n[:, None].to(tl.int64) * K + kk[None, :])
        acc = tl.dot(x, tl.trans(w), acc)
    if F32:
        tl.store(Y + mi[:, None] * N + n[None, :], acc, mask=mi[:, None] < M)
    else:
        tl.store(Y + mi[:, None] * N + n[None, :], acc.to(tl.bfloat16), mask=mi[:, None] < M)


@triton.jit
def k_fma(X, W, Y, N, K, BN: tl.constexpr, BK: tl.constexpr):
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    a0 = tl.zeros([BN, BK], dtype=tl.float32)
    a1 = tl.zeros([BN, BK], dtype=tl.float32)
    a2 = tl.zeros([BN, BK], dtype=tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + tl.arange(0, BK)
        w = tl.load(W + n[:, None].to(tl.int64) * K + kk[None, :]).to(tl.float32)
        a0 += w * tl.load(X + kk).to(tl.float32)[None, :]
        a1 += w * tl.load(X + K + kk).to(tl.float32)[None, :]
        a2 += w * tl.load(X + 2 * K + kk).to(tl.float32)[None, :]
    tl.store(Y + n, tl.sum(a0, 1).to(tl.bfloat16))
    tl.store(Y + N + n, tl.sum(a1, 1).to(tl.bfloat16))
    tl.store(Y + 2 * N + n, tl.sum(a2, 1).to(tl.bfloat16))


def gtime(fn, rep=200):
    s = torch.cuda.Stream(d)
    s.wait_stream(torch.cuda.current_stream(d))
    with torch.cuda.stream(s):
        fn(); fn()
    torch.cuda.synchronize(d)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        for _ in range(rep):
            fn()
    g.replay(); torch.cuda.synchronize(d)
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record(); g.replay(); e1.record(); torch.cuda.synchronize(d)
    return e0.elapsed_time(e1) / rep * 1000


res = {}
for (N, K) in ((2560, 5120), (12800, 5120), (5120, 2048), (5120, 6400), (1280, 5120), (6400, 5120), (5120, 1024), (5120, 3200)):
    # several weight copies so the timed loop streams from HBM, not L2
    nw = max(2, int(200e6 // (N * K * 2)))
    Ws = [(torch.randn(N, K, device=d) * 0.02).to(BF) for _ in range(nw)]
    x = torch.randn(3, K, device=d).to(BF)
    y = torch.empty(3, N, dtype=BF, device=d)
    yf = torch.empty(3, N, dtype=torch.float32, device=d)
    it = [0]

    def nxt():
        it[0] = (it[0] + 1) % nw
        return Ws[it[0]]
    r = {"MB": round(N * K * 2 / 1e6, 1)}
    r["cublas"] = gtime(lambda: F.linear(x, nxt()))
    best = (1e9, None)
    for BN in (16, 32, 64, 128):
        for BK in (64, 128, 256):
            if K % BK or N % BN:
                continue
            for w_ in (4, 8):
                for st in (3, 4, 5):
                    try:
                        t = gtime(lambda: k_dot[(N // BN,)](x, nxt(), y, 3, N, K, BN=BN, BK=BK, F32=False, num_warps=w_, num_stages=st))
                    except Exception as ex:  # noqa
                        continue
                    if t < best[0]:
                        best = (t, (BN, BK, w_, st))
    r["dot_best"] = best
    bf = (1e9, None)
    for BN in (4, 8, 16):
        for BK in (128, 256, 512):
            if K % BK or N % BN or BN * BK > 4096:
                continue
            for w_ in (2, 4, 8):
                t = gtime(lambda: k_fma[(N // BN,)](x, nxt(), y, N, K, BN=BN, BK=BK, num_warps=w_))
                if t < bf[0]:
                    bf = (t, (BN, BK, w_))
    r["fma_best"] = bf
    BN, BK, w_, st = best[1]
    r["dot_f32out"] = gtime(lambda: k_dot[(N // BN,)](x, nxt(), yf, 3, N, K, BN=BN, BK=BK, F32=True, num_warps=w_, num_stages=st))
    r["GBps_best"] = round(N * K * 2 / (min(best[0], bf[0], r["cublas"]) * 1e-6) / 1e9)
    res[f"{N}x{K}"] = r
    print(f"{N}x{K}", r, flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
