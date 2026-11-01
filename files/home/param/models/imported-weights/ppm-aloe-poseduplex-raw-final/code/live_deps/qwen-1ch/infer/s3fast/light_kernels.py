"""Triton kernels for the fast light decoder (tiny batch: 2B rows, <= 64 positions per frame).

  gemv(x, w)          y = x @ w^T for M <= 8 rows (bf16 in / out, fp32 accumulation): one program per BN output columns reads its whole
                      [BN, K] weight slab once (the light's K is 1024..3072), all M rows reuse it -> weight-bandwidth bound, no split-K.
  light_attn(...)     per (stream, kv head): q / k per-head RMSNorm + RoPE (FO._qk_fwd_k's arithmetic and roundings), the new k / v
                      written into the static KV cache at positions t0 .. t0+L-1, then causal attention of the GQA group's q heads
                      over positions 0 .. t0+l (fp32 softmax) -> [N, nh * D]. Replaces FO.qk_prep + FA3 kvcache (+ its scheduler kernel).
  silu_mul_gu(gu)     gu [N, 2I] (gate | up of one concatenated GEMM) -> silu(gate) * up [N, I] (FO._silu_mul_fwd_k's roundings)."""
import torch
import triton
import triton.language as tl


@triton.autotune(configs=[triton.Config({"BN": bn, "BK": bk}, num_warps=w, num_stages=st)
                          for bn, bk, w, st in ((16, 128, 4, 3), (16, 256, 4, 3), (32, 128, 4, 3), (32, 256, 4, 3), (64, 128, 4, 3),
                                                (16, 512, 4, 2), (32, 64, 4, 4), (16, 128, 2, 4), (64, 64, 4, 4))],
                 key=["N", "K", "BM"])
@triton.jit
def _gemv_k(X, W, Y, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """y[:M] = x[:M] @ w^T with M padded to BM = 16 rows for tl.dot (weight-bandwidth bound)."""
    pid = tl.program_id(0)
    n = pid * BN + tl.arange(0, BN)
    mi = tl.arange(0, BM)
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + tl.arange(0, BK)
        x = tl.load(X + mi[:, None] * K + kk[None, :], mask=mi[:, None] < M, other=0.0)
        w = tl.load(W + n[:, None].to(tl.int64) * K + kk[None, :])
        acc = tl.dot(x, tl.trans(w), acc)
    tl.store(Y + mi[:, None] * N + n[None, :], acc.to(tl.bfloat16), mask=mi[:, None] < M)


@triton.autotune(configs=[triton.Config({"BN": bn, "BK": bk}, num_warps=w)
                          for bn in (4, 8, 16) for bk in (256, 512, 1024) for w in (2, 4, 8)],
                 key=["N", "K", "BM"])
@triton.jit
def _gemv_small_k(X, W, Y, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """y[:BM] = x[:BM] @ w^T for BM in {1, 2, 3, 4} rows (explicit accumulators; CUDA-core FMA, weight-bandwidth bound)."""
    pid = tl.program_id(0)
    n = pid * BN + tl.arange(0, BN)
    acc0 = tl.zeros([BN], dtype=tl.float32)
    acc1 = tl.zeros([BN], dtype=tl.float32)
    acc2 = tl.zeros([BN], dtype=tl.float32)
    acc3 = tl.zeros([BN], dtype=tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + tl.arange(0, BK)
        w = tl.load(W + n[:, None].to(tl.int64) * K + kk[None, :]).to(tl.float32)
        acc0 += tl.sum(w * tl.load(X + kk).to(tl.float32)[None, :], axis=1)
        if BM > 1:
            acc1 += tl.sum(w * tl.load(X + K + kk).to(tl.float32)[None, :], axis=1)
        if BM > 2:
            acc2 += tl.sum(w * tl.load(X + 2 * K + kk).to(tl.float32)[None, :], axis=1)
        if BM > 3:
            acc3 += tl.sum(w * tl.load(X + 3 * K + kk).to(tl.float32)[None, :], axis=1)
    tl.store(Y + n, acc0.to(tl.bfloat16))
    if BM > 1:
        tl.store(Y + N + n, acc1.to(tl.bfloat16))
    if BM > 2:
        tl.store(Y + 2 * N + n, acc2.to(tl.bfloat16))
    if BM > 3:
        tl.store(Y + 3 * N + n, acc3.to(tl.bfloat16))


def gemv(x, w, out=None):
    """x [..., K] (M = prod(leading) <= 16 rows) @ w[N, K]^T -> [..., N] bf16 (N % 64 == 0, K % 1024 == 0 or K in {..})."""
    shp = x.shape
    K = shp[-1]
    x2 = x.reshape(-1, K)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    M = x2.shape[0]
    N = w.shape[0]
    assert M <= 16 and w.shape[1] == K and w.is_contiguous() and N % 64 == 0 and K % 1024 == 0, (x.shape, w.shape)
    y = out if out is not None else torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
    if M <= 4:
        _gemv_small_k[lambda meta: (triton.cdiv(N, meta["BN"]),)](x2, w, y, N, K, BM=M)
    else:
        _gemv_k[lambda meta: (triton.cdiv(N, meta["BN"]),)](x2, w, y, M, N, K, BM=16)
    return y.view(*shp[:-1], N)


@triton.jit
def _light_attn_k(QKV, QN, KN, COS, SIN, KC, VC, O, eps, scale, t0, MP,
                  L: tl.constexpr, NH: tl.constexpr, KVH: tl.constexpr, D: tl.constexpr, HALF: tl.constexpr, BLK: tl.constexpr):
    s = tl.program_id(0)
    h = tl.program_id(1)
    gq = tl.program_id(2)                                   # this program's q head within the GQA group (cache rows written by each)
    G: tl.constexpr = NH // KVH
    W: tl.constexpr = (NH + 2 * KVH) * D
    o1 = tl.arange(0, HALF)
    kn1 = tl.load(KN + o1).to(tl.float32)
    kn2 = tl.load(KN + HALF + o1).to(tl.float32)
    qn1 = tl.load(QN + o1).to(tl.float32)
    qn2 = tl.load(QN + HALF + o1).to(tl.float32)
    s64 = s.to(tl.int64)
    # 1) the new keys / values -> cache
    for l in tl.static_range(L):
        row = s64 * L + l
        pos = t0 + l
        base = QKV + row * W
        x1 = tl.load(base + NH * D + h * D + o1).to(tl.float32)
        x2 = tl.load(base + NH * D + h * D + HALF + o1).to(tl.float32)
        rstd = 1.0 / tl.sqrt((tl.sum(x1 * x1, axis=0) + tl.sum(x2 * x2, axis=0)) / D + eps)
        k1 = (kn1 * (x1 * rstd).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        k2 = (kn2 * (x2 * rstd).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        c1 = tl.load(COS + row * D + o1).to(tl.float32)
        c2 = tl.load(COS + row * D + HALF + o1).to(tl.float32)
        s1 = tl.load(SIN + row * D + o1).to(tl.float32)
        s2 = tl.load(SIN + row * D + HALF + o1).to(tl.float32)
        cb = ((s64 * MP + pos) * KVH + h) * D
        tl.store(KC + cb + o1, (k1 * c1 - k2 * s1).to(tl.bfloat16))
        tl.store(KC + cb + HALF + o1, (k2 * c2 + k1 * s2).to(tl.bfloat16))
        v1 = tl.load(base + (NH + KVH) * D + h * D + o1)
        v2 = tl.load(base + (NH + KVH) * D + h * D + HALF + o1)
        tl.store(VC + cb + o1, v1)
        tl.store(VC + cb + HALF + o1, v2)
    tl.debug_barrier()
    # 2) attention of the group's q heads over positions 0 .. t0 + l
    j = tl.arange(0, BLK)
    kb = ((s64 * MP + j) * KVH + h) * D
    for l in tl.static_range(L):
        row = s64 * L + l
        pos = t0 + l
        jm = j <= pos
        K1 = tl.load(KC + kb[:, None] + o1[None, :], mask=jm[:, None], other=0.0).to(tl.float32)
        K2 = tl.load(KC + kb[:, None] + HALF + o1[None, :], mask=jm[:, None], other=0.0).to(tl.float32)
        V1 = tl.load(VC + kb[:, None] + o1[None, :], mask=jm[:, None], other=0.0).to(tl.float32)
        V2 = tl.load(VC + kb[:, None] + HALF + o1[None, :], mask=jm[:, None], other=0.0).to(tl.float32)
        c1 = tl.load(COS + row * D + o1).to(tl.float32)
        c2 = tl.load(COS + row * D + HALF + o1).to(tl.float32)
        s1 = tl.load(SIN + row * D + o1).to(tl.float32)
        s2 = tl.load(SIN + row * D + HALF + o1).to(tl.float32)
        for g in tl.static_range(1):
            qh = h * G + gq
            base = QKV + row * W + qh * D
            x1 = tl.load(base + o1).to(tl.float32)
            x2 = tl.load(base + HALF + o1).to(tl.float32)
            rstd = 1.0 / tl.sqrt((tl.sum(x1 * x1, axis=0) + tl.sum(x2 * x2, axis=0)) / D + eps)
            q1 = (qn1 * (x1 * rstd).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
            q2 = (qn2 * (x2 * rstd).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
            r1 = (q1 * c1 - q2 * s1).to(tl.bfloat16).to(tl.float32)
            r2 = (q2 * c2 + q1 * s2).to(tl.bfloat16).to(tl.float32)
            sc = (tl.sum(K1 * r1[None, :], axis=1) + tl.sum(K2 * r2[None, :], axis=1)) * scale
            sc = tl.where(jm, sc, float("-inf"))
            p = tl.exp(sc - tl.max(sc, axis=0))
            p = p / tl.sum(p, axis=0)
            ob = O + row * (NH * D) + qh * D
            tl.store(ob + o1, tl.sum(V1 * p[:, None], axis=0).to(tl.bfloat16))
            tl.store(ob + HALF + o1, tl.sum(V2 * p[:, None], axis=0).to(tl.bfloat16))


def light_attn(qkv, qn, kn, eps, cos, sin, KC, VC, t0, L, nh, kvh, D, scale, out=None):
    """qkv [N = S * L, (nh + 2 kvh) * D] bf16 contiguous; cos / sin [N, D]; KC / VC [S, MP, kvh, D] -> [N, nh * D]."""
    N = qkv.shape[0]
    S = N // L
    MP = KC.shape[1]
    o = out if out is not None else torch.empty(N, nh * D, dtype=torch.bfloat16, device=qkv.device)
    _light_attn_k[(S, kvh, nh // kvh)](qkv, qn, kn, cos, sin, KC, VC, o, float(eps), float(scale), int(t0), MP, L=L, NH=nh, KVH=kvh, D=D, HALF=D // 2,
                            BLK=triton.next_power_of_2(MP), num_warps=4)
    return o


@triton.jit
def _silu_mul_gu_k(GU, Y, I, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = offs < I
    g = tl.load(GU + row * 2 * I + offs, mask=m, other=0.0).to(tl.float32)
    u = tl.load(GU + row * 2 * I + I + offs, mask=m, other=0.0).to(tl.float32)
    a = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + row * I + offs, (a * u).to(tl.bfloat16), mask=m)


def silu_mul_gu(gu, out=None):
    N, I2 = gu.shape
    I = I2 // 2
    y = out if out is not None else torch.empty(N, I, dtype=torch.bfloat16, device=gu.device)
    _silu_mul_gu_k[(N, triton.cdiv(I, 1024))](gu, y, I, BLOCK=1024, num_warps=4)
    return y
