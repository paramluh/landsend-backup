"""fastmg kernels (10-25): single-process multi-GPU primitives for the live PPM engine. Every GPU of the engine has peer access to every
other (NVLink); cross-GPU dependencies are DEVICE-SIDE: a producer kernel writes its data straight into the consumer GPU's buffer (peer
stores) and then raises a flag there (release, system scope); the consumer kernel spins on its local flag (acquire, system scope). So
every per-frame step is a CUDA graph per GPU with no host-side cross-device event, and the trunk's tensor parallelism costs one
one-shot all-reduce (G peer stores + flag + local fixed-order sum) per row-parallel projection.

  gemv(x, w, y)                        y = x @ w^T, M <= 4 rows, bf16 in / out, fp32 accumulation (column-parallel q/k/v, gate|up).
  gemv_push(x, w, ar, site, silu)      fp32 partial of a ROW-parallel projection (o_proj / down_proj shard) stored into slot <rank> of every
                                       rank's all-reduce receive buffer, then the flags (last program of the grid; release, sys scope).
                                       silu=True: x is gate|up [M, 2K] and the input is silu(gate) * up computed on the fly (the
                                       light_kernels.silu_mul_gu / fused_triton silu_mul roundings), saving that kernel.
  ar_rms(ar, site, res, w, y)          wait for the G partials of `site`, sum them in rank order in fp32 (identical bits on every rank),
                                       round to bf16 = the projection output, then fused_triton._rms_fwd_k's residual add + RMSNorm
                                       arithmetic: res <- bf16(res + a), y <- bf16(w * bf16((res) * rstd)).
  push(src, dst_ptr, flag_ptr, ep)     small copy into another GPU's buffer + flag (cross-GPU hand-offs: lane token, pose codes, a_vec, codes).
  wait(flag, ep)                       spin until a local flag reaches the frame's value.
  qkv_post(...)                        trunk shard: q / k per-head RMSNorm + RoPE (fused_triton._qk_fwd_k arithmetic, same 16-row tiles)
                                       -> q buffer; k, v -> the shard's KV cache at each row's position T (FA3 then reads the cache with
                                       cache_seqlens = T + 1, no k / v append).
Flag values: hand-offs ep + 1, all-reduce sites ep * NSITE + site + 1 (ep = the host's frame serial, written into a per-GPU scalar each
frame; monotonic for the life of the engine, so flags never need a reset). All-reduce receive buffers rotate over 3 copies (site % 3):
two consecutive sites are the most that can be in flight. A spin gives up after ~SPIN_LIMIT polls and records the site in an error word
(read by the host each frame) instead of hanging the GPU."""
import ctypes

import torch
import triton
import triton.language as tl

BF = torch.bfloat16
SPIN_LIMIT = 1 << 26


def enable_peer_access(devs):
    """cudaDeviceEnablePeerAccess for every ordered pair (already-enabled is fine)."""
    rt = None
    for name in ("libcudart.so", "libcudart.so.12"):
        try:
            rt = ctypes.CDLL(name)
            break
        except OSError:
            continue
    if rt is None:                                            # torch's bundled runtime
        import glob
        import os
        for p in glob.glob(os.path.join(os.path.dirname(torch.__file__), "..", "nvidia", "cuda_runtime", "lib", "libcudart.so*")):
            rt = ctypes.CDLL(p)
            break
    assert rt is not None, "libcudart not found"
    for a in devs:
        for b in devs:
            if a == b:
                continue
            assert torch.cuda.can_device_access_peer(a.index, b.index), ("no P2P", a, b)
            with torch.cuda.device(a):
                torch.zeros(1, device=a)                      # context
                torch.zeros(1, device=b).to(a); torch.zeros(1, device=a).to(b)   # torch's own p2p enabling (both directions)
                assert rt.cudaSetDevice(ctypes.c_int(a.index)) == 0      # ctypes' runtime keeps its own current device
                err = rt.cudaDeviceEnablePeerAccess(ctypes.c_int(b.index), ctypes.c_uint(0))
                if err not in (0, 704):                       # 704 = cudaErrorPeerAccessAlreadyEnabled
                    raise RuntimeError(f"cudaDeviceEnablePeerAccess({a} -> {b}) = {err}")
                rt.cudaGetLastError()
    for d in devs:
        torch.cuda.synchronize(d)


# ================================================================================================ spin helpers (inlined)
@triton.jit
def _spin_ge(F, tgt, ERR, code, LIMIT: tl.constexpr):
    f = tl.atomic_add(F, 0, sem="acquire", scope="sys")
    n = 0
    while (f < tgt) & (n < LIMIT):
        f = tl.atomic_add(F, 0, sem="acquire", scope="sys")
        n += 1
    if f < tgt:
        tl.atomic_max(ERR, code.to(tl.int64))


# ================================================================================================ GEMV (M <= 16 rows, tensor cores)
def _gemv_cfgs():
    return [triton.Config({"BN": bn, "BK": bk}, num_warps=w, num_stages=st) for bn in (16, 32, 64, 128) for bk in (64, 128, 256)
            for w in (4, 8) for st in (3, 4, 5) if bn * bk <= 32768]


def _prune(configs, named_args, **kw):
    K, N = named_args["K"], named_args["N"]
    return [c for c in configs if K % c.kwargs["BK"] == 0 and N % c.kwargs["BN"] == 0]


@triton.jit
def _xtile(X, mi, kk, M, K, SILU: tl.constexpr):
    """[16, BK] bf16 input tile (rows >= M zero); SILU: X = gate | up [M, 2K] -> silu(gate) * up (silu_mul_gu's roundings)."""
    msk = mi[:, None] < M
    if SILU:
        g = tl.load(X + mi[:, None] * 2 * K + kk[None, :], mask=msk, other=0.0).to(tl.float32)
        u = tl.load(X + mi[:, None] * 2 * K + K + kk[None, :], mask=msk, other=0.0).to(tl.float32)
        a = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
        return (a * u).to(tl.bfloat16)
    else:
        return tl.load(X + mi[:, None] * K + kk[None, :], mask=msk, other=0.0)


@triton.jit
def _gemv_acc(X, W, n, M, K, BN: tl.constexpr, BK: tl.constexpr, SILU: tl.constexpr):
    mi = tl.arange(0, 16)
    acc = tl.zeros([16, BN], dtype=tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + tl.arange(0, BK)
        w = tl.load(W + n[:, None].to(tl.int64) * K + kk[None, :])
        acc = tl.dot(_xtile(X, mi, kk, M, K, SILU), tl.trans(w), acc)
    return acc, mi


@triton.autotune(configs=_gemv_cfgs(), key=["N", "K", "M", "SILU", "F32"], prune_configs_by={"early_config_prune": _prune})
@triton.jit
def _gemv_k(X, W, Y, M, N, K, BN: tl.constexpr, BK: tl.constexpr, SILU: tl.constexpr, F32: tl.constexpr):
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    acc, mi = _gemv_acc(X, W, n, M, K, BN, BK, SILU)
    if F32:
        tl.store(Y + mi[:, None] * N + n[None, :], acc, mask=mi[:, None] < M)
    else:
        tl.store(Y + mi[:, None] * N + n[None, :], acc.to(tl.bfloat16), mask=mi[:, None] < M)


def gemv(x, w, y=None, silu=False, f32=False):
    """x [M, K] (silu: [M, 2K] gate | up) bf16 contiguous, w [N, K] -> y [M, N] bf16 (f32: fp32, unrounded) - fp32 accumulation on
    tensor cores (rows padded to 16)."""
    M = x.shape[0]
    N, K = w.shape
    assert M <= 16 and x.is_contiguous() and w.is_contiguous() and x.shape[1] == (2 * K if silu else K) and K % 64 == 0 and N % 16 == 0
    y = y if y is not None else torch.empty(M, N, dtype=torch.float32 if f32 else BF, device=x.device)
    assert y.dtype == (torch.float32 if f32 else BF) and y.is_contiguous()
    _gemv_k[lambda m: (triton.cdiv(N, m["BN"]),)](x, w, y, M, N, K, SILU=silu, F32=f32)
    return y


class AllReduce:
    """one-shot all-reduce state of ONE rank: local partial part fp32 [BMAX, N]; receive buffer recv fp32 [3, G, BMAX, N] (rotating copies x
    source rank x rows); flags int64 [G, BMAX] (source rank x row); error word. AllReduce.link(list over ranks) wires the peer tables."""

    def __init__(self, rank, G, dev, N, BMAX, nsite):
        self.rank, self.G, self.dev, self.N, self.BMAX, self.nsite = rank, G, dev, N, BMAX, nsite
        self.part = torch.zeros(BMAX, N, dtype=torch.float32, device=dev)
        self.recv = torch.zeros(3, G, BMAX, N, dtype=torch.float32, device=dev)
        self.flags = torch.zeros(G, BMAX, dtype=torch.int64, device=dev)
        self.err = torch.zeros(1, dtype=torch.int64, device=dev)
        self.ep = torch.zeros(1, dtype=torch.int64, device=dev)

    @staticmethod
    def link(ars):
        G = len(ars)
        for r, a in enumerate(ars):
            es = a.recv.element_size()
            a.dst = torch.tensor([b.recv.data_ptr() + r * a.BMAX * a.N * es for b in ars], dtype=torch.int64, device=a.dev)
            a.flg = torch.tensor([b.flags.data_ptr() + r * a.BMAX * 8 for b in ars], dtype=torch.int64, device=a.dev)
            assert len(a.dst) == G

    def slot_off(self, site):
        return (site % 3) * self.G * self.BMAX * self.N

    def reduce(self, site, res, w, eps, y, rows, src0=False, has_res=True):
        """push this rank's partial rows (part) to every rank, wait for every rank's, sum in rank order (fp32), round once to bf16, then
        residual add + RMSNorm (fused_triton _rms_fwd_k arithmetic): res <- bf16(res + a) [has_res; else res <- a], y <- RMSNorm(res) x w.
        src0: only rank 0's partial exists (the trunk input broadcast)."""
        N = self.N
        _ar_k[(rows,)](self.part, self.dst, self.flg, self.recv, self.flags, self.ep, self.err, self.nsite, site, self.slot_off(site), res, w, y,
                       N, float(eps), self.rank, G=self.G, BMAX=self.BMAX, SRC0=src0, HAS_RES=has_res, BLOCK=triton.next_power_of_2(N),
                       LIMIT=SPIN_LIMIT, num_warps=8)


@triton.jit
def _fence_sys(x):
    return tl.inline_asm_elementwise("fence.acq_rel.sys; mov.u32 $0, $1;", "=r,r", [x], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit(do_not_specialize=["site", "me"])
def _ar_k(PART, DST, FLG, RECV, FLAGS, EP, ERR, nsite, site, slot_off, RES, Wn, Y, N, eps, me, G: tl.constexpr, BMAX: tl.constexpr,
          SRC0: tl.constexpr, HAS_RES: tl.constexpr, BLOCK: tl.constexpr, LIMIT: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    m = offs < N
    val = tl.load(EP) * nsite + site + 1
    if (not SRC0) or (me == 0):                               # 1. my partial row -> slot <me> of every rank, then its flag there
        x = tl.load(PART + row * N + offs, mask=m, other=0.0)
        for g in tl.static_range(G):
            d = tl.load(DST + g).to(tl.pointer_type(tl.float32)) + slot_off + row * N
            tl.store(d + offs, x, mask=m)
        tl.debug_barrier()
        _fence_sys(row)                                       # one system-scope fence, then relaxed flag stores (= a release; 1 fence, not G)
        for g in tl.static_range(G):
            f = tl.load(FLG + g).to(tl.pointer_type(tl.int64)) + row
            tl.atomic_xchg(f, val, sem="relaxed", scope="sys")
    if SRC0:                                                  # 2. wait for every source's row
        _spin_ge(FLAGS + row, val, ERR, site + 1, LIMIT)
    else:
        for g in tl.static_range(G):
            _spin_ge(FLAGS + g * BMAX + row, val, ERR, site + 1, LIMIT)
    base = RECV + slot_off + row * N                          # 3. fixed rank-order fp32 sum -> one bf16 rounding
    acc = tl.load(base + offs, mask=m, other=0.0)
    if not SRC0:
        for g in tl.static_range(1, G):
            acc = acc + tl.load(base + g * BMAX * N + offs, mask=m, other=0.0)
    x = acc.to(tl.bfloat16).to(tl.float32)
    if HAS_RES:                                               # 4. fused_triton _rms_fwd_k (HAS_RES) arithmetic
        r = tl.load(RES + row * N + offs, mask=m, other=0.0).to(tl.float32)
        s = (x + r).to(tl.bfloat16)
        tl.store(RES + row * N + offs, s, mask=m)
        x = s.to(tl.float32)
    else:
        tl.store(RES + row * N + offs, x.to(tl.bfloat16), mask=m)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    xn = (x * rstd).to(tl.bfloat16).to(tl.float32)
    w = tl.load(Wn + offs, mask=m, other=0.0).to(tl.float32)
    tl.store(Y + row * N + offs, (w * xn).to(tl.bfloat16), mask=m)


# ================================================================================================ hand-offs
class Mailbox:
    """per-GPU hand-off flags (int64 [NSLOT]) + error word + frame serial (ep)."""
    NSLOT = 16

    def __init__(self, dev):
        self.dev = dev
        self.flags = torch.zeros(self.NSLOT, dtype=torch.int64, device=dev)
        self.err = torch.zeros(1, dtype=torch.int64, device=dev)

    def fptr(self, slot):
        return self.flags.data_ptr() + slot * 8


@triton.jit
def _push_k(SRC, TAB, EP, n, BLOCK: tl.constexpr):
    """TAB int64 [2] = (destination address on the consumer GPU, its flag address): copy, then flag <- ep + 1 (release, sys)."""
    d = tl.load(TAB).to(tl.pointer_type(SRC.dtype.element_ty))
    for i0 in range(0, n, BLOCK):
        o = i0 + tl.arange(0, BLOCK)
        m = o < n
        tl.store(d + o, tl.load(SRC + o, mask=m), mask=m)
    tl.debug_barrier()
    _fence_sys(n)
    f = tl.load(TAB + 1).to(tl.pointer_type(tl.int64))
    tl.atomic_xchg(f, tl.load(EP) + 1, sem="relaxed", scope="sys")


def push(src, tab, ep):
    """src (local, contiguous) -> the hand-off destination in tab (int64 [2] device tensor: dst address, flag address)."""
    assert src.is_contiguous()
    _push_k[(1,)](src, tab, ep, src.numel(), BLOCK=1024, num_warps=4)


@triton.jit(do_not_specialize=["code"])
def _wait_k(F, EP, ERR, code, LIMIT: tl.constexpr):
    _spin_ge(F, tl.load(EP) + 1, ERR, code, LIMIT)


def wait(mb, slot, ep):
    _wait_k[(1,)](mb.flags[slot:slot + 1], ep, mb.err, 1000 + slot, LIMIT=SPIN_LIMIT, num_warps=1)


# ================================================================================================ trunk shard q / k / v post-processing
@triton.jit
def _qk_rows(QKV, WN, COS, SIN, T3, OUT, KC, rid, R, eps, MP, col0, NH: tl.constexpr, NKV: tl.constexpr, W: tl.constexpr, D: tl.constexpr,
             HALF: tl.constexpr, ROWS: tl.constexpr, TOCACHE: tl.constexpr):
    """ROWS (row r, head h) pairs (index r * NH + h) of q (TOCACHE False -> OUT [R * NH, D]) or k (-> KC[r, T3[r], h]): fused_triton
    _qk_fwd_k's per-head RMSNorm + RoPE arithmetic and [ROWS, HALF] tiles."""
    o1 = tl.arange(0, HALF)
    rows = (rid * ROWS + tl.arange(0, ROWS)).to(tl.int64)
    rm = rows < R * NH
    r = rows // NH
    h = rows % NH
    mk = rm[:, None]
    base = r[:, None] * W + col0 + h[:, None] * D
    x1 = tl.load(QKV + base + o1[None, :], mask=mk, other=0.0).to(tl.float32)
    x2 = tl.load(QKV + base + HALF + o1[None, :], mask=mk, other=0.0).to(tl.float32)
    var = (tl.sum(x1 * x1, axis=1) + tl.sum(x2 * x2, axis=1)) / D
    rstd = 1.0 / tl.sqrt(var + eps)
    w1 = tl.load(WN + o1).to(tl.float32)
    w2 = tl.load(WN + HALF + o1).to(tl.float32)
    q1 = (w1[None, :] * (x1 * rstd[:, None]).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    q2 = (w2[None, :] * (x2 * rstd[:, None]).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    cb = r[:, None] * D
    c1 = tl.load(COS + cb + o1[None, :], mask=mk, other=0.0).to(tl.float32)
    c2 = tl.load(COS + cb + HALF + o1[None, :], mask=mk, other=0.0).to(tl.float32)
    s1 = tl.load(SIN + cb + o1[None, :], mask=mk, other=0.0).to(tl.float32)
    s2 = tl.load(SIN + cb + HALF + o1[None, :], mask=mk, other=0.0).to(tl.float32)
    y1 = (q1 * c1 - q2 * s1).to(tl.bfloat16)
    y2 = (q2 * c2 + q1 * s2).to(tl.bfloat16)
    if TOCACHE:
        t = tl.load(T3 + r, mask=rm, other=0)
        ob = ((r * MP + t) * NKV + h)[:, None] * D
        tl.store(KC + ob + o1[None, :], y1, mask=mk)
        tl.store(KC + ob + HALF + o1[None, :], y2, mask=mk)
    else:
        ob = rows[:, None] * D
        tl.store(OUT + ob + o1[None, :], y1, mask=mk)
        tl.store(OUT + ob + HALF + o1[None, :], y2, mask=mk)


@triton.jit
def _qkv_post_k(QKV, QN, KN, COS, SIN, T3, Q, KC, VC, R, eps, MP, NQ: tl.constexpr, NKV: tl.constexpr, D: tl.constexpr, HALF: tl.constexpr,
                ROWS: tl.constexpr, PQ: tl.constexpr, PK: tl.constexpr):
    """programs [0, PQ): q rows (row = r * NQ + h, 16 per program = fused_triton _qk_fwd_k's tiles); [PQ, PQ + PK): k rows (r * NKV + h)
    -> the cache at T3[r]; the rest: v rows -> the cache. QKV [R, (NQ + 2 NKV) D]; COS / SIN [R, D]; KC / VC [R, MP, NKV, D]."""
    pid = tl.program_id(0)
    W: tl.constexpr = (NQ + 2 * NKV) * D
    if pid < PQ:
        _qk_rows(QKV, QN, COS, SIN, T3, Q, KC, pid, R, eps, MP, 0, NQ, NKV, W, D, HALF, ROWS, False)
    elif pid < PQ + PK:
        _qk_rows(QKV, KN, COS, SIN, T3, Q, KC, pid - PQ, R, eps, MP, NQ * D, NKV, NKV, W, D, HALF, ROWS, True)
    else:
        rows = ((pid - PQ - PK) * ROWS + tl.arange(0, ROWS)).to(tl.int64)
        rm = rows < R * NKV
        r = rows // NKV
        h = rows % NKV
        t = tl.load(T3 + r, mask=rm, other=0)
        oD = tl.arange(0, D)
        v = tl.load(QKV + r[:, None] * W + (NQ + NKV) * D + h[:, None] * D + oD[None, :], mask=rm[:, None], other=0.0)
        tl.store(VC + ((r * MP + t) * NKV + h)[:, None] * D + oD[None, :], v, mask=rm[:, None])


def qkv_post(qkv, qn, kn, eps, cos, sin, T3, q, kc, vc, nq, nkv, D):
    R = qkv.shape[0]
    PQ, PK, PV = triton.cdiv(R * nq, 16), triton.cdiv(R * nkv, 16), triton.cdiv(R * nkv, 16)
    _qkv_post_k[(PQ + PK + PV,)](qkv, qn, kn, cos, sin, T3, q, kc, vc, R, float(eps), kc.shape[1], NQ=nq, NKV=nkv, D=D, HALF=D // 2, ROWS=16,
                                 PQ=PQ, PK=PK, num_warps=4)
