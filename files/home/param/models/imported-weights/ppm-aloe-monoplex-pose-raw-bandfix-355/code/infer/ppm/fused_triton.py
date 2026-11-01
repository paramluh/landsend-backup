# VENDORED for aloe-pose-monoplex-infer. Upstream: ppm/aloe-m1vox-ft/code/vendor/fused_triton.py. Edits (behaviour-neutral for this model): none (verbatim).
"""speedopt B: hand-written Triton kernels (forward + backward, torch.autograd.Function) for the elementwise / row-wise
work of the Qwen3 decoder layer: RMSNorm, residual-add + RMSNorm, q/k per-head RMSNorm + RoPE (the student's TWO
rotations, or the teacher's one), SiLU(gate) * up, and the mix-RoPE gate + blend.

Why hand-written and not torch.compile: the kernels must run inside activation checkpointing AND inside selective
activation checkpointing, which executes the layer under a TorchDispatchMode. torch.compile refuses to run under such a
mode (dynamo: "non-infra torch dispatch mode present") - it silently falls back to eager and marks the frame skipped
for the rest of the process (measured). A Triton launch is not a dispatcher op, so these kernels behave identically
with or without any mode, compile once per dtype (the sequence length is a runtime argument, never a constexpr), and
need no inductor cache on the 112 ranks.

NUMERICS. The forward follows the eager HF/E25 arithmetic (fp32 statistics, the same bf16 roundings of the normalised
activations and the same blend algebra); intermediate products are kept in fp32 where eager rounds them to bf16, and
backward is computed in fp32 and rounded once. Differences to eager are bf16 round-off and, against an fp32 reference,
never larger than eager's own (tests/parity.py --ref, tests/triton_exact.py).
"""
import torch
import triton
import triton.language as tl


def _bf(*ts):
    """the kernels are written for bf16 activations and weights (FSDP2 MixedPrecisionPolicy param_dtype) - fail loudly."""
    for t in ts:
        assert t.dtype == torch.bfloat16 and t.is_cuda, (t.dtype, t.device)


def _nsm():
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


# ================================================================================================ RMSNorm (+ residual)
@triton.jit
def _rms_fwd_k(X, R, S, Y, W, RSTD, M, N, eps, HAS_RES: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    m = offs < N
    x = tl.load(X + row * N + offs, mask=m, other=0.0).to(tl.float32)
    if HAS_RES:
        r = tl.load(R + row * N + offs, mask=m, other=0.0).to(tl.float32)
        s = (x + r).to(tl.bfloat16)                          # eager: residual + attn, rounded
        tl.store(S + row * N + offs, s, mask=m)
        x = s.to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    xn = (x * rstd).to(tl.bfloat16).to(tl.float32)           # eager: hidden_states.to(input_dtype)
    w = tl.load(W + offs, mask=m, other=0.0).to(tl.float32)
    tl.store(Y + row * N + offs, (w * xn).to(tl.bfloat16), mask=m)
    tl.store(RSTD + row, rstd)


@triton.jit
def _rms_bwd_k(DY, DS, X, W, RSTD, DX, DWP, M, N, rows_per_prog, HAS_DS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    m = offs < N
    w = tl.load(W + offs, mask=m, other=0.0).to(tl.float32)
    dw = tl.zeros([BLOCK], dtype=tl.float32)
    r0 = pid * rows_per_prog
    for i in range(0, rows_per_prog):
        row = (r0 + i).to(tl.int64)
        if row < M:
            x = tl.load(X + row * N + offs, mask=m, other=0.0).to(tl.float32)
            rstd = tl.load(RSTD + row)
            dy = tl.load(DY + row * N + offs, mask=m, other=0.0).to(tl.float32)
            xh = x * rstd
            xn = xh.to(tl.bfloat16).to(tl.float32)
            dw += dy * xn
            g = dy * w
            dx = rstd * (g - xh * (tl.sum(g * xh, axis=0) / N))
            if HAS_DS:
                dx += tl.load(DS + row * N + offs, mask=m, other=0.0).to(tl.float32)
            tl.store(DX + row * N + offs, dx.to(tl.bfloat16), mask=m)
    tl.store(DWP + pid * N + offs, dw, mask=m)


def _rms_bwd(dy, ds, x, w, rstd):
    M, N = x.shape
    nprog = min(M, 4 * _nsm())
    rpp = triton.cdiv(M, nprog)
    nprog = triton.cdiv(M, rpp)
    dx = torch.empty_like(x)
    dwp = torch.empty(nprog, N, dtype=torch.float32, device=x.device)
    _rms_bwd_k[(nprog,)](dy, ds if ds is not None else dy, x, w, rstd, dx, dwp, M, N, rpp, HAS_DS=ds is not None,
                         BLOCK=triton.next_power_of_2(N), num_warps=8 if N >= 2048 else 4)
    return dx, dwp.sum(0).to(w.dtype)


class _RMS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, eps):
        _bf(x, w)
        shp = x.shape
        x2 = x.reshape(-1, shp[-1]).contiguous()
        M, N = x2.shape
        y = torch.empty_like(x2)
        rstd = torch.empty(M, dtype=torch.float32, device=x.device)
        _rms_fwd_k[(M,)](x2, x2, x2, y, w, rstd, M, N, eps, HAS_RES=False, BLOCK=triton.next_power_of_2(N),
                         num_warps=8 if N >= 2048 else 4)
        ctx.save_for_backward(x2, w, rstd)
        ctx.shp = shp
        return y.view(shp)

    @staticmethod
    def backward(ctx, dy):
        x2, w, rstd = ctx.saved_tensors
        dx, dw = _rms_bwd(dy.reshape(x2.shape).contiguous(), None, x2, w, rstd)
        return dx.view(ctx.shp), dw, None


class _AddRMS(torch.autograd.Function):
    """(s, y) = (res + a, RMSNorm(res + a))."""
    @staticmethod
    def forward(ctx, res, a, w, eps):
        _bf(res, a, w)
        shp = res.shape
        r2, a2 = res.reshape(-1, shp[-1]).contiguous(), a.reshape(-1, shp[-1]).contiguous()
        M, N = r2.shape
        s, y = torch.empty_like(r2), torch.empty_like(r2)
        rstd = torch.empty(M, dtype=torch.float32, device=res.device)
        _rms_fwd_k[(M,)](a2, r2, s, y, w, rstd, M, N, eps, HAS_RES=True, BLOCK=triton.next_power_of_2(N), num_warps=8)
        ctx.save_for_backward(s, w, rstd)
        ctx.shp = shp
        return s.view(shp), y.view(shp)

    @staticmethod
    def backward(ctx, ds, dy):
        s, w, rstd = ctx.saved_tensors
        dx, dw = _rms_bwd(dy.reshape(s.shape).contiguous(), ds.reshape(s.shape).contiguous(), s, w, rstd)
        dx = dx.view(ctx.shp)
        return dx, dx, dw, None


# ================================================================================================ q/k norm + RoPE
@triton.jit
def _qk_fwd_k(X, W, CW, SW, CF, SF, OW, OF, RSTD, R, NH, eps, TWO: tl.constexpr, D: tl.constexpr, HALF: tl.constexpr,
              ROWS: tl.constexpr):
    """ROWS rows of D per program; row r = t * NH + h. X [R, D] (a [T, NH*D] tensor), cos/sin [T, D]."""
    pid = tl.program_id(0)
    rows = (pid * ROWS + tl.arange(0, ROWS)).to(tl.int64)
    rm = rows < R
    t = rows // NH
    o1 = tl.arange(0, HALF)
    mk = rm[:, None]
    base = rows[:, None] * D
    x1 = tl.load(X + base + o1[None, :], mask=mk, other=0.0).to(tl.float32)
    x2 = tl.load(X + base + HALF + o1[None, :], mask=mk, other=0.0).to(tl.float32)
    var = (tl.sum(x1 * x1, axis=1) + tl.sum(x2 * x2, axis=1)) / D
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(RSTD + rows, rstd, mask=rm)
    w1 = tl.load(W + o1).to(tl.float32)
    w2 = tl.load(W + HALF + o1).to(tl.float32)
    q1 = (w1[None, :] * (x1 * rstd[:, None]).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    q2 = (w2[None, :] * (x2 * rstd[:, None]).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    cb = t[:, None] * D
    c1 = tl.load(CW + cb + o1[None, :], mask=mk, other=0.0).to(tl.float32)
    c2 = tl.load(CW + cb + HALF + o1[None, :], mask=mk, other=0.0).to(tl.float32)
    s1 = tl.load(SW + cb + o1[None, :], mask=mk, other=0.0).to(tl.float32)
    s2 = tl.load(SW + cb + HALF + o1[None, :], mask=mk, other=0.0).to(tl.float32)
    tl.store(OW + base + o1[None, :], (q1 * c1 - q2 * s1).to(tl.bfloat16), mask=mk)       # rotate_half(q) = [-q2, q1]
    tl.store(OW + base + HALF + o1[None, :], (q2 * c2 + q1 * s2).to(tl.bfloat16), mask=mk)
    if TWO:
        c1 = tl.load(CF + cb + o1[None, :], mask=mk, other=0.0).to(tl.float32)
        c2 = tl.load(CF + cb + HALF + o1[None, :], mask=mk, other=0.0).to(tl.float32)
        s1 = tl.load(SF + cb + o1[None, :], mask=mk, other=0.0).to(tl.float32)
        s2 = tl.load(SF + cb + HALF + o1[None, :], mask=mk, other=0.0).to(tl.float32)
        tl.store(OF + base + o1[None, :], (q1 * c1 - q2 * s1).to(tl.bfloat16), mask=mk)
        tl.store(OF + base + HALF + o1[None, :], (q2 * c2 + q1 * s2).to(tl.bfloat16), mask=mk)


@triton.jit
def _qk_bwd_k(GW, GF, X, W, CW, SW, CF, SF, RSTD, DX, DWP, R, NH, iters, TWO: tl.constexpr, D: tl.constexpr,
              HALF: tl.constexpr, ROWS: tl.constexpr):
    pid = tl.program_id(0)
    o1 = tl.arange(0, HALF)
    w1 = tl.load(W + o1).to(tl.float32)
    w2 = tl.load(W + HALF + o1).to(tl.float32)
    dw1 = tl.zeros([HALF], dtype=tl.float32)
    dw2 = tl.zeros([HALF], dtype=tl.float32)
    for it in range(0, iters):
        rows = ((pid * iters + it) * ROWS + tl.arange(0, ROWS)).to(tl.int64)
        rm = rows < R
        mk = rm[:, None]
        t = rows // NH
        base = rows[:, None] * D
        cb = t[:, None] * D
        rstd = tl.load(RSTD + rows, mask=rm, other=0.0)
        x1 = tl.load(X + base + o1[None, :], mask=mk, other=0.0).to(tl.float32)
        x2 = tl.load(X + base + HALF + o1[None, :], mask=mk, other=0.0).to(tl.float32)
        # d out / d q through out1 = q1 c1 - q2 s1, out2 = q2 c2 + q1 s2
        g1 = tl.load(GW + base + o1[None, :], mask=mk, other=0.0).to(tl.float32)
        g2 = tl.load(GW + base + HALF + o1[None, :], mask=mk, other=0.0).to(tl.float32)
        c1 = tl.load(CW + cb + o1[None, :], mask=mk, other=0.0).to(tl.float32)
        c2 = tl.load(CW + cb + HALF + o1[None, :], mask=mk, other=0.0).to(tl.float32)
        s1 = tl.load(SW + cb + o1[None, :], mask=mk, other=0.0).to(tl.float32)
        s2 = tl.load(SW + cb + HALF + o1[None, :], mask=mk, other=0.0).to(tl.float32)
        gq1 = g1 * c1 + g2 * s2
        gq2 = g2 * c2 - g1 * s1
        if TWO:
            g1 = tl.load(GF + base + o1[None, :], mask=mk, other=0.0).to(tl.float32)
            g2 = tl.load(GF + base + HALF + o1[None, :], mask=mk, other=0.0).to(tl.float32)
            c1 = tl.load(CF + cb + o1[None, :], mask=mk, other=0.0).to(tl.float32)
            c2 = tl.load(CF + cb + HALF + o1[None, :], mask=mk, other=0.0).to(tl.float32)
            s1 = tl.load(SF + cb + o1[None, :], mask=mk, other=0.0).to(tl.float32)
            s2 = tl.load(SF + cb + HALF + o1[None, :], mask=mk, other=0.0).to(tl.float32)
            gq1 += g1 * c1 + g2 * s2
            gq2 += g2 * c2 - g1 * s1
        xh1 = x1 * rstd[:, None]
        xh2 = x2 * rstd[:, None]
        dw1 += tl.sum(gq1 * xh1.to(tl.bfloat16).to(tl.float32), axis=0)
        dw2 += tl.sum(gq2 * xh2.to(tl.bfloat16).to(tl.float32), axis=0)
        gn1 = gq1 * w1[None, :]
        gn2 = gq2 * w2[None, :]
        dot = (tl.sum(gn1 * xh1, axis=1) + tl.sum(gn2 * xh2, axis=1)) / D
        tl.store(DX + base + o1[None, :], (rstd[:, None] * (gn1 - xh1 * dot[:, None])).to(tl.bfloat16), mask=mk)
        tl.store(DX + base + HALF + o1[None, :], (rstd[:, None] * (gn2 - xh2 * dot[:, None])).to(tl.bfloat16), mask=mk)
    tl.store(DWP + pid * D + o1, dw1)
    tl.store(DWP + pid * D + HALF + o1, dw2)


_QROWS = 16


def _qk_fwd(x, w, eps, cw, sw, cf, sf, two):
    B, T, HD = x.shape
    D = w.shape[0]
    NH = HD // D
    R = B * T * NH
    assert B == 1 and cw.shape == (1, T, D), (x.shape, cw.shape)
    x = x.contiguous()
    ow = torch.empty(B, T, NH, D, dtype=x.dtype, device=x.device)
    of = torch.empty_like(ow) if two else ow
    rstd = torch.empty(R, dtype=torch.float32, device=x.device)
    _qk_fwd_k[(triton.cdiv(R, _QROWS),)](x, w, cw.contiguous(), sw.contiguous(), cf.contiguous() if two else cw,
                                         sf.contiguous() if two else sw, ow, of, rstd, R, NH, eps, TWO=two, D=D,
                                         HALF=D // 2, ROWS=_QROWS, num_warps=4)
    return ow, of, rstd


def _qk_bwd(gw, gf, x, w, cw, sw, cf, sf, rstd, two):
    B, T, HD = x.shape
    D = w.shape[0]
    NH = HD // D
    R = B * T * NH
    blocks = triton.cdiv(R, _QROWS)
    nprog = min(blocks, 8 * _nsm())
    iters = triton.cdiv(blocks, nprog)
    nprog = triton.cdiv(blocks, iters)
    dx = torch.empty_like(x)
    dwp = torch.empty(nprog, D, dtype=torch.float32, device=x.device)
    _qk_bwd_k[(nprog,)](gw.contiguous(), gf.contiguous() if two else gw, x, w, cw, sw, cf if two else cw, sf if two else sw,
                        rstd, dx, dwp, R, NH, iters, TWO=two, D=D, HALF=D // 2, ROWS=_QROWS, num_warps=4)
    return dx, dwp.sum(0).to(w.dtype)


class _QKPrepMix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qlin, klin, wq, wk, eps, cw, sw, cf, sf):
        _bf(qlin, klin, wq, wk, cw, sw, cf, sf)
        cw, sw, cf, sf = cw.contiguous(), sw.contiguous(), cf.contiguous(), sf.contiguous()
        qlin, klin = qlin.contiguous(), klin.contiguous()
        qw, qf, rq = _qk_fwd(qlin, wq, eps, cw, sw, cf, sf, True)
        kw, kf, rk = _qk_fwd(klin, wk, eps, cw, sw, cf, sf, True)
        ctx.save_for_backward(qlin, klin, wq, wk, cw, sw, cf, sf, rq, rk)
        return qw, kw, qf, kf

    @staticmethod
    def backward(ctx, gqw, gkw, gqf, gkf):
        qlin, klin, wq, wk, cw, sw, cf, sf, rq, rk = ctx.saved_tensors
        dq, dwq = _qk_bwd(gqw, gqf, qlin, wq, cw, sw, cf, sf, rq, True)
        dk, dwk = _qk_bwd(gkw, gkf, klin, wk, cw, sw, cf, sf, rk, True)
        return dq, dk, dwq, dwk, None, None, None, None, None


class _QKPrep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qlin, klin, wq, wk, eps, c, s):
        _bf(qlin, klin, wq, wk, c, s)
        c, s = c.contiguous(), s.contiguous()
        qlin, klin = qlin.contiguous(), klin.contiguous()
        q, _, rq = _qk_fwd(qlin, wq, eps, c, s, None, None, False)
        k, _, rk = _qk_fwd(klin, wk, eps, c, s, None, None, False)
        ctx.save_for_backward(qlin, klin, wq, wk, c, s, rq, rk)
        return q, k

    @staticmethod
    def backward(ctx, gq, gk):
        qlin, klin, wq, wk, c, s, rq, rk = ctx.saved_tensors
        dq, dwq = _qk_bwd(gq, None, qlin, wq, c, s, None, None, rq, False)
        dk, dwk = _qk_bwd(gk, None, klin, wk, c, s, None, None, rk, False)
        return dq, dk, dwq, dwk, None, None, None


# ================================================================================================ SiLU(gate) * up
@triton.jit
def _silu_mul_fwd_k(G, U, Y, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    g = tl.load(G + offs, mask=m, other=0.0).to(tl.float32)
    u = tl.load(U + offs, mask=m, other=0.0).to(tl.float32)
    a = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)            # eager: silu(gate) rounded, then * up
    tl.store(Y + offs, (a * u).to(tl.bfloat16), mask=m)


@triton.jit
def _silu_mul_bwd_k(DY, G, U, DG, DU, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    dy = tl.load(DY + offs, mask=m, other=0.0).to(tl.float32)
    g = tl.load(G + offs, mask=m, other=0.0).to(tl.float32)
    u = tl.load(U + offs, mask=m, other=0.0).to(tl.float32)
    sg = 1.0 / (1.0 + tl.exp(-g))
    a = (g * sg).to(tl.bfloat16).to(tl.float32)
    tl.store(DU + offs, (dy * a).to(tl.bfloat16), mask=m)
    tl.store(DG + offs, (dy * u * sg * (1.0 + g * (1.0 - sg))).to(tl.bfloat16), mask=m)


class _SiluMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, g, u):
        _bf(g, u)
        g, u = g.contiguous(), u.contiguous()
        y = torch.empty_like(g)
        n = g.numel()
        _silu_mul_fwd_k[(triton.cdiv(n, 2048),)](g, u, y, n, BLOCK=2048, num_warps=8)
        ctx.save_for_backward(g, u)
        return y

    @staticmethod
    def backward(ctx, dy):
        g, u = ctx.saved_tensors
        dg, du = torch.empty_like(g), torch.empty_like(u)
        n = g.numel()
        _silu_mul_bwd_k[(triton.cdiv(n, 2048),)](dy.contiguous(), g, u, dg, du, n, BLOCK=2048, num_warps=8)
        return dg, du


# ================================================================================================ gate + blend
@triton.jit
def _blend_fwd_k(Z, Bb, AW, AF, O, R, NH, D: tl.constexpr, ROWS: tl.constexpr):
    """row r = (t, h): g = sigmoid(z[r] + b[h]); out[r, :] = (1 - g) aw[r, :] + g af[r, :]"""
    rows = (tl.program_id(0) * ROWS + tl.arange(0, ROWS)).to(tl.int64)
    rm = rows < R
    h = rows % NH
    z = tl.load(Z + rows, mask=rm, other=0.0).to(tl.float32)
    b = tl.load(Bb + h, mask=rm, other=0.0).to(tl.float32)
    zb = (z + b).to(tl.bfloat16).to(tl.float32)                            # eager: x @ W + b in bf16
    g = (1.0 / (1.0 + tl.exp(-zb))).to(tl.bfloat16).to(tl.float32)        # eager: sigmoid in bf16
    om = (1.0 - g).to(tl.bfloat16).to(tl.float32)
    od = tl.arange(0, D)
    mk = rm[:, None]
    base = rows[:, None] * D + od[None, :]
    aw = tl.load(AW + base, mask=mk, other=0.0).to(tl.float32)
    af = tl.load(AF + base, mask=mk, other=0.0).to(tl.float32)
    tl.store(O + base, (om[:, None] * aw + g[:, None] * af).to(tl.bfloat16), mask=mk)


@triton.jit
def _blend_bwd_k(GO, Z, Bb, AW, AF, DAW, DAF, DZ, R, NH, D: tl.constexpr, ROWS: tl.constexpr):
    rows = (tl.program_id(0) * ROWS + tl.arange(0, ROWS)).to(tl.int64)
    rm = rows < R
    h = rows % NH
    z = tl.load(Z + rows, mask=rm, other=0.0).to(tl.float32)
    b = tl.load(Bb + h, mask=rm, other=0.0).to(tl.float32)
    zb = (z + b).to(tl.bfloat16).to(tl.float32)
    sg = 1.0 / (1.0 + tl.exp(-zb))
    g = sg.to(tl.bfloat16).to(tl.float32)
    om = (1.0 - g).to(tl.bfloat16).to(tl.float32)
    od = tl.arange(0, D)
    mk = rm[:, None]
    base = rows[:, None] * D + od[None, :]
    go = tl.load(GO + base, mask=mk, other=0.0).to(tl.float32)
    aw = tl.load(AW + base, mask=mk, other=0.0).to(tl.float32)
    af = tl.load(AF + base, mask=mk, other=0.0).to(tl.float32)
    tl.store(DAW + base, (go * om[:, None]).to(tl.bfloat16), mask=mk)
    tl.store(DAF + base, (go * g[:, None]).to(tl.bfloat16), mask=mk)
    dg = tl.sum(go * (af - aw), axis=1)
    tl.store(DZ + rows, (dg * sg * (1.0 - sg)).to(tl.bfloat16), mask=rm)


_BROWS = 16


class _GateBlend(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, b, aw, af):
        _bf(z, b, aw, af)
        B, T, NH, D = aw.shape
        z, aw, af = z.contiguous(), aw.contiguous(), af.contiguous()
        o = torch.empty(B, T, NH * D, dtype=aw.dtype, device=aw.device)
        R = B * T * NH
        _blend_fwd_k[(triton.cdiv(R, _BROWS),)](z, b, aw, af, o, R, NH, D=D, ROWS=_BROWS, num_warps=4)
        ctx.save_for_backward(z, b, aw, af)
        return o

    @staticmethod
    def backward(ctx, go):
        z, b, aw, af = ctx.saved_tensors
        B, T, NH, D = aw.shape
        R = B * T * NH
        daw, daf, dz = torch.empty_like(aw), torch.empty_like(af), torch.empty_like(z)
        _blend_bwd_k[(triton.cdiv(R, _BROWS),)](go.contiguous(), z, b, aw, af, daw, daf, dz, R, NH, D=D, ROWS=_BROWS,
                                                 num_warps=4)
        db = dz.float().sum(dim=(0, 1)).to(b.dtype)
        return dz, db, daw, daf


# ================================================================================================ public API
def rms(x, w, eps):
    return _RMS.apply(x, w, float(eps))


def add_rms(res, y, w, eps):
    """(res + y, RMSNorm(res + y)) - the attention residual add fused with post_attention_layernorm."""
    return _AddRMS.apply(res, y, w, float(eps))


def silu_mul(g, u):
    return _SiluMul.apply(g, u)


def qk_prep_mix(qlin, klin, wq, wk, eps, cw, sw, cf, sf):
    """E25 student: q/k per-head RMSNorm, then BOTH rotations. qlin [1,T,H*D], klin [1,T,KV*D], cos/sin [1,T,D].
    -> qw, kw (word RoPE), qf, kf (frame RoPE), each [1,T,heads,D] contiguous (FA3's native layout)."""
    return _QKPrepMix.apply(qlin, klin, wq, wk, float(eps), cw, sw, cf, sf)


def qk_prep(qlin, klin, wq, wk, eps, c, s):
    """stock (teacher): per-head RMSNorm + one rotation -> q, k [1,T,heads,D] contiguous."""
    return _QKPrep.apply(qlin, klin, wq, wk, float(eps), c, s)


def gate_blend(z, b, aw, af):
    """z = x @ W_router (eager matmul, bf16) [1,T,H], b [H]: out = (1-g) aw + g af, g = sigmoid(z + b), as [1,T,H*D]."""
    return _GateBlend.apply(z, b, aw, af)


def forced_blend(aw, af, gval):
    """E25 forced gate (eval only), eager: exactly the live code's arithmetic."""
    gg = torch.full((*aw.shape[:-1], 1), float(gval), dtype=aw.dtype, device=aw.device)
    return ((1 - gg) * aw + gg * af).flatten(2)
