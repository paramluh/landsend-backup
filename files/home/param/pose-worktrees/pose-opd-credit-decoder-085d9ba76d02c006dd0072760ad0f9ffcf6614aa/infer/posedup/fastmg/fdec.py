"""fastmg FAST 0.6B DEPTH DECODER (10-25): the live engine's FusedDec / FusedDecS (light_kernels path: 7 kernels per layer) with the same
weights, KV caches, RoPE tables, sampling code and RNG use, but 5 kernels per layer on tensor cores:
  K1  residual add + RMSNorm (input_layernorm) recomputed by every program from the residual rows, then the q|k|v GEMV
  K2  light_kernels.light_attn (unchanged: q / k norm + RoPE + cache write + attention)
  K3  o_proj GEMV
  K4  residual add + RMSNorm (post_attention_layernorm) + gate|up GEMV
  K5  silu(gate) * up computed on the fly + down_proj GEMV
and the final norm + codebook head as one more K1-type kernel (fp32 logits of each row's last token). The residual stream ping-pongs
between two buffers (program 0 of K1 / K4 writes the new residual; every program reads the old one).
MATHS: the same ops and bf16 rounding points as FusedDec (residual add rounded to bf16, fp32 RMS statistics, bf16 normalised activations,
fp32-accumulated GEMVs rounded to bf16, silu / product roundings of light_kernels.silu_mul_gu); summation orders inside the RMS statistic
and the GEMVs differ (tensor-core dot vs CUDA-core FMA) -> bf16 round-off level differences, not bitwise."""
import torch
import triton
import triton.language as tl

import tpk

BF = torch.bfloat16


@triton.autotune(configs=tpk._gemv_cfgs(), key=["N", "K", "M", "HAS_MO", "F32"], prune_configs_by={"early_config_prune": tpk._prune})
@triton.jit
def _rms_gemv_k(RIN, MO, ROUT, LNW, eps, W, Y, M, N, K, LSTR, LOFF, HAS_MO: tl.constexpr, HAS_ROUT: tl.constexpr, F32: tl.constexpr,
                BN: tl.constexpr, BK: tl.constexpr):
    """output row r <- input row r * LSTR + LOFF: s = bf16(rin + mo) (HAS_MO) else rin; y[r] = (w_ln * bf16(s * rstd)) @ W^T."""
    pid = tl.program_id(0)
    mi = tl.arange(0, 16)
    msk = mi < M
    src = mi * LSTR + LOFF
    ss = tl.zeros([16], dtype=tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + tl.arange(0, BK)
        s = tl.load(RIN + src[:, None] * K + kk[None, :], mask=msk[:, None], other=0.0).to(tl.float32)
        if HAS_MO:
            s = (tl.load(MO + src[:, None] * K + kk[None, :], mask=msk[:, None], other=0.0).to(tl.float32) + s).to(tl.bfloat16).to(tl.float32)
        ss += tl.sum(s * s, axis=1)
        if HAS_ROUT:
            if pid == 0:
                tl.store(ROUT + src[:, None] * K + kk[None, :], s.to(tl.bfloat16), mask=msk[:, None])
    rstd = 1.0 / tl.sqrt(ss / K + eps)
    n = pid * BN + tl.arange(0, BN)
    acc = tl.zeros([16, BN], dtype=tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + tl.arange(0, BK)
        s = tl.load(RIN + src[:, None] * K + kk[None, :], mask=msk[:, None], other=0.0).to(tl.float32)
        if HAS_MO:
            s = (tl.load(MO + src[:, None] * K + kk[None, :], mask=msk[:, None], other=0.0).to(tl.float32) + s).to(tl.bfloat16).to(tl.float32)
        xn = (s * rstd[:, None]).to(tl.bfloat16).to(tl.float32)
        x = (tl.load(LNW + kk).to(tl.float32)[None, :] * xn).to(tl.bfloat16)
        w = tl.load(W + n[:, None].to(tl.int64) * K + kk[None, :])
        acc = tl.dot(x, tl.trans(w), acc)
    if F32:
        tl.store(Y + mi[:, None] * N + n[None, :], acc, mask=msk[:, None])
    else:
        tl.store(Y + mi[:, None] * N + n[None, :], acc.to(tl.bfloat16), mask=msk[:, None])


def rms_gemv(rin, mo, rout, lnw, eps, w, y, M, lstr=1, loff=0, f32=False):
    N, K = w.shape
    _rms_gemv_k[lambda m: (triton.cdiv(N, m["BN"]),)](rin, mo if mo is not None else rin, rout if rout is not None else rin, lnw, float(eps), w, y,
                                                      M, N, K, lstr, loff, HAS_MO=mo is not None, HAS_ROUT=rout is not None, F32=f32)
    return y


class FastDec:
    """wraps a live FusedDec (S = 1) or FusedDecS (S rows): same weights / caches / RoPE / gen() semantics; step_head() = 5-kernel layers."""

    def __init__(self, fd, S=1):
        self.fd, self.S = fd, int(S)
        self.dec = fd.dec
        dev = fd.dev
        L0 = fd.layers[0]
        Hl = L0["wqkv"].shape[1]
        self.Hl = Hl
        self.nq, self.kv, self.D = fd.nh, fd.kv, fd.D
        I2 = L0["wgu"].shape[0]
        Mx = 2 * self.S
        z = lambda n: torch.zeros(Mx, n, dtype=BF, device=dev)
        self.rA, self.rB, self.qkv, self.ao, self.gu, self.mo = z(Hl), z(Hl), z(L0["wqkv"].shape[0]), z(Hl), z(I2), z(Hl)
        self.a = z(self.nq * self.D)
        self.logit = torch.zeros(self.S, fd.W.shape[1], dtype=torch.float32, device=dev)
        self.single = not hasattr(fd, "S")                       # FusedDec (cs[j] = ([1, L, D], ..)) vs FusedDecS ([S * L, D])

    def step_head(self, x, j, d):
        """x [S, L, Hl] decoder inputs at depth step j -> fp32 logits [S, CBS] of codebook d (= fd.head(fd.step(x, j), d))."""
        fd = self.fd
        S, L, Hl = x.shape
        M = S * L
        x2 = x.reshape(M, Hl)
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        c, s = fd.cs[j]
        if self.single:
            c, s = c[0], s[0]
        t0 = 0 if j == 0 else j + 1
        rin, mo = x2, None
        for i, Ly in enumerate(fd.layers):
            rms_gemv(rin, mo, self.rB, Ly["ln1"], Ly["e1"], Ly["wqkv"], self.qkv, M)
            tpk_lk().light_attn(self.qkv[:M], Ly["qn"], Ly["kn"], Ly["eqk"], c, s, fd.K[i], fd.V[i], t0, L, self.nq, self.kv, self.D, Ly["sc"],
                                out=self.a[:M])
            tpk.gemv(self.a[:M], Ly["wo"], self.ao[:M])
            rms_gemv(self.rB, self.ao, self.rA, Ly["ln2"], Ly["e2"], Ly["wgu"], self.gu, M)
            tpk.gemv(self.gu[:M], Ly["wd"], self.mo[:M], silu=True)
            rin, mo = self.rA, self.mo
        rms_gemv(rin, mo, None, fd.nw, fd.ne, fd.W[d - 1], self.logit, S, lstr=L, loff=L - 1, f32=True)
        return self.logit

    def gen(self, hv, c0, gen, temp, topk, pres=None, rp=None, zb=None, cfg_w=None, cfg_on=None):
        """= FusedDec.gen (S = 1) / FusedDecS.gen (S = 2 NI with CFG): the same slot-0 / embedding / CFG mix / penalty / sampling ops."""
        import live_ppm_pose as LP
        dec, fd = self.dec, self.fd
        NI = c0.shape[0]
        S = hv.shape[0]
        two = S == 2 * NI
        assert S == self.S and (S == NI or two), (S, NI, self.S)
        cc = torch.cat([c0, c0]) if two else c0
        if self.single:
            s0 = dec.proj(hv)[:, None].to(BF) if zb is None else (dec.proj(hv).to(BF) + zb.to(BF))[:, None].to(BF)
        else:
            s0 = fd.slot0(hv, zb)[:, None]
        x = torch.cat([s0, dec.cb0_emb(cc)[:, None].to(BF)], 1)
        out = [c0]
        for d in range(1, LP.DEPTH):
            zd = self.step_head(x, d - 1, d)
            if two:
                zc, zu = zd[:NI], zd[NI:]
                zd = torch.where(cfg_on, zu + cfg_w * (zc - zu), zc)
            if pres is not None:
                zd = LP.ctrl_penalty(zd, pres[:, d], rp)
            cd = LP.t_sample(zd, gen, temp, topk)
            out.append(cd)
            if d < LP.DEPTH - 1:
                x = dec.model.embed_tokens((torch.cat([cd, cd]) if two else cd) + (d - 1) * LP.CBS)[:, None].to(BF)
        return torch.stack(out, 1)


_LK = None


def tpk_lk():
    global _LK
    if _LK is None:
        import live_ppm_pose as LP
        _LK = LP.LK
    return _LK
