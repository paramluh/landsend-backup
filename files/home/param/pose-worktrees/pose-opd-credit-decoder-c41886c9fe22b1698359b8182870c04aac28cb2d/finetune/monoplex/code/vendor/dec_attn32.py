"""Fused causal GQA attention for the 0.6B depth decoders' SHORT sequences (T = 32 positions per frame, head_dim 128), forward + backward
as Triton kernels behind torch custom ops (opaque to torch.compile). One program per (frame, kv head) holds the whole 32 x 32 score tile:
  forward   S = Q K^T * scale (fp32 accumulate), causal mask, softmax in fp32, O = P V with P cast to bf16 and fp32 accumulation, LSE kept
  backward  P recomputed from LSE (fp32); dV = P^T dO, dP = dO V^T, dS = P (dP - rowsum(dO * O)), dQ = dS K * scale, dK = dS^T Q * scale
            (bf16 GEMM operands, fp32 accumulation) = FlashAttention-2's numerics. Same function as SDPA with a causal mask; kernel-level rounding
            only. Replaces the SDPA memory-efficient kernel the explicit HF mask forced (25% of the decoder's GPU time at T = 32)."""
import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _fwd(Q, K, V, O, L, sqb, sqh, sqt, skb, skh, skt, sob, soh, sot, scale, H: tl.constexpr, R: tl.constexpr, T: tl.constexpr, D: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // (H // R)
    g = pid % (H // R)
    t = tl.arange(0, T)
    d = tl.arange(0, D)
    kv_off = b * skb + g * skh + t[:, None] * skt + d[None, :]
    k = tl.load(K + kv_off)
    v = tl.load(V + kv_off)
    causal = t[:, None] >= t[None, :]
    for r in tl.static_range(R):
        h = g * R + r
        q = tl.load(Q + b * sqb + h * sqh + t[:, None] * sqt + d[None, :])
        s = tl.dot(q, tl.trans(k)) * scale
        s = tl.where(causal, s, float("-inf"))
        m = tl.max(s, 1)
        p = tl.exp(s - m[:, None])
        l = tl.sum(p, 1)
        p = p / l[:, None]
        o = tl.dot(p, v.to(tl.float32), input_precision="ieee")      # P V in fp32 (P never rounded to bf16)
        tl.store(O + b * sob + h * soh + t[:, None] * sot + d[None, :], o.to(O.dtype.element_ty))
        tl.store(L + (b * H + h) * T + t, m + tl.log(l))


@triton.jit
def _bwd(Q, K, V, O, DO, L, DQ, DK, DV, sqb, sqh, sqt, skb, skh, skt, sob, soh, sot, sdb, sdh, sdt, scale,
         H: tl.constexpr, R: tl.constexpr, T: tl.constexpr, D: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // (H // R)
    g = pid % (H // R)
    t = tl.arange(0, T)
    d = tl.arange(0, D)
    kv_off = b * skb + g * skh + t[:, None] * skt + d[None, :]
    k = tl.load(K + kv_off)
    v = tl.load(V + kv_off)
    causal = t[:, None] >= t[None, :]
    dk = tl.zeros((T, D), dtype=tl.float32)
    dv = tl.zeros((T, D), dtype=tl.float32)
    for r in tl.static_range(R):
        h = g * R + r
        q = tl.load(Q + b * sqb + h * sqh + t[:, None] * sqt + d[None, :])
        o = tl.load(O + b * sob + h * soh + t[:, None] * sot + d[None, :])
        do = tl.load(DO + b * sdb + h * sdh + t[:, None] * sdt + d[None, :])
        lse = tl.load(L + (b * H + h) * T + t)
        s = tl.dot(q, tl.trans(k)) * scale
        p = tl.where(causal, tl.exp(s - lse[:, None]), 0.0)
        dv += tl.dot(tl.trans(p), do.to(tl.float32), input_precision="ieee")
        dp = tl.dot(do, tl.trans(v))
        delta = tl.sum(do.to(tl.float32) * o.to(tl.float32), 1)
        ds = p * (dp - delta[:, None])
        dsb = ds.to(q.dtype)
        dq = tl.dot(dsb, k) * scale
        tl.store(DQ + b * sqb + h * sqh + t[:, None] * sqt + d[None, :], dq.to(DQ.dtype.element_ty))
        dk += tl.dot(tl.trans(dsb), q)
    tl.store(DK + kv_off, (dk * scale).to(DK.dtype.element_ty))
    tl.store(DV + kv_off, dv.to(DV.dtype.element_ty))


def _check(q, k, v):
    B, T, H, D = q.shape
    Hk = k.shape[2]
    assert T == 32 and D == 128 and H % Hk == 0 and q.dtype == torch.bfloat16, (q.shape, k.shape, q.dtype)
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous(), "attn32: contiguous [B, T, H, D] inputs" 
    return B, T, H, D, Hk


@torch.library.custom_op("ppmspeed::attn32_fwd", mutates_args=())
def attn32_fwd(q: Tensor, k: Tensor, v: Tensor, scale: float) -> tuple[Tensor, Tensor]:
    """q [B, 32, H, 128], k / v [B, 32, Hk, 128] (contiguous) -> (o [B, 32, H, 128] contiguous, lse [B, H, 32] fp32)."""
    B, T, H, D, Hk = _check(q, k, v)
    o = torch.empty((B, T, H, D), dtype=q.dtype, device=q.device)
    lse = torch.empty((B, H, T), dtype=torch.float32, device=q.device)
    if B:
        _fwd[(B * Hk,)](q, k, v, o, lse, q.stride(0), q.stride(2), q.stride(1), k.stride(0), k.stride(2), k.stride(1),
                        o.stride(0), o.stride(2), o.stride(1), float(scale), H=H, R=H // Hk, T=T, D=D, num_warps=4, num_stages=1)
    return o, lse


@attn32_fwd.register_fake
def _(q, k, v, scale):
    B, T, H, D = q.shape
    return q.new_empty((B, T, H, D)), q.new_empty((B, H, T), dtype=torch.float32)


@torch.library.custom_op("ppmspeed::attn32_bwd", mutates_args=())
def attn32_bwd(do: Tensor, q: Tensor, k: Tensor, v: Tensor, o: Tensor, lse: Tensor, scale: float) -> tuple[Tensor, Tensor, Tensor]:
    B, T, H, D, Hk = _check(q, k, v)
    do = do.contiguous()
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    if B:
        _bwd[(B * Hk,)](q, k, v, o, do, lse, dq, dk, dv, q.stride(0), q.stride(2), q.stride(1), k.stride(0), k.stride(2), k.stride(1),
                        o.stride(0), o.stride(2), o.stride(1), do.stride(0), do.stride(2), do.stride(1), float(scale),
                        H=H, R=H // Hk, T=T, D=D, num_warps=4, num_stages=1)
    return dq, dk, dv


@attn32_bwd.register_fake
def _(do, q, k, v, o, lse, scale):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _setup(ctx, inputs, output):
    q, k, v, scale = inputs
    o, lse = output
    ctx.save_for_backward(q, k, v, o, lse)
    ctx.scale = scale


def _backward(ctx, do, dlse):
    q, k, v, o, lse = ctx.saved_tensors
    dq, dk, dv = attn32_bwd(do, q, k, v, o, lse, ctx.scale)
    return dq, dk, dv, None


attn32_fwd.register_autograd(_backward, setup_context=_setup)


def attn32(q, k, v, scale):
    """[B, 32, H, 128] / [B, 32, Hk, 128] (bf16) -> o [B, 32, H, 128]. Causal."""
    return attn32_fwd(q.contiguous(), k.contiguous(), v.contiguous(), float(scale))[0]
