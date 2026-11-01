"""speedopt B: the pinned FA3 forward/backward as OPAQUE torch custom ops (speedopt::fa3_fwd / fa3_bwd) around the very
same pinned kernels (_flash_attn_forward / _flash_attn_backward, the arguments FlashAttnFunc passes). Why: the pinned
package's raw torch.ops.<ns>.fwd cannot be re-dispatched from Python ("Unable to cast Python instance of type list"),
which is exactly what a TorchDispatchMode (selective activation checkpointing) does to every op; a custom op is
dispatched as one unit and runs the kernel below the mode. It also gives torch.compile a fake (meta) implementation."""
import torch
from torch import Tensor


def _mod():
    import pinned_fa3
    return pinned_fa3._load_module()


@torch.library.custom_op("speedopt::fa3_fwd", mutates_args=())
def _fa3_fwd(q: Tensor, k: Tensor, v: Tensor, scale: float, causal: bool) -> tuple[Tensor, Tensor]:
    fi = _mod().flash_attn_interface
    out, lse, *_ = fi._flash_attn_forward(q, k, v, None, None, None, None, None, None, None, None, None, None, None,
                                          None, None, None, None, None, None, None, None, None, scale, causal=causal,
                                          window_size=(-1, -1), softcap=0.0, num_splits=1, pack_gqa=None, sm_margin=0)
    return out, lse


@_fa3_fwd.register_fake
def _(q, k, v, scale, causal):
    B, T, H, D = q.shape
    return q.new_empty(B, T, H, v.shape[-1]), q.new_empty(B, H, T, dtype=torch.float32)


@torch.library.custom_op("speedopt::fa3_bwd", mutates_args=())
def _fa3_bwd(dout: Tensor, q: Tensor, k: Tensor, v: Tensor, out: Tensor, lse: Tensor, scale: float,
             causal: bool) -> tuple[Tensor, Tensor, Tensor]:
    fi = _mod().flash_attn_interface
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    fi._flash_attn_backward(dout, q, k, v, out, lse, None, None, None, None, None, None, dq, dk, dv, scale, causal,
                            (-1, -1), 0.0, False, 0)
    return dq, dk, dv


@_fa3_bwd.register_fake
def _(dout, q, k, v, out, lse, scale, causal):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


class _FA3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale, causal):
        out, lse = torch.ops.speedopt.fa3_fwd(q, k, v, scale, causal)
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.scale, ctx.causal = scale, causal
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = torch.ops.speedopt.fa3_bwd(dout, q, k, v, out, lse, ctx.scale, ctx.causal)
        return dq, dk, dv, None, None


def fa3_func(q, k, v, softmax_scale, causal):
    """== flash_attn_func(q, k, v, softmax_scale=..., causal=...)[0] of the pinned package, bit for bit ([B, T, H, D])."""
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    return _FA3.apply(q, k, v, float(softmax_scale), bool(causal))


# ------------------------------------------------------------------ MERGE B+C: the VARLEN pinned FA3 as custom ops too
# (BAL=1 packed segments / CP pieces). Same kernels and arguments as the pinned FlashAttnVarlenFunc (seqused None,
# num_splits 1, pack_gqa None, sm_margin 0, deterministic False); opaque to selective activation checkpointing.
@torch.library.custom_op("speedopt::fa3_varlen_fwd", mutates_args=())
def _fa3_varlen_fwd(q: Tensor, k: Tensor, v: Tensor, cu_q: Tensor, cu_k: Tensor, max_q: int, max_k: int, scale: float,
                    causal: bool) -> tuple[Tensor, Tensor]:
    fi = _mod().flash_attn_interface
    out, lse, *_ = fi._flash_attn_forward(q, k, v, None, None, None, None, cu_q, cu_k, None, None, None, max_q, max_k,
                                          None, None, None, None, None, None, None, None, None, scale, causal=causal,
                                          window_size=(-1, -1), softcap=0.0, num_splits=1, pack_gqa=None, sm_margin=0)
    return out, lse


@_fa3_varlen_fwd.register_fake
def _(q, k, v, cu_q, cu_k, max_q, max_k, scale, causal):
    N, H, D = q.shape
    return q.new_empty(N, H, v.shape[-1]), q.new_empty(H, N, dtype=torch.float32)


@torch.library.custom_op("speedopt::fa3_varlen_bwd", mutates_args=())
def _fa3_varlen_bwd(dout: Tensor, q: Tensor, k: Tensor, v: Tensor, out: Tensor, lse: Tensor, cu_q: Tensor, cu_k: Tensor,
                    max_q: int, max_k: int, scale: float, causal: bool) -> tuple[Tensor, Tensor, Tensor]:
    fi = _mod().flash_attn_interface
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    fi._flash_attn_backward(dout, q, k, v, out, lse, cu_q, cu_k, None, None, max_q, max_k, dq, dk, dv, scale, causal,
                            (-1, -1), 0.0, False, 0)
    return dq, dk, dv


@_fa3_varlen_bwd.register_fake
def _(dout, q, k, v, out, lse, cu_q, cu_k, max_q, max_k, scale, causal):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


class _FA3Varlen(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cu_q, cu_k, max_q, max_k, scale, causal):
        out, lse = torch.ops.speedopt.fa3_varlen_fwd(q, k, v, cu_q, cu_k, max_q, max_k, scale, causal)
        ctx.save_for_backward(q, k, v, out, lse, cu_q, cu_k)
        ctx.mq, ctx.mk, ctx.scale, ctx.causal = max_q, max_k, scale, causal
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse, cu_q, cu_k = ctx.saved_tensors
        dq, dk, dv = torch.ops.speedopt.fa3_varlen_bwd(dout, q, k, v, out, lse, cu_q, cu_k, ctx.mq, ctx.mk,
                                                       ctx.scale, ctx.causal)
        return dq, dk, dv, None, None, None, None, None, None


def fa3_varlen_func(q, k, v, cu_q, cu_k, max_q, max_k, softmax_scale, causal):
    """== flash_attn_varlen_func(q, k, v, cu_q, cu_k, max_q, max_k, softmax_scale=..., causal=...)[0], [N, H, D]."""
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    return _FA3Varlen.apply(q, k, v, cu_q, cu_k, int(max_q), int(max_k), float(softmax_scale), bool(causal))
