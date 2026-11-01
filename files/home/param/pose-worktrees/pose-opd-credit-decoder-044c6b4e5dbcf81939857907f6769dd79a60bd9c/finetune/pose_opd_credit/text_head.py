"""Bounded full-vocabulary reverse KL and exact sampled text score gradients.

Vocabulary-sized activations are bounded by the chunk size; no full
[conversation, vocabulary] activation is retained. The custom first-order
backward is evaluated while FSDP has the complete head weights. Teacher
probabilities and finite-window credit coefficients are constants. CUDA uses
low-precision GEMM operands with FP32 gradient accumulation; kernel rounding
can differ from eager autograd and is not claimed to be bit-identical.
"""
from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable


def _mm_accum(a, b, dtype):
    """Tensor-core input precision with FP32 accumulation on CUDA."""
    if a.is_cuda and a.dtype in (torch.float16, torch.bfloat16) and dtype == torch.float32:
        return torch.mm(a, b, out_dtype=torch.float32)
    return a.to(dtype) @ b.to(dtype)


def logits_objective(student, teacher, actions, direct_weights, credit_weights):
    """Dense reference and analytic d(loss)/d(student logits), also used in chunks."""
    dtype = torch.float64 if student.dtype == torch.float64 else torch.float32
    sl = student.to(dtype).log_softmax(-1)
    tl = teacher.detach().to(device=student.device, dtype=dtype).log_softmax(-1)
    p = sl.exp()
    delta = torch.where(p > 0, sl - tl, torch.zeros_like(sl))
    kl = (p * delta).sum(-1)
    logp = sl.gather(-1, actions[:, None]).squeeze(-1)
    dw = direct_weights.detach().to(dtype)
    cw = credit_weights.detach().to(dtype)
    loss = (dw * kl + cw * (logp - logp.detach())).sum()
    gradient = dw[:, None] * p * (delta - kl[:, None]) - cw[:, None] * p
    gradient.scatter_add_(1, actions[:, None], cw[:, None])
    return loss, kl, logp, gradient


class _ChunkedReverseKL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, special, tx, tw, tr, actions, dw, cw, spec, vocab, chunk, need_grad, logits_callback):
        tensors = (x, weight, special, tx, tw, tr)
        if any(t.ndim != 2 or not t.is_floating_point() for t in tensors):
            raise ValueError('Hidden states and head weights must be floating-point matrices')
        if any(t.device != x.device for t in (*tensors, actions, dw, cw)):
            raise ValueError('Text objective inputs must be on the same device')
        if x.shape[0] != actions.numel() or tx.shape[0] != x.shape[0]:
            raise ValueError('Text hidden states and actions must align')
        if actions.shape != (len(x),) or actions.dtype != torch.int64:
            raise ValueError('Actions must be an int64 vector')
        if dw.shape != actions.shape or cw.shape != actions.shape:
            raise ValueError('Direct and credit weights must align with actions')
        if not bool((torch.isfinite(dw) & (dw >= 0)).all()) or not bool(torch.isfinite(cw).all()):
            raise ValueError('Direct weights must be finite/nonnegative and credit weights finite')
        if weight.shape[1] != x.shape[1] or special.shape != (len(spec), x.shape[1]):
            raise ValueError('Student head/replacement weights do not match hidden width or special IDs')
        if tw.shape[1] != tx.shape[1] or tr.shape != (len(spec), tx.shape[1]):
            raise ValueError('Teacher head/replacement weights do not match hidden width or special IDs')
        if not 0 < vocab <= min(weight.shape[0], tw.shape[0]) or chunk < 1:
            raise ValueError('Invalid executable vocabulary/chunk size')
        if bool(((actions < 0) | (actions >= vocab)).any()):
            raise ValueError('Sampled text outside executable vocabulary')
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        index = torch.as_tensor(spec, device=x.device, dtype=torch.long)
        if bool(((index < 0) | (index >= vocab)).any()) or len(set(spec)) != len(spec):
            raise ValueError('Special replacement IDs must be unique and inside executable vocabulary')
        need_x, need_w, need_r = need_grad
        grad_enabled = any(need_grad)
        dx = torch.zeros_like(x, dtype=dtype) if need_x else None
        gw = torch.zeros_like(weight, dtype=dtype) if need_w else None
        gr = torch.zeros_like(special, dtype=dtype) if need_r else None
        kl = torch.empty(len(x), dtype=dtype, device=x.device)
        logp = torch.empty_like(kl)
        loss = torch.zeros((), dtype=dtype, device=x.device)
        wc, rc = weight[:vocab].to(x.dtype), special.to(x.dtype)
        tc, tsc = tw[:vocab].detach().to(tx.dtype), tr.detach().to(tx.dtype)
        with torch.autocast(device_type=x.device.type, enabled=False):
            for start in range(0, len(x), chunk):
                end = min(start + chunk, len(x))
                xs, xt = x[start:end], tx[start:end].detach()
                zs, zt = xs @ wc.t(), xt @ tc.t()
                zs[:, index] = xs @ rc.t()
                zt[:, index] = xt @ tsc.t()
                if logits_callback is not None:
                    logits_callback(start, zs)
                value, k, lp, dz = logits_objective(
                    zs.to(dtype), zt.to(dtype), actions[start:end], dw[start:end], cw[start:end])
                loss.add_(value)
                kl[start:end], logp[start:end] = k, lp
                if grad_enabled:
                    # Match the activation precision at the projection's
                    # backward boundary, while accumulating shard gradients
                    # in FP32. Avoid full FP32 copies of the large head weight.
                    dz = dz.to(xs.dtype)
                    own = dz[:, index].clone()
                    dz[:, index] = 0
                    # FP32 accumulation; return each gradient in the input's dtype.
                    if need_x:
                        dx[start:end] = _mm_accum(dz, wc, dtype) + _mm_accum(own, rc, dtype)
                    if need_w:
                        gw[:vocab].add_(_mm_accum(dz.t(), xs, dtype))
                    if need_r:
                        gr.add_(_mm_accum(own.t(), xs, dtype))
        ctx.save_for_backward(None if dx is None else dx.to(x.dtype),
                              None if gw is None else gw.to(weight.dtype),
                              None if gr is None else gr.to(special.dtype))
        ctx.mark_non_differentiable(kl, logp)
        return loss, kl, logp

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream, _kl, _logp):
        dx, dw, dr = ctx.saved_tensors
        return tuple(None if grad is None else grad * upstream for grad in (dx, dw, dr)) + (None,) * 11


def chunked_text_objective(x, weight, special, teacher_x, teacher_weight, teacher_special,
                           actions, direct_weights, credit_weights, *, special_ids,
                           executable_vocab, chunk_size=32, logits_callback=None):
    """Return (differentiable scalar, detached per-position KL, sampled logp).

    Masks are encoded by zero direct/credit weights; actions must still be
    valid executable IDs. Special replacement rows override their base-head
    rows; vocabulary padding outside executable_vocab has zero gradient.
    No-gradient scoring performs no gradient GEMMs or full head-grad allocation.
    """
    need_grad = tuple(torch.is_grad_enabled() and t.requires_grad for t in (x, weight, special))
    return _ChunkedReverseKL.apply(x, weight, special, teacher_x, teacher_weight,
                                  teacher_special, actions, direct_weights, credit_weights,
                                  tuple(special_ids), int(executable_vocab), int(chunk_size), need_grad, logits_callback)
