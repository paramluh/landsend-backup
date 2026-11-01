"""Forward-KL objectives for pose duplex distillation.

All targets are complete teacher distributions, with teachers detached. Dataset
labels only select valid positions and provide detached CE diagnostics. Text uses
production's category/lexical factorization: WORD is the aggregate of IDs below
151643 and every other vocabulary ID is a separate category. Its default loss
is KL(category) + p_teacher(WORD) * KL(lexical | WORD), exactly
KL(teacher || student) over the entire vocabulary. The optional lexical_weight
permits a weighted diagnostic/experiment; production distillation keeps it 1.

Position weights are *already normalized* (the production trainer uses
WORLD/global_step_token_count); supplied weights are never renormalized locally.
The default is a local valid-token mean, useful for tests and standalone calls.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F

WORD_END = 151643


def _math_dtype(x: torch.Tensor) -> torch.dtype:
    return torch.float64 if x.dtype == torch.float64 else torch.float32


def _weights_and_targets(shape, device, dtype, targets, position_weights, mask):
    if targets is None:
        target = torch.zeros(shape, device=device, dtype=torch.long)
        valid = torch.ones(shape, device=device, dtype=torch.bool)
    else:
        target = targets.detach().to(device=device, dtype=torch.long)
        if target.shape != shape:
            raise ValueError(f"targets shape {target.shape} does not match {shape}")
        valid = target >= 0
    if mask is not None:
        valid = valid & torch.broadcast_to(mask.detach().to(device=device, dtype=torch.bool), shape)
    if position_weights is None:
        weight = valid.to(dtype) / valid.sum().clamp_min(1)
    else:
        supplied = position_weights.detach().to(device=device, dtype=dtype)
        # A decoder frame weight applies to all its 31 codebooks. The caller
        # includes the factor 1/31 when preserving the production token mean.
        if supplied.ndim == len(shape) - 1 and tuple(supplied.shape) == tuple(shape[:-1]):
            supplied = supplied.unsqueeze(-1)
        weight = torch.broadcast_to(supplied, shape) * valid.to(dtype)
    return target, valid, weight


def distribution_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor | None = None,
    position_weights: torch.Tensor | None = None,
    *,
    mask: torch.Tensor | None = None,
):
    """Full categorical KL for cb0 [N,V] or light codes [N,31,V].

    Returns ``(loss, detached_stats)``. No top-k truncation, sampling, temperature
    scaling, teacher gradient, or data-CE term is part of the objective. ``kl``
    and ``data_ce`` have shape logits.shape[:-1]; decoder ``*_by_codebook``
    values are unnormalized sums over frames, suitable for distributed reduction.
    """
    if student_logits.shape != teacher_logits.shape or student_logits.ndim < 2:
        raise ValueError("student and teacher logits must have equal [..., vocabulary] shapes")
    shape = student_logits.shape[:-1]
    dtype = _math_dtype(student_logits)
    target, valid, weight = _weights_and_targets(
        shape, student_logits.device, dtype, targets, position_weights, mask
    )
    student_logp = F.log_softmax(student_logits.to(dtype), dim=-1)
    teacher_logp = F.log_softmax(teacher_logits.detach().to(device=student_logits.device, dtype=dtype), dim=-1)
    teacher_p = teacher_logp.exp()
    # Zero-probability teacher terms contribute zero, including masked -inf logits.
    terms = torch.where(teacher_p > 0, teacher_p * (teacher_logp - student_logp), torch.zeros_like(teacher_p))
    kl = terms.sum(-1)
    loss = (torch.where(valid, kl, torch.zeros_like(kl)) * weight).sum()
    with torch.no_grad():
        safe_target = target.masked_fill(~valid, 0)
        ce = -student_logp.detach().gather(-1, safe_target.unsqueeze(-1)).squeeze(-1)
        if targets is None:
            ce.zero_()
        ce = ce.masked_fill(~valid, 0)
        kd = kl.detach().masked_fill(~valid, 0)
        stats = {
            "kl": kd, "data_ce": ce, "valid": valid, "count": valid.sum(),
            "kl_sum": kd.sum(), "data_ce_sum": ce.sum(),
            "weighted_kl": loss.detach(), "weighted_data_ce": (ce * weight).sum(),
        }
        if student_logits.ndim == 3:
            stats.update(kl_by_codebook=kd.sum(0), data_ce_by_codebook=ce.sum(0),
                         count_by_codebook=valid.sum(0))
    return loss, stats


# Returned chunked-text rows, all nondifferentiable. Keep raw sums separate from
# training weights so a distributed caller can all-reduce sums and denominators.
_TEXT_STATS = (
    "full_kl", "category_kl", "lexical_kl", "lexical_mass_kl", "kl",
    "data_ce", "data_word_ce_norm", "data_category_ce", "data_weighted_ce",
    "teacher_word_probability",
)


def _text_chunk(student_z, teacher_z, target, valid, word_end, lexical_weight, data_lexical_weight):
    """Loss/statistics plus the analytic d(weighted KL)/d(student logits)."""
    slp = student_z.log_softmax(-1)
    tlp = teacher_z.log_softmax(-1)
    sp, tp = slp.exp(), tlp.exp()
    slw = student_z[:, :word_end].log_softmax(-1)
    tlw = teacher_z[:, :word_end].log_softmax(-1)
    sq, tq = slw.exp(), tlw.exp()
    tmass = tp[:, :word_end].sum(-1)
    full = (tp * (tlp - slp)).sum(-1)
    lexical = (tq * (tlw - slw)).sum(-1)
    mass_lexical = tmass * lexical
    # Chain-rule identity avoids constructing a second category-vocabulary tensor.
    category = full - mass_lexical
    weighted = full + (lexical_weight - 1.0) * mass_lexical
    grad = sp - tp
    grad[:, :word_end].add_((lexical_weight - 1.0) * tmass[:, None] * (sq - tq))
    safe = target.masked_fill(~valid, 0)
    data_ce = -slp.gather(1, safe[:, None]).squeeze(1)
    isword = (safe < word_end) & valid
    data_lexical = -slw.gather(1, safe.clamp_max(word_end - 1)[:, None]).squeeze(1)
    data_lexical = data_lexical.masked_fill(~isword, 0)
    data_category = data_ce - data_lexical
    data_weighted = data_category + data_lexical_weight * data_lexical
    stats = torch.stack((full, category, lexical, mass_lexical, weighted,
                         data_ce, data_lexical, data_category, data_weighted, tmass))
    stats.masked_fill_(~valid[None, :], 0)
    return weighted, stats, grad


def _mm_accum(a, b, dtype):
    """Low-precision GEMM inputs with fp32 accumulation, as in the production head."""
    if a.is_cuda and a.dtype in (torch.float16, torch.bfloat16) and dtype == torch.float32:
        return torch.mm(a, b, out_dtype=torch.float32)
    return a.to(dtype) @ b.to(dtype)


class _ChunkedTextKL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sx, sw, sr, tx, tw, tr, targets, valid, weights, spec, word_end, lexical_weight, chunk_size, data_lexical_weight):
        # Like production ce_head, form gradients while full FSDP head weights
        # are available. Only [chunk,V] logits exist; no [sequence,V] is retained.
        need_x, need_w, need_r = ctx.needs_input_grad[:3]
        need_grad = need_x or need_w or need_r
        dtype = _math_dtype(sx)
        rows = sx.shape[0]
        stats = torch.empty((len(_TEXT_STATS), rows), device=sx.device, dtype=dtype)
        dx = torch.zeros_like(sx, dtype=dtype) if need_x else None
        dw = torch.zeros_like(sw, dtype=dtype) if need_w else None
        dr = torch.zeros_like(sr, dtype=dtype) if need_r else None
        loss = torch.zeros((), device=sx.device, dtype=torch.float64)
        spec_idx = torch.tensor(spec, device=sx.device, dtype=torch.long)
        swc, src = sw.to(sx.dtype), sr.to(sx.dtype)
        twc, trc = tw.detach().to(tx.dtype), tr.detach().to(tx.dtype)
        with torch.autocast(device_type=sx.device.type, enabled=False):
            for start in range(0, rows, chunk_size):
                stop = min(start + chunk_size, rows)
                xs, xt = sx[start:stop], tx[start:stop].detach()
                zs = xs @ swc.t()
                zt = xt @ twc.t()
                if spec:
                    zs[:, spec_idx] = xs @ src.t()
                    zt[:, spec_idx] = xt @ trc.t()
                value, stat, grad = _text_chunk(
                    zs.to(dtype), zt.to(device=sx.device, dtype=dtype), targets[start:stop],
                    valid[start:stop], word_end, lexical_weight, data_lexical_weight
                )
                weight = weights[start:stop]
                loss.add_((torch.where(valid[start:stop], value, torch.zeros_like(value)).double() * weight.double()).sum())
                stats[:, start:stop] = stat
                if need_grad:
                    grad = (grad * weight[:, None]).to(sx.dtype)
                    own = grad[:, spec_idx].clone()
                    grad[:, spec_idx] = 0
                    if need_x:
                        dx[start:stop] = _mm_accum(grad, swc, dtype)
                        if spec:
                            dx[start:stop].add_(_mm_accum(own, src, dtype))
                    if need_w:
                        dw.add_(_mm_accum(grad.t(), xs, dtype))
                    if need_r and spec:
                        dr.add_(_mm_accum(own.t(), xs, dtype))
        ctx.save_for_backward(dx, dw, dr)
        ctx.dtypes = (sx.dtype, sw.dtype, sr.dtype)
        ctx.mark_non_differentiable(stats)
        return loss.to(dtype), stats

    @staticmethod
    def backward(ctx, grad_loss, grad_stats):
        grads = tuple(None if g is None else (g * grad_loss.to(g.dtype)).to(dt)
                      for g, dt in zip(ctx.saved_tensors, ctx.dtypes))
        return (*grads, None, None, None, None, None, None, None, None, None, None, None)


def factorized_text_kl_from_hidden(
    student_x: torch.Tensor,
    student_W: torch.Tensor,
    student_R: torch.Tensor,
    teacher_x: torch.Tensor,
    teacher_W: torch.Tensor,
    teacher_R: torch.Tensor,
    targets: torch.Tensor,
    position_weights: torch.Tensor | None,
    spec: Sequence[int],
    *,
    word_end: int = WORD_END,
    lexical_weight: float = 1.0,
    data_lexical_weight: float = 5.0,
    chunk_size: int = 32,
    mask: torch.Tensor | None = None,
):
    """Memory-bounded production text-head KL; returns ``(loss, stats_dict)``.

    ``x`` is the *projected* selected trunk hidden, after head.norm/head.proj.
    ``R`` is cat([spec_out,nw_out,eob_out]); these replace the corresponding
    lm_head rows given by ``spec``. Teacher/student can have different hidden
    widths, but their positions and vocabularies must align. Supply full local
    unsharded head tensors inside the relevant FSDP forward hook.

    This first-order custom autograd follows production's bf16 GEMM/fp32 softmax
    convention. Peak logits memory scales with chunk_size, not sequence length.
    ``data_lexical_weight`` independently preserves production CE diagnostics;
    use lexical_weight=1 for pure full-vocabulary KL while keeping this at 5.
    Gradients for replaced lm_head rows are exactly zero. Gradients of teacher
    tensors, position weights, targets and all diagnostics are always absent.
    """
    spec = tuple(int(i) for i in spec)
    rows, vocab = student_x.shape[0], student_W.shape[0]
    if student_x.ndim != 2 or teacher_x.ndim != 2 or teacher_x.shape[0] != rows:
        raise ValueError("teacher and student hidden states must have aligned [positions,hidden] shapes")
    if teacher_W.shape[0] != vocab or not 0 < word_end < vocab:
        raise ValueError("teacher/student vocabularies must agree and contain lexical and special IDs")
    if chunk_size <= 0 or lexical_weight < 0 or data_lexical_weight < 0:
        raise ValueError("chunk_size must be positive and lexical weights nonnegative")
    if len(set(spec)) != len(spec) or any(i < word_end or i >= vocab for i in spec):
        raise ValueError("replacement rows must be distinct special-token IDs")
    for x, w, r in ((student_x, student_W, student_R), (teacher_x, teacher_W, teacher_R)):
        if w.ndim != 2 or w.shape[1] != x.shape[1] or r.shape != (len(spec), x.shape[1]):
            raise ValueError("head weights/replacement rows do not match hidden width")
        if x.device != student_x.device or w.device != x.device or r.device != x.device:
            raise ValueError("all text-head tensors must be on the same device")
    target, valid, weight = _weights_and_targets(
        (rows,), student_x.device, _math_dtype(student_x), targets, position_weights, mask
    )
    if not torch.is_grad_enabled():
        # autograd.Function.needs_input_grad still reflects requires_grad on
        # input parameters under no_grad. Avoid forming full head gradients
        # during evaluation, matching production ce_head's separate eval path.
        student_x, student_W, student_R = (t.detach() for t in (student_x, student_W, student_R))
    loss, packed = _ChunkedTextKL.apply(
        student_x, student_W, student_R, teacher_x.detach(), teacher_W.detach(), teacher_R.detach(),
        target, valid, weight, spec, int(word_end), float(lexical_weight), int(chunk_size), float(data_lexical_weight)
    )
    stats = dict(zip(_TEXT_STATS, packed.unbind(0)))
    stats.update(valid=valid, count=valid.sum(), targets=target, kl_sum=stats['kl'].sum(),
                 full_kl_sum=stats['full_kl'].sum(), data_ce_sum=stats['data_ce'].sum(),
                 data_weighted_ce_sum=stats['data_weighted_ce'].sum(), weighted_kl=loss.detach(),
                 weighted_data_ce=(stats['data_weighted_ce'] * weight).sum())
    return loss, stats


def paired_metric_namespaces(
    kl_metrics: Mapping[str, torch.Tensor],
    data_metrics: Mapping[str, torch.Tensor],
    audio_kl: torch.Tensor | None = None,
    audio_ce: torch.Tensor | None = None,
    pose_kl: torch.Tensor | None = None,
    pose_ce: torch.Tensor | None = None,
):
    """Keep familiar dashboard names, with CE isolated under data_* prefixes.

    Main metric names are supplied without a prefix, e.g. ``word_ce_norm`` or
    ``loss``. Their existing ``*_ce`` suffix is retained for dashboard continuity
    even though ``train/*`` now measures KL. Codebook vectors must contain means
    for all 32 codebooks, including cb0; this helper does not average rank means.
    Every returned tensor is detached, including dataset CE diagnostics.
    """
    if set(kl_metrics) != set(data_metrics):
        raise ValueError("KL and data-CE dashboard metrics must have matching keys")
    result = {f'train/{key}': value.detach() for key, value in kl_metrics.items()}
    result.update({f'data_loss/{key}': value.detach() for key, value in data_metrics.items()})
    for name, kl, ce in (('audio', audio_kl, audio_ce), ('pose', pose_kl, pose_ce)):
        if kl is None and ce is None:
            continue
        if kl is None or ce is None or kl.shape != (32,) or ce.shape != (32,):
            raise ValueError(f'{name} codebook metrics require all 32 KL and CE means')
        result.update({f'{name}_codebooks/cb{i}': kl[i].detach() for i in range(32)})
        result.update({f'data_{name}_codebooks/cb{i}': ce[i].detach() for i in range(32)})
    return result
