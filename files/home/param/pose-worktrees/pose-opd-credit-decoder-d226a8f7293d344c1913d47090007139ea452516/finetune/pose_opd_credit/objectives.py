"""Full categorical KL and finite-window credit through the sampled policy.

This uses Amu's zero-valued first-order score-function surrogate, with a frame
window and the actual duplex sampling graph replacing its word window. Model
frame order is text, followed by independent audio and pose branches; codebooks
within each branch are autoregressive. All three streams feed subsequent model
frames. A full analytic KL exists *before* its categorical draw, so that draw is
not its own ancestor. In particular audio(t) is not an ancestor of pose(t).

The cutoff limits score-function credit, not ordinary differentiable replay
through the model. Prefix/history tokens are constants. The caller records
forced/prefilled actions with sampled_mask=False and excludes prefilling KL
positions with loss_mask=False, accounting for native text/media frame delays.
KL and log probabilities are categorical softmax probabilities at temperature
one; there is no probit link, sampled-KL approximation, baseline or PPO ratio.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch

TEXT, AUDIO, POSE = 0, 1, 2
WINDOW_MODES = ("previous_frames", "including_current")


@dataclass(frozen=True)
class DecisionLayout:
    """One coordinate per decision; arbitrary order and sparse frames are legal.

    ``rows`` separates conversations (also when frame numbers reset). ``frames``
    are native *model* frame positions, not physical media frame positions.
    Text has codebook -1; audio/pose have codebook 0..C-1. The row identifiers
    need not be consecutive. Repeated loss coordinates are allowed, permitting
    several auxiliary costs at one decision. Actions normally occur once.
    """

    rows: torch.Tensor
    frames: torch.Tensor
    streams: torch.Tensor
    codebooks: torch.Tensor

    def __len__(self):
        return self.frames.numel()

    def validate(self, size: int, device: torch.device):
        for name in ("rows", "frames", "streams", "codebooks"):
            value = getattr(self, name)
            if value.shape != (size,) or value.dtype != torch.int64:
                raise ValueError(f"layout {name} must be int64 [{size}]")
            if value.device != device:
                raise ValueError(f"layout {name} must be on {device}")
        if not bool(((self.streams >= TEXT) & (self.streams <= POSE)).all()):
            raise ValueError("streams must be TEXT=0, AUDIO=1 or POSE=2")
        if not bool(torch.where(self.streams == TEXT, self.codebooks == -1,
                                self.codebooks >= 0).all()):
            raise ValueError("text codebook must be -1; media codebooks must be nonnegative")


class CreditLoss(NamedTuple):
    loss: torch.Tensor
    direct_loss: torch.Tensor
    credit_surrogate: torch.Tensor
    credit_weights: torch.Tensor


def _math_dtype(tensor):
    return torch.float64 if tensor.dtype == torch.float64 else torch.float32


def categorical_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor):
    """Analytic KL(student || teacher), one value per [..., vocabulary] row.

    Both distributions use every categorical outcome. The teacher is detached,
    including when a caller accidentally supplies a gradient-bearing tensor.
    Precision is FP32 for low-precision logits and FP64 for FP64 inputs.
    """
    if student_logits.shape != teacher_logits.shape or student_logits.ndim < 2:
        raise ValueError("student and teacher logits must have equal [..., vocabulary] shapes")
    dtype = _math_dtype(student_logits)
    slp = student_logits.to(dtype).log_softmax(-1)
    tlp = teacher_logits.detach().to(device=student_logits.device, dtype=dtype).log_softmax(-1)
    probability = slp.exp()
    # Zero-probability outcomes contribute zero, including joint -inf masks.
    delta = torch.where(probability > 0, slp - tlp, torch.zeros_like(slp))
    return (probability * delta).sum(-1)


def sampled_log_probs(logits: torch.Tensor, sampled_ids: torch.Tensor):
    """Replay log pi(a|history) under the actual categorical sampling logits.

    Supply the same logits/support/temperature used for sampling. A top-k or
    guided sampler cannot be replayed using unmodified base-model logits.
    """
    if logits.ndim < 2 or sampled_ids.shape != logits.shape[:-1]:
        raise ValueError("sampled_ids must match logits' non-vocabulary dimensions")
    if sampled_ids.dtype != torch.int64 or sampled_ids.device != logits.device:
        raise ValueError("sampled_ids must be int64 on the logits device")
    return logits.to(_math_dtype(logits)).log_softmax(-1).gather(
        -1, sampled_ids.unsqueeze(-1)).squeeze(-1)


def _mask(value, size, device, name):
    if value is None:
        return torch.ones(size, dtype=torch.bool, device=device)
    if value.shape != (size,) or value.dtype != torch.bool or value.device != device:
        raise ValueError(f"{name} must be bool [{size}] on {device}")
    return value.detach()


def causal_credit_weights(
    costs: torch.Tensor,
    loss_layout: DecisionLayout,
    action_layout: DecisionLayout | None = None,
    *,
    lookback_frames: int = 8,
    window_mode: str = "previous_frames",
    loss_mask: torch.Tensor | None = None,
    sampled_mask: torch.Tensor | None = None,
):
    """Sum detached weighted KL costs for each sampled causal ancestor.

    ``costs`` already includes all modality/codebook/global-normalization
    weights; no local renormalization is performed. ``previous_frames`` credits
    t-L..t-1 and same-frame ancestors for a KL at t. ``including_current`` credits
    t-L+1..t-1 and same-frame ancestors (L frames total).

    Frame costs use prefix sums, and within-frame codebook costs use reverse
    cumulative sums. Memory is O(decisions + unique_frames * codebooks), never
    an action-by-loss matrix. Search bounds are computed per unique frame, not
    individually per codebook action. Different rows cannot exchange credit.
    """
    if costs.ndim != 1 or not costs.is_floating_point():
        raise ValueError("costs must be a floating-point vector")
    if type(lookback_frames) is not int or lookback_frames < 0:
        raise ValueError("lookback_frames must be a nonnegative integer")
    if window_mode not in WINDOW_MODES:
        raise ValueError(f"window_mode must be one of {WINDOW_MODES}")
    if window_mode == "including_current" and lookback_frames < 1:
        raise ValueError("including_current needs at least one frame")
    previous = lookback_frames - (window_mode == "including_current")
    action_layout = loss_layout if action_layout is None else action_layout
    nl, na = len(loss_layout), len(action_layout)
    device = costs.device
    loss_layout.validate(costs.numel(), device)
    action_layout.validate(na, device)
    valid = _mask(loss_mask, nl, device, "loss_mask")
    sampled = _mask(sampled_mask, na, device, "sampled_mask")
    if na == 0:
        return costs.detach().new_zeros((0,), dtype=torch.float64)
    if nl == 0:
        return costs.detach().new_zeros((na,), dtype=torch.float64)
    with torch.no_grad():
        values = torch.where(valid, costs.detach().double(), 0.)
        if not bool(torch.isfinite(values).all()):
            raise ValueError("valid KL costs must be finite")
        all_rows = torch.cat((loss_layout.rows, action_layout.rows))
        all_frames = torch.cat((loss_layout.frames, action_layout.frames))
        _, row_index = torch.unique(all_rows, sorted=True, return_inverse=True)
        first = int(all_frames.min())
        last = int(all_frames.max())
        # Leave enough space after the final frame that a future-window bound
        # cannot reach the next row's keys, even on arbitrarily sparse frames.
        span = last - first + previous + 2
        row_count = int(row_index.max()) + 1
        if span * row_count > torch.iinfo(torch.int64).max:
            raise ValueError("row/frame coordinate span exceeds int64 key range")
        keys = row_index * span + (all_frames - first)
        frame_keys, frame_index = torch.unique(keys, sorted=True, return_inverse=True)
        loss_frame, action_frame = frame_index[:nl], frame_index[nl:]
        frame_cost = values.new_zeros(len(frame_keys)).index_add_(0, loss_frame, values)
        prefix = torch.cat((values.new_zeros(1), frame_cost.cumsum(0)))
        end = torch.searchsorted(frame_keys, frame_keys + previous, right=True)
        past_credit = prefix[end] - prefix[torch.arange(len(frame_keys), device=device) + 1]

        max_book = max(int(loss_layout.codebooks.max()), int(action_layout.codebooks.max()), 0)
        books = max_book + 1
        # Text lives in slot zero of stream TEXT. Its same-frame KL contributes
        # to no action, because all current media is sampled after current text.
        flat = ((loss_frame * 3 + loss_layout.streams) * books
                + loss_layout.codebooks.clamp_min(0))
        grid = values.new_zeros(len(frame_keys) * 3 * books).index_add_(0, flat, values)
        grid = grid.view(len(frame_keys), 3, books)
        suffix = grid.flip(-1).cumsum(-1).flip(-1) - grid
        same = suffix[action_frame, action_layout.streams,
                      action_layout.codebooks.clamp_min(0)]
        media_cost = grid[:, AUDIO:, :].sum((1, 2))
        same = torch.where(action_layout.streams == TEXT, media_cost[action_frame], same)
        return (past_credit[action_frame] + same).masked_fill(~sampled, 0.)


def finite_credit_loss(
    kl: torch.Tensor,
    log_probs: torch.Tensor,
    loss_layout: DecisionLayout,
    action_layout: DecisionLayout | None = None,
    *,
    position_weights: torch.Tensor | None = None,
    loss_mask: torch.Tensor | None = None,
    sampled_mask: torch.Tensor | None = None,
    lookback_frames: int = 8,
    window_mode: str = "previous_frames",
) -> CreditLoss:
    """Direct analytic KL plus zero-valued finite-window score surrogate.

    Inputs are per-decision vectors, possibly with separate loss/action layouts.
    Position weights must already contain desired global denominators and loss
    coefficients. Omitting them means an unnormalized sum, not a hidden mean.
    ``credit_surrogate`` is numerically zero but differentiates through replayed
    sampled log probabilities. Detached costs avoid counting their derivative
    twice. This implements first-order gradients, not higher derivatives.
    """
    if kl.ndim != 1 or log_probs.ndim != 1 or kl.device != log_probs.device:
        raise ValueError("kl and log_probs must be vectors on the same device")
    action_layout = loss_layout if action_layout is None else action_layout
    if len(action_layout) != log_probs.numel():
        raise ValueError("action layout size must match log_probs")
    valid = _mask(loss_mask, kl.numel(), kl.device, "loss_mask")
    sampled = _mask(sampled_mask, log_probs.numel(), kl.device, "sampled_mask")
    if position_weights is None:
        weights = torch.ones_like(kl)
    else:
        if position_weights.shape != kl.shape or position_weights.device != kl.device:
            raise ValueError("position_weights must match kl shape and device")
        weights = position_weights.detach().to(kl.dtype)
        if not bool((torch.isfinite(weights) & (weights >= 0)).all()):
            raise ValueError("position_weights must be finite and nonnegative")
    weighted = torch.where(valid, kl, torch.zeros_like(kl)) * weights
    credit = causal_credit_weights(weighted, loss_layout, action_layout,
        lookback_frames=lookback_frames, window_mode=window_mode,
        loss_mask=valid, sampled_mask=sampled)
    safe_logp = torch.where(sampled, log_probs, torch.zeros_like(log_probs))
    if not bool(torch.isfinite(safe_logp).all()):
        raise ValueError("sampled log probabilities must be finite")
    direct = weighted.sum()
    surrogate = (credit.to(safe_logp.dtype) * (safe_logp - safe_logp.detach())).sum()
    return CreditLoss(direct + surrogate, direct, surrogate, credit)
