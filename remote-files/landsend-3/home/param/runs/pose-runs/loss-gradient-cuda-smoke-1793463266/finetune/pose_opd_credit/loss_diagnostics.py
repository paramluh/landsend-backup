"""Detached logit-gradient diagnostics, without another network pass or backward.

These are gradients with respect to each categorical prediction's logits, not
model parameters. Only three scalars per decision survive each scoring chunk.
Raw squared norms/dot products are additive across chunks, rows and ranks;
square roots, ratios and cosines are computed only after global reduction.
"""
from __future__ import annotations

import math

import torch

WORD_COUNT = 151643
MODALITIES = ('text', 'audio', 'pose')
STREAM_SLICES = (slice(0, 1), slice(1, 33), slice(33, 65))
STAT_NAMES = ('direct_sq', 'score_sq', 'direct_score_dot')


@torch.no_grad()
def categorical_statistics(logp, teacher_logp, probability, delta, kl, actions, *, word_count=None):
    """Reuse categorical arithmetic to summarize exact unit logit derivatives."""
    direct = probability * (delta - kl[..., None])
    selected_p = probability.gather(-1, actions[..., None]).squeeze(-1)
    selected_direct = direct.gather(-1, actions[..., None]).squeeze(-1)
    # Avoid cancellation in 1 - 2*p(a) + sum(p²), especially for confident PAD.
    score_squared = probability.square()
    score_squared.scatter_(-1, actions[..., None], (1 - selected_p).square()[..., None])
    score_sq = score_squared.sum(-1)
    del score_squared
    # Likewise sum the signed per-category products instead of subtracting two
    # nearly equal totals for <dKL/dz, dlogp(a)/dz>.
    product = direct * probability
    product.scatter_(-1, actions[..., None], (selected_direct * (selected_p - 1))[..., None])
    direct_score_dot = -product.sum(-1)
    del product
    statistics = {
        'direct_sq': direct.square().sum(-1),
        'score_sq': score_sq,
        'direct_score_dot': direct_score_dot,
    }
    del direct
    if word_count is not None:
        words = min(int(word_count), logp.shape[-1])
        if words < 1:
            raise ValueError('The lexical category must contain at least one ID')
        word_logp, word_logq = logp[..., :words], teacher_logp[..., :words]
        log_mass = word_logp.logsumexp(-1)
        teacher_log_mass = word_logq.logsumexp(-1)
        # Normalizing in log space keeps the lexical diagnostic meaningful even
        # when total word probability underflows in FP32.
        conditional_logp = word_logp - log_mass[..., None]
        conditional_logq = word_logq - teacher_log_mass[..., None]
        conditional_p = conditional_logp.exp()
        lexical = (conditional_p * torch.where(conditional_p > 0,
            conditional_logp - conditional_logq, torch.zeros_like(conditional_logp))).sum(-1)
        lexical = torch.where(torch.isfinite(log_mass), lexical, torch.zeros_like(lexical))
        mass = log_mass.exp()
        contribution = mass * lexical
        # Category = WORD as one outcome, plus every remaining executable ID
        # separately. This is a partition of the exact full distribution.
        word_category = torch.where(mass > 0, mass * (log_mass - teacher_log_mass), 0.)
        category = word_category + (probability[..., words:] * delta[..., words:]).sum(-1)
        statistics.update(lexical=lexical, category=category,
                          lexical_contribution=contribution, word_mass=mass)
    return statistics


@torch.no_grad()
def scored_categorical_statistics(student, teacher, actions, *, word_count=None):
    """Detached scoring replacement that computes KL/logp and stats together."""
    dtype = torch.float64 if student.dtype == torch.float64 else torch.float32
    sl = student.to(dtype).log_softmax(-1)
    tl = teacher.detach().to(device=student.device, dtype=dtype).log_softmax(-1)
    p = sl.exp()
    delta = torch.where(p > 0, sl - tl, torch.zeros_like(sl))
    kl = (p * delta).sum(-1)
    lp = sl.gather(-1, actions[..., None]).squeeze(-1)
    return kl, lp, categorical_statistics(sl, tl, p, delta, kl, actions, word_count=word_count)


@torch.no_grad()
def logit_gradient_sums(statistics, weights, credit, credit_by_origin, *, world=1):
    """Return local SUM statistics for the global, DDP-normalized objective.

    Replay weights are multiplied by world size to compensate DDP averaging.
    Logits are distinct between ranks, so remove that factor before measuring
    their concatenated derivative vector. Credit origins overlap: their squared
    norms must never be added to obtain the combined credit norm.
    """
    if type(world) is not int or world < 1:
        raise ValueError('world must be a positive integer')
    d2, r2, dr = (statistics[name].double() for name in STAT_NAMES)
    w, c = weights.double() / world, credit.double() / world
    direct_sq = w.square() * d2
    credit_sq = c.square() * r2
    direct_credit_dot = w * c * dr
    values = {}
    for name, columns in zip(MODALITIES, STREAM_SLICES):
        origin = credit_by_origin[name].double() / world
        dsq = direct_sq[:, columns].sum()
        csq = (origin.square() * r2).sum()
        dot = (w[:, columns] * origin[:, columns] * dr[:, columns]).sum()
        values[f'_logit_grad_sq/direct/{name}'] = dsq
        values[f'_logit_grad_sq/credit_from/{name}'] = csq
        values[f'_logit_grad_sq/credit_on/{name}'] = credit_sq[:, columns].sum()
        values[f'_logit_grad_dot/{name}'] = dot
        values[f'_logit_grad_sq/combined/{name}'] = dsq + csq + 2 * dot
    values['_logit_grad_sq/direct/all'] = direct_sq.sum()
    values['_logit_grad_sq/credit_from/all'] = credit_sq.sum()
    values['_logit_grad_dot/all'] = direct_credit_dot.sum()
    values['_logit_grad_sq/combined/all'] = (direct_sq + credit_sq + 2 * direct_credit_dot).sum()
    names = list(values)
    numbers = torch.stack([values[name] for name in names]).cpu().tolist()
    return dict(zip(names, numbers))


def finalize_loss_diagnostics(metrics, world):
    """Finalize after the trainer's existing rank MEAN of additive metrics.

    Multiply that mean by world to restore SUM squared norms/dot products.
    Ratios/cosines with a zero denominator are omitted, never fabricated as 0.
    Ordinary KL/loss metrics retain their normal rank-mean semantics.
    """
    if type(world) is not int or world < 1:
        raise ValueError('world must be a positive integer')
    result = {}
    for name, value in metrics.items():
        if name.startswith('_logit_grad_sq/'):
            result['logit_grad_norm/' + name.removeprefix('_logit_grad_sq/')] = math.sqrt(max(0., value * world))
        elif name.startswith('_logit_grad_dot/'):
            result['logit_grad_dot/' + name.removeprefix('_logit_grad_dot/')] = value * world
        else:
            result[name] = value
    for name in (*MODALITIES, 'all'):
        direct = result.get(f'logit_grad_norm/direct/{name}', 0.)
        credit = result.get(f'logit_grad_norm/credit_from/{name}', 0.)
        if direct > 0:
            result[f'logit_grad_credit_direct_ratio/{name}'] = credit / direct
            if credit > 0:
                dot = result[f'logit_grad_dot/{name}']
                result[f'logit_grad_alignment/{name}'] = max(-1., min(1., dot / (direct * credit)))
    return result
