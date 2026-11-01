"""Additive conversation-token statistics from unique committed actor draws.

Probabilities must describe the distribution actually sampled, after any
partner sampling policy. These helpers neither reconstruct probabilities from
tokens nor count repeated teacher replays. Arrays have one slot per model
position; the physical conversation and sampled mask select unique draws.
"""
from __future__ import annotations

import math

import numpy as np


CATEGORIES = ('pad', 'new_word', 'word', 'other')
PUBLIC_CATEGORIES = CATEGORIES[:3]
ROLES = ('student', 'partner')
PAD_ID, NEW_WORD_ID, WORD_END, EXECUTABLE_VOCAB = 151670, 151671, 151643, 151682


def empty_conversation_metrics(role=None):
    """Stable additive keys, including on ranks holding only dummy rows."""
    if role is not None and role not in ROLES:
        raise ValueError(f'Unknown conversation role: {role}')
    result = {}
    for speaker in ROLES if role is None else (role,):
        prefix = f'_conversation/{speaker}'
        result[f'{prefix}/decisions'] = 0.
        for category in CATEGORIES:
            result[f'{prefix}/count/{category}'] = 0.
            result[f'{prefix}/probability_sum/{category}'] = 0.
    return result


def row_conversation_metrics(text, category_probabilities, *, prefix, prefill_frames,
                             frames, role, dummy=False, sampled_mask=None):
    """Sum one speaker's post-prefill text decisions, excluding physical tails.

    ``text`` and ``category_probabilities[:, 4]`` share model-position indexing,
    including the system prefix. Probability columns are PAD, NEW_WORD, WORD,
    OTHER, respectively. A sampled mask can exclude forced positions; it must
    describe unique committed draws, never replay invocations. Prefill/tail
    probability slots may be uninitialized because they are never inspected.
    """
    result = empty_conversation_metrics(role)
    for name, value in (('prefix', prefix), ('prefill_frames', prefill_frames), ('frames', frames)):
        if type(value) is not int or value < 0:
            raise ValueError(f'{name} must be a nonnegative integer')
    if frames < prefill_frames:
        raise ValueError('Conversation ends before its prefill')
    if dummy:
        return result
    tokens = np.asarray(text)
    probabilities = np.asarray(category_probabilities)
    if tokens.ndim != 1 or probabilities.shape != (len(tokens), len(CATEGORIES)):
        raise ValueError('Tokens and four-category probabilities must share model positions')
    begin, end = prefix + prefill_frames, prefix + frames
    if len(tokens) < end:
        raise ValueError('Conversation telemetry does not cover the complete physical row')
    selected = np.arange(begin, end)
    if sampled_mask is not None:
        mask = np.asarray(sampled_mask)
        if mask.shape != tokens.shape or mask.dtype != np.bool_:
            raise ValueError('Sampled mask must be boolean and match token positions')
        selected = selected[mask[selected]]
    tokens = tokens[selected]
    probabilities = probabilities[selected].astype(np.float64, copy=False)
    if not np.issubdtype(tokens.dtype, np.integer) or np.any(tokens < 0) or np.any(tokens >= EXECUTABLE_VOCAB):
        raise ValueError('Sampled text IDs must be executable integer tokens')
    if (not np.isfinite(probabilities).all() or np.any(probabilities < 0)
            or np.any(probabilities > 1) or not np.allclose(probabilities.sum(axis=1), 1., rtol=0, atol=2e-5)):
        raise ValueError('Sampled category probabilities must be finite probabilities summing to one')
    masks = (tokens == PAD_ID, tokens == NEW_WORD_ID, tokens < WORD_END)
    masks = (*masks, ~(masks[0] | masks[1] | masks[2]))
    stem = f'_conversation/{role}'
    result[f'{stem}/decisions'] = float(len(tokens))
    for index, (category, mask) in enumerate(zip(CATEGORIES, masks, strict=True)):
        result[f'{stem}/count/{category}'] = float(np.count_nonzero(mask))
        result[f'{stem}/probability_sum/{category}'] = float(probabilities[:, index].sum())
    return result


def finalize_conversation_metrics(metrics, reduced_mean_world_size=1):
    """Turn globally reduced additive statistics into counts and probabilities.

    Pass world size when the trainer's scalar all-reduce averages rank-local
    SUMS. Do not apply row/microbatch normalization before this function. With a
    global SUM reduction use the default one. Never average rank fractions.
    Only PAD, NEW_WORD and WORD are published. OTHER remains internal for mass
    validation, so the three displayed probabilities are not renormalized.
    Other metrics are preserved; internal additive keys are removed.
    """
    if type(reduced_mean_world_size) is not int or reduced_mean_world_size < 1:
        raise ValueError('Reduction world size must be a positive integer')
    result = {key: value for key, value in metrics.items() if not key.startswith('_conversation/')}
    for role in ROLES:
        stem = f'_conversation/{role}'
        required = empty_conversation_metrics(role)
        if not any(key in metrics for key in required):
            continue
        if not all(key in metrics for key in required):
            raise ValueError(f'Incomplete additive conversation metrics for {role}')
        raw = {key: float(metrics[key]) * reduced_mean_world_size for key in required}
        if any(not math.isfinite(value) or value < 0 for value in raw.values()):
            raise ValueError('Conversation statistics must be finite and nonnegative')
        count = raw[f'{stem}/decisions']
        counts = [raw[f'{stem}/count/{category}'] for category in CATEGORIES]
        sums = [raw[f'{stem}/probability_sum/{category}'] for category in CATEGORIES]
        if (not math.isclose(sum(counts), count, rel_tol=1e-10, abs_tol=1e-6)
                or not math.isclose(sum(sums), count, rel_tol=2e-5, abs_tol=1e-6)
                or any(abs(value - round(value)) > max(1e-6, value * 1e-10) for value in (count, *counts))):
            raise ValueError('Conversation categories must partition the actual sampled decisions')
        public = f'conversation/{role}'
        result[f'{public}/decisions'] = float(round(count))
        for category, sampled, probability_sum in zip(CATEGORIES, counts, sums, strict=True):
            if category not in PUBLIC_CATEGORIES:
                continue
            result[f'{public}/sampled_count/{category}'] = float(round(sampled))
            if count:
                result[f'{public}/sampled_fraction/{category}'] = sampled / count
                result[f'{public}/probability/{category}'] = probability_sum / count
    return result
