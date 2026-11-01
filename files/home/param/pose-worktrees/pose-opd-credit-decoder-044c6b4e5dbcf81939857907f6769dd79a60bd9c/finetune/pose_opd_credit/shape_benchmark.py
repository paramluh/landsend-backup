"""Synthetic maximum-shape stress inputs; never on-policy training examples."""
from __future__ import annotations

import hashlib
import json

import numpy as np

from .rollout import Rollout


def worst_shape_plan(specs, *, world, rows_per_rank):
    if not specs:
        raise ValueError('Shape benchmark requires eligible real rows')
    chosen = max(specs, key=lambda value: (value['total_frames'], value['conversation_frames'], value['prefix_frames']))
    bounds = {'conversation_frames': max(value['conversation_frames'] for value in specs),
              'prefix_frames': max(value['prefix_frames'] for value in specs)}
    if any(chosen[key] != value for key, value in bounds.items()):
        raise ValueError(f'No single eligible row reaches both maximum conversation and prefix lengths: selected={chosen}, bounds={bounds}')
    updates = [[list(chosen['item']) for _ in range(world * rows_per_rank)]]
    digest = hashlib.sha256(json.dumps({'updates': updates, 'shape': chosen}, sort_keys=True).encode()).hexdigest()
    return {'format': 'pose-opd-synthetic-shape-plan-v1', 'updates': updates, 'digest': digest,
            'rows_per_epoch': 1, 'epochs': 1, 'world': world, 'rows_per_rank': rows_per_rank,
            'effective_batch': world * rows_per_rank, 'synthetic_shape_only': True,
            'shape_spec': dict(chosen), 'independent_maxima': bounds}


def validate_shape_seed(seed, spec):
    actual = {'conversation_frames': int(seed['conversation_frames']),
              'prefix_frames': int(np.asarray(seed['P']).reshape(-1)[0]),
              'total_frames': int(np.asarray(seed['ids']).shape[1])}
    if any(actual[key] != int(spec[key]) for key in actual):
        raise ValueError(f'Loaded row does not reach requested maximum shape: actual={actual}, expected={spec}')
    return actual


def synthetic_rollout(seed):
    """Cycle only forced prefill B tokens; retain every conditioning input.

    Empty behavior probabilities deliberately prevent treating this as an
    actor sample. The entry point disables policy comparisons for this mode.
    """
    prefix = int(np.asarray(seed['P']).reshape(-1)[0])
    frames, prefill = int(seed['conversation_frames']), int(seed['prefill_frames'])
    if prefill < 1:
        raise ValueError('Synthetic shape stress requires a nonempty forced prefill')
    text = np.array(seed['ids'][1], copy=True)
    audio = np.array(seed['depth'][1], copy=True)
    pose = np.array(seed['pdepth'], copy=True)
    masks = {key: np.array(value, copy=True) for key, value in seed['sampled_masks'].items()}
    for stream, values, delay in (('text', text, 0), ('audio', audio, 2), ('pose', pose, 2)):
        known = values[prefix + delay:prefix + delay + prefill].copy()
        positions = np.flatnonzero(masks[stream])
        values[positions] = known[np.arange(len(positions)) % prefill]
    return Rollout(text, audio, pose, masks, {}, prefix, frames, prefill)
