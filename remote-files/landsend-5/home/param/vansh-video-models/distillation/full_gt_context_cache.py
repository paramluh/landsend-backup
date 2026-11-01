"""Differentiable fixed-history K/V for one full-GT stabilizer rollout.

The cache belongs to one model, reference scope, and retained context. It is
never returned as part of the model-independent training context or reused
after a parameter update. Keys stay unrotated until each query selects its
history identities and relative window positions.
"""

import torch


HISTORY_SLOTS = 3
HISTORY_IDENTITIES = 6


def prefill_stabilizer_context(net, context, history_t):
    """Compute six independent fixed histories and the sink, retaining autograd.

    Each history attends references, sink, and itself; the sink attends itself.
    No target, varying present frame, or audio enters this pass. The caller
    owns the reference scope and gradient mode, including block checkpointing.
    """
    frames = []
    for index in range(HISTORY_IDENTITIES):
        valid = context['keep'][:, index]
        time = valid.float() * history_t
        t = time[:, None, None, None]
        value = ((1 - t) * context['history'][:, :, index]
                 + t * context['noise'][:, :, index])
        value = torch.where(valid[:, None, None, None], value, torch.zeros_like(value))
        pose = context['pose'][:, :, index]
        pose = torch.where(valid[:, None, None, None], pose, torch.zeros_like(pose))
        frames.append((value, pose, time, HISTORY_SLOTS, valid))
    sink = HISTORY_IDENTITIES
    frames.append((context['sink'], None, 0., 0, True))
    plan = [(0, [[('f', sink), ('f', index)] for index in range(HISTORY_IDENTITIES)]),
            (sink, [[('f', sink)]])]
    _, keys, values = net.forward_pass(frames, plan, None, None, None, 0,
                                       keep=tuple(range(HISTORY_IDENTITIES + 1)))
    return tuple(zip(keys, values))


def select_stabilizer_context(cache, bands):
    """Select sink + three history identities per row; gathers retain gradients."""
    indices = (HISTORY_SLOTS - bands)[:, None] + torch.arange(HISTORY_SLOTS, device=bands.device)
    external = []
    for keys, values in cache:
        index = indices.reshape(*indices.shape, *([1] * (keys.ndim - 2)))
        index = index.expand(-1, -1, *keys.shape[2:])
        selected_keys = keys.gather(1, index)
        selected_values = values.gather(1, index)
        layer = [keys[:, HISTORY_IDENTITIES], values[:, HISTORY_IDENTITIES]]
        for offset in range(HISTORY_SLOTS):
            layer.extend((selected_keys[:, offset], selected_values[:, offset]))
        external.append(layer)
    return external


def cached_stabilizer_plan(present):
    """External raw keys receive the same relative RoPE as inline histories."""
    queries = []
    for slot in range(present):
        previous = [('e', window, window) if window <= HISTORY_SLOTS
                    else ('f', window - HISTORY_SLOTS - 1)
                    for window in range(1 + slot, 1 + HISTORY_SLOTS + slot)]
        queries.append([('e', 0, 0), *previous, ('f', slot)])
    return [(0, queries)]
