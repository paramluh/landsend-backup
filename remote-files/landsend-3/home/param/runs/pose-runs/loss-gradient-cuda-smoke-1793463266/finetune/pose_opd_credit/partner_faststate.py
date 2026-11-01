"""Optional removal of state unused by the fixed natural sampling policy.

NaturalCachedFastPPM._text reads logits directly and never reads ring.counts;
start() rejects repetition/adaptive sampling overrides. For audio-only partners,
only hn_prev is needed at each committed-backbone rewind boundary. Ring state
can therefore be omitted while preserving every sampled token and probability.
"""
from __future__ import annotations

import torch


def disable_partner_text_history(engine):
    """Opt in before capture or between sessions, keeping current graph safety.

    A previously captured text graph may still do unused ring work; that is
    harmless. Recapturing removes it too. The eager forced-token path and block
    finalization immediately use the reduced state. This is one-way per actor.
    """
    if (engine.pose or not hasattr(engine, '_opd_track_text_history')
            or getattr(engine, '_opd_sampler_backend', None) != 'inverse_cdf'):
        raise ValueError('Fast partner state requires the natural audio-only inverse-CDF actor')
    engine._opd_track_text_history = False
    engine._opd_rewind_states = {name: value for name, value in engine._opd_rewind_states.items()
                                 if not name.startswith('ring.')}
    return {'text_history_tracked': False, 'rewind_state_tensors': list(engine._rewind_tensors())}


@torch.no_grad()
def finalize_natural_block(engine, hidden, tokens, start, stop, offsets):
    """Store all six causal hidden-state boundaries with one indexed copy.

    Tokens are already committed in laneB. Rewriting them at intermediate
    positions and updating unused ring counts has no downstream reader.
    Preserve final lane/tok_now/adapt/position state and every hn_prev snapshot.
    """
    if (engine.pose or getattr(engine, '_opd_track_text_history', True)
            or set(engine._rewind_tensors()) != {'hn_prev'}):
        raise ValueError('Vectorized finalization requires only hidden-state rewind storage')
    count, window = stop - start, engine._opd_rewind_window
    if count < 1 or count > window or hidden.shape[1] != count or len(offsets) != count:
        raise ValueError('Block must fit the existing hidden-state rewind ring')
    slots = (offsets + start + 1).remainder(window)
    engine._opd_rewind_states['hn_prev'].index_copy_(0, slots, hidden.transpose(0, 1))
    for frame in range(start + 1, stop + 1):
        engine._opd_rewind_frames[frame % window] = frame
    engine.set_f(stop - 1)
    engine._supply_text(tokens[:, -1])
    engine.hn_prev.copy_(hidden[:, -1])
    engine._opd_next_frame, engine._opd_pending_frame = stop, None
