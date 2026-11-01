"""Joint nine-reference stabilizer/resolver DMD layouts.

The stabilizer generates six detached temporal prefix frames and four
differentiable targets. All ten begin as noise and traverse the causal four-band
wavefront; only the provided sink and reference images are fixed visual inputs.
The resolver retains the existing GT-context, one-step t=.2 -> 0 objective.
"""

from contextlib import nullcontext
import math

import torch

from distillation.bimodal_dmd import H, P, path
from distillation.resolver_dmd import gt_noised_input, one_step_resolve, resolve_gt_pass, target_start

HISTORY_T = .2


def _references(net, rows):
    return net.references(rows.get('refs'), rows.get('ref_keep'))


def _stabilizer_rows(rows, num_targets=P, prefix_frames=H):
    """Select sink + prefix poses + targets, never their GT image values."""
    start = rows['latents'].shape[2] - num_targets
    if start < 1 + prefix_frames:
        raise ValueError(f'Stabilizer rows require sink, {prefix_frames} preceding poses, and targets')
    indices = [0, *range(start-prefix_frames, rows['latents'].shape[2])]
    selected = dict(rows)
    selected['latents'] = rows['latents'][:, :, indices]
    selected['pose'] = rows['pose'][:, :, indices]
    selected['ofi'] = rows['ofi'][:, indices]
    selected['history_t'] = HISTORY_T
    return selected


def generate_stabilizer(net, rows, band_steps=(2, 2, 2, 2), *, noise=None,
                        num_targets=P, history_perturbation=None, prefix_frames=6,
                        history_dropout=(), history_keep=None, static_layout=True, timed=None,
                        return_prefix=False):
    """Return (target endpoints, final three detached generated prefix frames).

    Default output is [B,C,4,h,w]; supplied noise is [B,C,prefix_frames+N,h,w].
    Every generated frame starts at t=1 and receives sum(band_steps) Euler
    updates, ending at .2. Prefix frames are evaluated under no_grad, including
    their cached present/history K/V; target trajectories and target-to-target
    conditioning remain differentiable. No GT image after the sink is read.

    With the default six-frame prefix, target 7 enters the highest-noise slot
    only once frames 1..3 fill the history and frames 4..6 fill the earlier
    present slots. Those last three prefix frames finish their staggered
    denoising concurrently, still without gradients. This preserves the causal
    wavefront, rather than completing all prefix endpoints and renoising them.
    The critic pool explicitly requests prefix_frames=3 (seven total frames),
    preserving the requested shorter startup distribution; student/eval use six.

    history_dropout gives independent probabilities nearest-prefix-first,
    relative to the first target. One mask stays fixed across the rollout and
    scoring. history_keep optionally supplies bool [B,6], oldest-first, over
    the six canonical prefix identities; the short critic uses its last three.
    Bootstrap stays undropped. Masked prefix conditioning is recomputed from
    pre-update states, so dropped inputs cannot leak through retained prefix
    present K/V. All new targets stay valid, including as later conditioning.
    Empty dropout (the default for evaluation) keeps every prefix frame.

    Evaluation may request N>4 using rows with enough pose/audio positions. An
    optional [B,C,3,h,w] history perturbation affects the LAST three prefix
    frames only, at their first history commitment, and persists in all later
    uses of those caches. It does not change their earlier present-slot states.

    Optional timed(name) contexts observe detached prefix work and sink/pad
    prefills as "context", and target/history/output assembly as "forward". These
    stages interleave; neither hook changes model calls or tensor operations.
    return_prefix additionally exposes all generated startup endpoints in the
    returned context, for complete evaluation videos; it adds no model calls.
    """
    timing = timed if timed is not None else lambda _: nullcontext()
    if (len(band_steps) != P or len(set(band_steps)) != 1 or
            any(type(steps) is not int or steps < 1 for steps in band_steps)):
        raise ValueError('Joint wavefront requires four equally subdivided bands')
    if type(num_targets) is not int or num_targets < 1:
        raise ValueError('num_targets must be a positive integer')
    if type(prefix_frames) is not int or prefix_frames not in (3, 6):
        raise ValueError('prefix_frames must be three (startup variant) or six (full window)')
    rows = _stabilizer_rows(rows, num_targets, prefix_frames)
    shape, pose, ofi = rows['latents'], rows['pose'], rows['ofi']
    B, C, _, height, width = shape.shape
    total = prefix_frames + num_targets
    if noise is None:
        noise = torch.randn((B, C, total, height, width), device=shape.device, dtype=torch.float32)
    if noise.shape != (B, C, total, height, width):
        raise ValueError('Generation noise must cover exactly the configured prefix and all targets')
    if history_perturbation is not None and history_perturbation.shape != (B, C, H, height, width):
        raise ValueError('History perturbation must have shape [B,C,3,h,w]')
    probabilities = tuple(float(value) for value in history_dropout)
    if len(probabilities) > 6 or any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities):
        raise ValueError('History dropout requires at most six probabilities in [0,1], nearest first')
    if history_keep is not None:
        if history_keep.shape != (B, 6) or history_keep.dtype != torch.bool:
            raise ValueError('history_keep must be bool [B,6], canonical oldest-first')
        canonical_keep = history_keep.to(device=shape.device).detach()
    else:
        rates = torch.zeros(6, dtype=torch.float32, device=shape.device)
        for distance, probability in enumerate(probabilities):
            rates[5-distance] = probability
        canonical_keep = (torch.rand((B, 6), device=shape.device) >= rates
                          if any(probabilities) else torch.ones((B, 6), dtype=torch.bool, device=shape.device))
    prefix_keep = canonical_keep[:, -prefix_frames:]
    masked_conditioning = history_keep is not None or any(probabilities)
    M, band_width = band_steps[0], (1-HISTORY_T)/P
    layers = len(net.transformer.blocks)
    history, masked_history, latent, completed, prefix_values = {}, {}, {}, {}, {}

    with _references(net, rows):
        tokens = net.audio_tokens(rows['audio'])

        def context_frame(value, frame_pose, time, slot, valid, sink=None):
            specs = [[('f', 0)]] if sink is None else [[('e', 0, 0), ('f', 0)]]
            ext = None if sink is None else [[k, v] for k, v in sink]
            _, ks, vs = net.forward_pass([(value, frame_pose, time, slot, valid)],
                                         [(0, specs)], ext, None, None, 0, keep=(0,))
            return [(k[:, 0], v[:, 0]) for k, v in zip(ks, vs)]

        with timing('context'):
            sink = context_frame(shape[:, :, 0], None, 0., 0, True)
            pad = context_frame(torch.zeros_like(shape[:, :, 0]), None, 0., H, False, sink)
        # At r=-2, frame 1 enters slot 3 at t=1. Startup is represented by
        # actual pad tokens, never substituted GT frames or extra generation.
        for r in range(1-(P-1), total+1):
            entering = r+P-1
            if 1 <= entering <= total:
                latent[entering] = noise[:, :, entering-1]
            leaving = r-1
            if 1 <= leaving <= total:
                completed[leaving] = latent.pop(leaving)
                value = completed[leaving]
                if leaving <= prefix_frames:
                    value = value.detach()
                    history_index = leaving - (prefix_frames-H) - 1
                    if history_perturbation is not None and 0 <= history_index < H:
                        value = value + history_perturbation[:, :, history_index]
                    prefix_values[leaving] = value.detach()
                with (torch.no_grad() if leaving <= prefix_frames else nullcontext()), \
                        timing('context' if leaving <= prefix_frames else 'forward'):
                    history[leaving] = context_frame(value, pose[:, :, leaving], HISTORY_T, H, True, sink)
                    if leaving <= prefix_frames:
                        # History queries attend refs + sink + self only. The
                        # invalid representation is exactly the t=0 pad prefill;
                        # selecting its K/V suppresses latent AND pose at every
                        # layer, without changing undropped bootstrap history.
                        if masked_conditioning:
                            valid = prefix_keep[:, leaving-1, None, None, None]
                            masked_history[leaving] = [
                                (torch.where(valid, key, pad_key.detach()),
                                 torch.where(valid, value, pad_value.detach()))
                                for (key, value), (pad_key, pad_value) in zip(history[leaving], pad)
                            ]
                        else:
                            masked_history[leaving] = history[leaving]

            active = [j for j in range(P) if 1 <= r+j <= total]
            prefix_slots = [j for j in active if r+j <= prefix_frames]
            target_slots = [j for j in active if r+j > prefix_frames]
            for substep in range(M):
                current_prefix = {}

                def trimmed_pass(slots, values, masked, retain):
                    ext, frames, specs, validity = [[] for _ in range(layers)], [], [], []

                    def external(kv, slot):
                        index = len(ext[0])//2
                        for layer, (key, value) in enumerate(kv):
                            ext[layer] += [key, value]
                        return ('e', index, slot)

                    sink_ref = external(sink, 0)
                    local = {r+j: index for index, j in enumerate(slots)}
                    for index, j in enumerate(slots):
                        frame = r+j
                        time = HISTORY_T+band_width*(j+1)-band_width*substep/M
                        valid = (prefix_keep[:, frame-1] if masked and frame <= prefix_frames else
                                 torch.ones(B, dtype=torch.bool, device=shape.device))
                        frames.append((values[frame], pose[:, :, frame], time*valid.float(), H+1+j, valid))
                        validity.append(valid)
                        left = []
                        for window_slot in range(1+j, 1+H+j):
                            previous = r-H-1+window_slot
                            if previous <= 0:
                                left.append(external(pad, window_slot))
                            elif window_slot <= H:
                                source = (masked_history[previous]
                                          if masked and previous <= prefix_frames else history[previous])
                                left.append(external(source, window_slot))
                            elif previous in local:
                                left.append(('f', local[previous]))
                            else:
                                left.append(external(current_prefix[previous], window_slot))
                        specs.append([sink_ref, *left, ('f', index)])
                    audio = net.frame_audio(tokens, ofi[:, [r+j for j in slots]], rows['drop'])
                    mask = torch.stack(validity, dim=1)
                    return net.forward_pass(frames, [(0, specs)], ext, audio, mask, len(slots),
                                            keep=tuple(range(len(slots))) if retain else ())

                def static_pass(slots, values, masked, retain):
                    # Four fixed present queries and eight fixed external slots:
                    # sink, three histories, and four present fallbacks. Tensor
                    # selectors choose local versus detached prefix/pad K/V;
                    # startup, tail, and prefix/target boundaries share a graph.
                    external = [sink]
                    for window_slot in range(1, H+1):
                        previous = r-H-1+window_slot
                        source = pad if previous <= 0 else (
                            masked_history[previous] if masked and previous <= prefix_frames
                            else history[previous])
                        external.append(source)
                    external += [current_prefix.get(r+j, pad) for j in range(P)]
                    # The checkpoint wrapper packs these shared tensors before
                    # compiled entry, avoiding alias guards without retaining
                    # fresh packed copies for every target pass and layer.
                    ext = [[tensor for source in external for tensor in source[layer]]
                           for layer in range(layers)]
                    frames, validity = [], []
                    for j in range(P):
                        frame = r+j
                        active_slot = j in slots
                        valid = (prefix_keep[:, frame-1] if active_slot and masked and frame <= prefix_frames
                                 else torch.full((B,), active_slot, dtype=torch.bool, device=shape.device))
                        time = HISTORY_T+band_width*(j+1)-band_width*substep/M
                        frames.append((values[frame] if active_slot else torch.zeros_like(shape[:, :, 0]),
                                       pose[:, :, frame] if active_slot else None,
                                       time*valid.float(), H+1+j, valid))
                        validity.append(valid)
                    specs = []
                    for j in range(P):
                        left = [('p', slot, slot) if slot <= H else ('m', slot-H-1, slot, slot)
                                for slot in range(1+j, 1+H+j)]
                        specs.append([('p', 0, 0), *left, ('f', j)])
                    audio = net.frame_audio(tokens, ofi[:, [min(max(r+j, 0), total) for j in range(P)]], rows['drop'])
                    mask = torch.stack(validity, dim=1)
                    local = torch.tensor([j in slots for j in range(P)], device=shape.device)[None].expand(B, -1)
                    velocity, keys, values_out = net.forward_pass(
                        frames, [(0, specs)], ext, audio, mask, P,
                        keep=tuple(range(P)) if retain else (), mix_mask=local)
                    return (velocity[:, :, slots], [key[:, slots] if key is not None else None for key in keys],
                            [value[:, slots] if value is not None else None for value in values_out])

                pass_slots = static_pass if static_layout else trimmed_pass

                if prefix_slots:
                    # Save the states BEFORE the undropped Euler update. Target
                    # queries in this substep must read those same-time states.
                    before = {r+j: latent[r+j] for j in prefix_slots}
                    with torch.no_grad(), timing('context'):
                        velocity, ks, vs = pass_slots(prefix_slots, before, False, True)
                        for index, j in enumerate(prefix_slots):
                            latent[r+j] = before[r+j] - (band_width/M)*velocity[:, :, index]
                        if target_slots and masked_conditioning:
                            # A separate masked view prevents the raw value of
                            # a dropped prefix from leaking via a kept prefix's
                            # deeper hidden states. These K/V never feed back
                            # into bootstrap generation or its history cache.
                            _, ks, vs = pass_slots(prefix_slots, before, True, True)
                        current_prefix = {
                            r+j: [(key[:, index], value[:, index]) for key, value in zip(ks, vs)]
                            for index, j in enumerate(prefix_slots)
                        }
                if target_slots:
                    with timing('forward'):
                        values = {r+j: latent[r+j] for j in target_slots}
                        velocity, _, _ = pass_slots(target_slots, values, True, False)
                        for index, j in enumerate(target_slots):
                            latent[r+j] = values[r+j] - (band_width/M)*velocity[:, :, index]
            for old in [frame for frame in history if frame < r-H]:
                del history[old]
                masked_history.pop(old, None)
        completed[total] = latent.pop(total)
    with timing('forward'):
        y = torch.stack([completed[frame] for frame in range(prefix_frames+1, total+1)], dim=2)
        context = {'history': torch.stack([prefix_values[frame]
                                          for frame in range(prefix_frames-H+1, prefix_frames+1)], dim=2),
                   'keep': canonical_keep[:, -H:], 'prefix_keep': canonical_keep}
        if return_prefix:
            context['prefix'] = torch.stack([prefix_values[frame]
                                              for frame in range(1, prefix_frames+1)], dim=2)
        return y, context

def _score_stabilizer_pass(net, rows, x, t, history, keep, audio, audio_mask):
    """Legacy stabilizer geometry with an explicit sticky prefix validity mask."""
    pose = rows['pose']
    frames = [(x[:, :, j], pose[:, :, H+1+j], t[:, j], H+1+j, True) for j in range(P)]
    frames += [(history[:, :, index], pose[:, :, index+1], HISTORY_T*keep[:, index].float(),
                H, keep[:, index]) for index in range(H)]
    sink = P+H
    frames.append((rows['latents'][:, :, 0], None, 0., 0, True))
    present = []
    for j in range(P):
        left = [('r', P+w-1, w) if w <= H else ('f', w-H-1)
                for w in range(1+j, 1+H+j)]
        present.append([('f', sink), *left, ('f', j)])
    histories = [[('f', sink), ('f', P+index)] for index in range(H)]
    plan = [(0, present), (P, histories), (sink, [[('f', sink)]])]
    velocity, _, _ = net.forward_pass(frames, plan, None, audio, audio_mask, P)
    return velocity


def score_stabilizer(net, rows, x, t, context, guidance_scale=1.):
    """Four-target scoring; audio CFG applies only to highest-noise slot 3.

    The null pass drops audio for the entire window, preserving pose, references,
    sink, history, and latent draws. Slots 0..2 retain conditional velocities.
    """
    if not math.isfinite(guidance_scale) or guidance_scale < 0:
        raise ValueError('Guidance scale must be finite and nonnegative')
    rows = _stabilizer_rows(rows)
    if x.shape != rows['latents'][:, :, -P:].shape or t.shape != (x.shape[0], P):
        raise ValueError('Stabilizer scoring requires four target frames and four times')
    history = context['history'].detach()
    if history.shape != rows['latents'][:, :, 1:1+H].shape:
        raise ValueError('Stabilizer scoring requires three generated history endpoints')
    keep = context.get('keep')
    if keep is None:
        keep = torch.ones((x.shape[0], H), dtype=torch.bool, device=x.device)
    if keep.shape != (x.shape[0], H) or keep.dtype != torch.bool:
        raise ValueError('Stabilizer context keep must be bool [B,3]')
    with _references(net, rows):
        tokens = net.audio_tokens(rows['audio'])
        audio = net.frame_audio(tokens, rows['ofi'][:, -P:], rows['drop'])
        mask = torch.ones_like(t, dtype=torch.bool)
        conditioned = _score_stabilizer_pass(net, rows, x, t, history, keep, audio, mask)
        if guidance_scale == 1.:
            return conditioned
        null_audio = net.frame_audio(tokens, rows['ofi'][:, -P:], torch.ones_like(rows['drop']))
        null = _score_stabilizer_pass(net, rows, x, t, history, keep, null_audio, mask)
        first = null[:, :, -1:] + guidance_scale*(conditioned[:, :, -1:] - null[:, :, -1:])
        return torch.cat((conditioned[:, :, :-1], first), dim=2)


def generate_resolver(net, rows, *, noise=None, timed=None):
    """Existing method-1 resolver, with optional input/forward timing contexts."""
    timing = timed if timed is not None else lambda _: nullcontext()
    with timing('input_generation'):
        target = rows['latents'][:, :, target_start(rows):]
        if noise is None:
            noise = torch.randn_like(target)
        x02 = gt_noised_input(rows, noise, t_start=HISTORY_T)
    with timing('forward'), _references(net, rows):
        return one_step_resolve(net, rows, x02, t_start=HISTORY_T)


def score_resolver(net, rows, x, t):
    """Score each resolver target against its original GT history, without CFG."""
    with _references(net, rows):
        tokens = net.audio_tokens(rows['audio'])
        audio = net.frame_audio(tokens, rows['ofi'][:, target_start(rows):], rows['drop'])
        return resolve_gt_pass(net, rows, x, t, audio, torch.ones_like(t, dtype=torch.bool))


def endpoint_fm_loss(velocity, y, times, eps, history_t=HISTORY_T):
    """Finite endpoint-prediction FM; returns mean and differentiable [B,F] losses.

    This is k^2 times velocity FM, k=(1-t0)*s^2/t. Algebraically forming
    k*(v-v*) avoids the singular s' noise target as t approaches t0.
    """
    with torch.no_grad():
        t = times.float()
        a, s, _, _ = path(history_t, t)
        k = (1-history_t)*s.square()/t
        y_scale = s.square()/t
        eps_scale = s*((1-history_t)*t+history_t*history_t*a)/t
        expand = lambda value: value[:, None, :, None, None]
        target = expand(y_scale)*y.detach().float() - expand(eps_scale)*eps.detach().float()
    residual = expand(k)*velocity.float() + target
    frames = residual.square().mean(dim=(1, 3, 4))
    return frames.mean(), frames
