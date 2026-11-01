"""Single-target DMD with six ground-truth predecessors and nine references.

One row is [sink | six GT predecessors | one target], plus a separate reference
bank. Predecessors follow fixed, analytic GT-to-noise paths; their predicted
velocities never update their values. Only the target is integrated. A retained
context carries one noise draw and one validity decision per predecessor into
every student, teacher, and critic pass.
"""

import math

import torch

from distillation.full_gt_context_cache import (cached_stabilizer_plan,
    prefill_stabilizer_context, select_stabilizer_context)

H, P, NUM_REFERENCES = 3, 4, 9


def _plan(present):
    """Pretraining attention geometry with optional future-slot trimming."""
    sink = present + H
    queries = []
    for slot in range(present):
        previous = [('r', present + window - 1, window) if window <= H
                    else ('f', window - H - 1)
                    for window in range(1 + slot, 1 + H + slot)]
        queries.append([('f', sink), *previous, ('f', slot)])
    return [(0, queries),
            (present, [[('f', sink), ('f', present + index)] for index in range(H)]),
            (sink, [[('f', sink)]])]


def _gather_frames(value, indices):
    """Per-row temporal gather from [B,C,F,h,w], retaining the frame axis."""
    batch, channels, _, height, width = value.shape
    if indices.ndim == 1:
        indices = indices[:, None]
    return value.gather(2, indices[:, None, :, None, None].expand(
        batch, channels, -1, height, width))


def _validate_rows(rows):
    gt, pose, ofi = rows['latents'], rows['pose'], rows['ofi']
    if gt.ndim != 5 or gt.shape[2] != 8 or pose.shape != gt.shape:
        raise ValueError('Full-GT rows require matching latents/pose [B,C,8,h,w]: sink, six predecessors, target')
    if ofi.shape != (gt.shape[0], 8):
        raise ValueError('Full-GT frame indices must have shape [B,8]')
    refs, ref_keep = rows.get('refs'), rows.get('ref_keep')
    if (refs is None or refs.shape != (gt.shape[0], gt.shape[1], NUM_REFERENCES, *gt.shape[-2:])
            or ref_keep is None or ref_keep.shape != (gt.shape[0], NUM_REFERENCES)
            or ref_keep.dtype != torch.bool):
        raise ValueError('Full-GT training requires nine reference latents [B,C,9,h,w] and bool ref_keep [B,9]')
    if rows['drop'].shape != (gt.shape[0],):
        raise ValueError('Audio dropout must have shape [B]')
    return gt, pose, ofi


def _validate_context(rows, context):
    gt = rows['latents']
    expected = gt[:, :, 1:7].shape
    if any(context[key].shape != expected for key in ('history', 'noise', 'pose')):
        raise ValueError('Retained GT context requires history, noise, and pose [B,C,6,h,w]')
    if context['ofi'].shape != (gt.shape[0], 6):
        raise ValueError('Retained context frame indices must have shape [B,6]')
    if context['keep'].shape != context['ofi'].shape or context['keep'].dtype != torch.bool:
        raise ValueError('Retained context keep must be bool [B,6], oldest first')
    if context['sink'].shape != gt[:, :, 0].shape or context['target_pose'].shape != gt[:, :, -1].shape:
        raise ValueError('Retained sink and target pose must match the row geometry')
    if context['target_ofi'].shape != (gt.shape[0],) or context['drop'].shape != (gt.shape[0],):
        raise ValueError('Retained target indices and audio dropout must have shape [B]')


def _gt_at(value, noise, time):
    time = time[:, None, None, None]
    return (1 - time) * value + time * noise


class FullGTStabilizer:
    """Integrate one target from 1 to .2 while recomputing GT conditioning.

    The original full-GT dropout probabilities are independent, nearest first:
    [0, .1, .2, .2]; unlisted older predecessors are retained. An explicit
    rows['history_keep'] bool [B,6], oldest first, overrides random dropout.
    Source-invalid predecessors remain invalid. The selected mask is stored in
    context and is never resampled by generate() with context or score_pass().

    GT latent values are detached. Fixed t=.2 history/sink representations are
    cached once per rollout with gradients preserved; varying present-frame
    representations are recomputed each step. Reference prefill likewise runs
    in the caller's gradient mode. Scoring retains its single inline pass.
    """

    def __init__(self, band_steps=(2, 2, 2, 2), history_t=.2,
                 history_dropout=(0., .1, .2, .2), trim=True, cache_context=True):
        if (len(band_steps) != P or len(set(band_steps)) != 1
                or any(type(steps) is not int or steps < 1 for steps in band_steps)):
            raise ValueError('Four equally subdivided stabilizer bands are required')
        if not math.isfinite(history_t) or not 0 < history_t < 1:
            raise ValueError('history_t must lie strictly inside (0,1)')
        if (len(history_dropout) > 6
                or any(not math.isfinite(p) or not 0 <= p <= 1 for p in history_dropout)):
            raise ValueError('History dropout requires at most six finite probabilities in [0,1], nearest first')
        if type(cache_context) is not bool:
            raise ValueError('cache_context must be a bool')
        self.steps = band_steps[0]
        self.history_t = history_t
        self.width = (1 - history_t) / P
        self.history_dropout = tuple(history_dropout)
        self.trim = trim
        self.cache_context = cache_context
        self.plans = {present: _plan(present) for present in range(1, P + 1)}
        self.cached_plans = {present: cached_stabilizer_plan(present) for present in range(1, P + 1)}

    def _keep(self, rows):
        valid = rows['ofi'][:, 1:7] >= 1
        explicit = rows.get('history_keep')
        if explicit is not None:
            if explicit.shape != valid.shape or explicit.dtype != torch.bool:
                raise ValueError('history_keep must be bool [B,6], oldest first')
            return valid & explicit
        rates = torch.zeros(6, dtype=torch.float32, device=valid.device)
        for distance, probability in enumerate(self.history_dropout):
            rates[5 - distance] = probability
        if not any(self.history_dropout):
            return valid
        return valid & (torch.rand(valid.shape, device=valid.device) >= rates)

    def prepare_context(self, rows, noise=None):
        """Return (full-row noise, detached context); no network call occurs.

        Noise covers [B,C,8,h,w]. The final frame seeds the target; frames 1..6
        supply independent context noises retained for its complete trajectory
        and scoring. All context fields are batch tensors for pool slicing.
        """
        gt, pose, ofi = _validate_rows(rows)
        if noise is None:
            noise = torch.randn_like(gt, dtype=torch.float32)
        if noise.shape != gt.shape:
            raise ValueError('Fixed generation noise must cover the complete eight-frame row')
        context = {
            'history': gt[:, :, 1:7].detach(),
            'noise': noise[:, :, 1:7].detach(),
            'pose': pose[:, :, 1:7].detach(),
            'ofi': ofi[:, 1:7].detach(),
            'keep': self._keep(rows).detach(),
            'sink': gt[:, :, 0].detach(),
            'target_pose': pose[:, :, -1].detach(),
            'target_ofi': ofi[:, -1].detach(),
            'drop': rows['drop'].detach(),
        }
        return noise, context

    def _forward(self, net, value, target_time, bands, context, tokens, present, drop, context_external=None):
        """Pack a moving target plus GT context, including mixed scoring bands."""
        frames, audio_indices, audio_valid = [], [], []
        first_history = H - bands
        for slot in range(present):
            source = (6 - bands + slot).clamp(max=5)
            history = _gather_frames(context['history'], source)[:, :, 0]
            noise = _gather_frames(context['noise'], source)[:, :, 0]
            history_pose = _gather_frames(context['pose'], source)[:, :, 0]
            target = bands == slot
            valid = (bands >= slot) & (target | context['keep'].gather(1, source[:, None])[:, 0])
            physical_time = target_time[:, 0] + self.width * (slot - bands)
            time = torch.where(valid, physical_time, torch.zeros_like(physical_time))
            x = torch.where(target[:, None, None, None], value[:, :, 0],
                            _gt_at(history, noise, physical_time))
            pose = torch.where(target[:, None, None, None], context['target_pose'], history_pose)
            x = torch.where(valid[:, None, None, None], x, torch.zeros_like(x))
            pose = torch.where(valid[:, None, None, None], pose, torch.zeros_like(pose))
            frames.append((x, pose, time, H + 1 + slot, valid))
            source_ofi = context['ofi'].gather(1, source[:, None])[:, 0]
            audio_indices.append(torch.where(target, context['target_ofi'], source_ofi))
            audio_valid.append(valid)
        if context_external is None:
            for offset in range(H):
                source = first_history + offset
                valid = context['keep'].gather(1, source[:, None])[:, 0]
                history = _gather_frames(context['history'], source)[:, :, 0]
                noise = _gather_frames(context['noise'], source)[:, :, 0]
                pose = _gather_frames(context['pose'], source)[:, :, 0]
                time = valid.float() * self.history_t
                x = _gt_at(history, noise, time)
                x = torch.where(valid[:, None, None, None], x, torch.zeros_like(x))
                pose = torch.where(valid[:, None, None, None], pose, torch.zeros_like(pose))
                frames.append((x, pose, time, H, valid))
            frames.append((context['sink'], None, 0., 0, True))
            plan, external = self.plans[present], None
        else:
            plan = self.cached_plans[present]
            external = context_external
        audio = net.frame_audio(tokens, torch.stack(audio_indices, dim=1), drop)
        audio_mask = torch.stack(audio_valid, dim=1)
        velocity, _, _ = net.forward_pass(frames, plan, external,
                                         audio, audio_mask, present)
        # Earlier present outputs are conditioning computations only. Their
        # predicted velocities are discarded, never integrated into context.
        return _gather_frames(velocity, bands)

    def generate(self, net, rows, noise=None, context=None, observe_velocity=None):
        """Return one endpoint [B,C,1,h,w] and its unchanged sticky context.

        An optional observer receives detached target velocities already
        computed for integration; it must not mutate them or retain tensors.
        """
        gt, _, _ = _validate_rows(rows)
        if context is None:
            noise, context = self.prepare_context(rows, noise)
        else:
            _validate_context(rows, context)
            if noise is None:
                noise = torch.randn_like(gt, dtype=torch.float32)
            if noise.shape != gt.shape:
                raise ValueError('Fixed generation noise must cover the complete eight-frame row')
        x = noise[:, :, -1:].clone()
        with net.references(rows['refs'], rows['ref_keep']):
            cache = prefill_stabilizer_context(net, context, self.history_t) if self.cache_context else None
            tokens = net.audio_tokens(rows['audio'])
            for band in reversed(range(P)):
                bands = torch.full((gt.shape[0],), band, dtype=torch.long, device=gt.device)
                # Both Euler substeps read the same fixed identities. Share
                # gathered K/V so checkpoint closures retain one copy per band.
                external = select_stabilizer_context(cache, bands) if cache is not None else None
                for step in range(self.steps):
                    time = self.history_t + self.width * (band + 1) - self.width * step / self.steps
                    times = x.new_full((x.shape[0], 1), time)
                    velocity = self._forward(net, x, times, bands, context, tokens,
                                             band + 1 if self.trim else P, context['drop'], external)
                    if observe_velocity is not None:
                        observe_velocity(velocity.detach())
                    x = x - (self.width / self.steps) * velocity
        return x, context

    def score_pass(self, net, rows, x, t, context, guidance_scale=1.):
        """Query one noisy target at its sampled time and corresponding slot.

        Context noise and dropout are reused verbatim. Optional audio-only CFG
        affects targets in the highest-noise band only. Conditional rows and
        their selected null-audio copies share ONE batched network pass; null
        rows retain pose, references, sink, GT/noise draws, and temporal masks.
        Every rank executes one reference prefill and one main pass, including
        ranks with no top-band targets, preserving teacher-shard collectives.
        """
        _validate_rows(rows)
        _validate_context(rows, context)
        if x.shape != rows['latents'][:, :, -1:].shape or t.shape != (x.shape[0], 1):
            raise ValueError('Scoring requires one target [B,C,1,h,w] and time [B,1]')
        if not math.isfinite(guidance_scale) or guidance_scale < 0:
            raise ValueError('Guidance scale must be finite and nonnegative')
        # Exact boundaries belong to the lower-noise band, matching the
        # incoming integration step at .4/.6/.8 in the recovered implementation.
        bands = ((t[:, 0] - self.history_t) / self.width - 1e-6).floor().long().clamp(0, P - 1)
        if guidance_scale == 1.:
            with net.references(rows['refs'], rows['ref_keep']):
                tokens = net.audio_tokens(rows['audio'])
                return self._forward(net, x, t, bands, context, tokens, P, context['drop'])

        # Lower-band rows have no null counterpart. Batch size is B + K, where
        # K is the local number of top-band targets (0..B). This changes only
        # local tensor shapes, never the number/order of sharded model calls.
        selected = torch.nonzero(bands == P - 1, as_tuple=False).flatten()
        indices = torch.cat((torch.arange(x.shape[0], device=x.device), selected))
        paired_context = {key: value.index_select(0, indices) for key, value in context.items()}
        drop = torch.cat((context['drop'], torch.ones_like(context['drop'].index_select(0, selected))))
        with net.references(rows['refs'].index_select(0, indices), rows['ref_keep'].index_select(0, indices)):
            # Keep the original encoder batch/layout. Copying the transposed
            # audio latents first changes BF16 GEMM numerics even when K=0.
            # Null copies reuse tokens but mask every audio window below.
            tokens = net.audio_tokens(rows['audio']).index_select(0, indices)
            velocity = self._forward(net, x.index_select(0, indices), t.index_select(0, indices),
                bands.index_select(0, indices), paired_context, tokens, P, drop)
        conditioned, null = velocity[:x.shape[0]], velocity[x.shape[0]:]
        guided = null + guidance_scale * (conditioned.index_select(0, selected) - null)
        return conditioned.index_copy(0, selected, guided)

    def passes(self):
        return P * self.steps


def score_resolver(net, rows, x, t, context):
    """One resolver target, with clean final-two GT predecessors and sink.

    The supplied context determines the globally sticky validity decisions;
    this function does not draw a mask. The same six-identity full-GT dropout
    policy applies, with only the final two predecessors participating here.
    """
    _validate_rows(rows)
    _validate_context(rows, context)
    if x.shape != rows['latents'][:, :, -1:].shape or t.shape != (x.shape[0], 1):
        raise ValueError('Resolver scoring requires one target [B,C,1,h,w] and time [B,1]')
    frames = [(x[:, :, 0], context['target_pose'], t[:, 0], 3, True)]
    for index in (4, 5):
        valid = context['keep'][:, index]
        history = context['history'][:, :, index]
        pose = context['pose'][:, :, index]
        history = torch.where(valid[:, None, None, None], history, torch.zeros_like(history))
        pose = torch.where(valid[:, None, None, None], pose, torch.zeros_like(pose))
        frames.append((history, pose, 0., 2, valid))
    frames.append((context['sink'], None, 0., 0, True))
    plan = [(0, [[('f', 3), ('r', 1, 1), ('r', 2, 2), ('f', 0)]]),
            (1, [[('f', 3), ('f', 1)], [('f', 3), ('f', 2)]]),
            (3, [[('f', 3)]])]
    with net.references(rows['refs'], rows['ref_keep']):
        tokens = net.audio_tokens(rows['audio'])
        audio = net.frame_audio(tokens, context['target_ofi'][:, None], context['drop'])
        audio_mask = torch.ones_like(t, dtype=torch.bool)
        velocity, _, _ = net.forward_pass(frames, plan, None, audio, audio_mask, 1)
        return velocity


def generate_resolver(net, rows, context, noise=None, t_start=.2, observe_velocity=None):
    """One differentiable GT-noised resolver step to zero, returning context.

    Target noise is [B,C,1,h,w]. Context is prepared once with
    FullGTStabilizer.prepare_context(), retaining its fixed history mask for
    student generation and the subsequent teacher/critic queries.
    An optional observer receives the detached integration velocity, without
    an extra network call, and must not mutate it or retain tensors.
    """
    gt, _, _ = _validate_rows(rows)
    if not math.isfinite(t_start) or not 0 < t_start < 1:
        raise ValueError('Resolver starting time must lie strictly inside (0,1)')
    target = gt[:, :, -1:].detach()
    if noise is None:
        noise = torch.randn_like(target, dtype=torch.float32)
    if noise.shape != target.shape:
        raise ValueError('Resolver generation noise must have one-target shape [B,C,1,h,w]')
    times = noise.new_full((noise.shape[0], 1), t_start)
    x = (1 - t_start) * target + t_start * noise
    velocity = score_resolver(net, rows, x, times, context)
    if observe_velocity is not None:
        observe_velocity(velocity.detach())
    return x - t_start * velocity, context
