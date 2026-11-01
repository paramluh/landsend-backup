"""Memory-bounded two-teacher replay using the production PPM architecture.

First score every categorical decision without gradients. Then replay the
student with direct analytic KL and detached finite-credit coefficients. Depth
chunks backpropagate immediately into detached tower roots; one final backward
propagates their accumulated gradients through the complete temporal graph.
"""
from __future__ import annotations

import contextlib
from dataclasses import replace
from functools import lru_cache
import hashlib
import importlib
from pathlib import Path
import sys
import types

import numpy as np
import torch
import torch.distributed as dist

from .objectives import DecisionLayout, categorical_kl, causal_credit_weights, sampled_log_probs
from .text_head import chunked_text_objective
from .teacher_batch import prepare_teacher_update, pose_temporal, teacher_depth_logits as _teacher_depth_logits

M = None


def load_native():
    global M
    if M is None:
        root = Path(__file__).resolve().parents[1] / 'duplex_distill' / 'production'
        for path in (root / 'lib', root):
            sys.path.insert(0, str(path))
        existing = sys.modules.get('model_ppm')
        if existing is not None and Path(existing.__file__).resolve() != root / 'model_ppm.py':
            raise RuntimeError('Another production model module is already loaded')
        M = importlib.import_module('model_ppm')
        M.Head.forward = _head_forward
        M.Tower.forward = _tower_forward
        M.PPM.forward = _ppm_forward
        M.Decoder.forward = _decoder_forward
    return M


def build_frozen(c32, c4, c06, device, *, proj_r=2048, tower_vocab=151682,
                 pose_head=False, decp_bb=False, vadapter=None, nopose=False, log=print):
    """Production architecture with BF16 frozen storage, before FSDP wrapping.

    Exported model weights are already BF16. Only the small trained Voxtral
    log-gain requires FP32 storage; converting an existing FSDP model would
    invalidate its parameter metadata, so construct this layout on meta first.
    """
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    with torch.device('meta'):
        model = M.PPM(c32, c4, c06, proj_r=proj_r, tower_vocab=tower_vocab,
                      pose_head=pose_head, decp_bb=decp_bb, vadapter=vadapter, nopose=nopose)
        model.to(dtype=torch.bfloat16)
        if hasattr(model, 'vgain'):
            model.vgain.float()
        model.requires_grad_(False)
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    mpn = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=False)
    fully_shard(model.inp, mp_policy=mp, reshard_after_forward=True)
    for layer in model.layers:
        fully_shard(layer, mp_policy=mp, reshard_after_forward=True)
    fully_shard(model.head, mp_policy=mpn, reshard_after_forward=True)
    for name in ('tower_s', 'tower_p'):
        tower = getattr(model, name, None)
        if tower is not None:
            for layer in tower.layers:
                fully_shard(layer, mp_policy=mp, reshard_after_forward=True)
            fully_shard(tower, mp_policy=mpn, reshard_after_forward=True)
    for name in ('pose_head', 'decp_bb', 'vadapter'):
        if hasattr(model, name):
            fully_shard(getattr(model, name), mp_policy=mp if name == 'vadapter' else mpn, reshard_after_forward=True)
    if hasattr(model, 'vgain'):
        fully_shard(model.vgain, mp_policy=MixedPrecisionPolicy(param_dtype=torch.float32,
                    reduce_dtype=torch.float32, cast_forward_inputs=False), reshard_after_forward=True)
    for name in ('dec_s', 'dec_p'):
        decoder = getattr(model, name, None)
        if decoder is not None:
            for layer in decoder.model.layers:
                fully_shard(layer, mp_policy=mp, reshard_after_forward=True)
            fully_shard(decoder, mp_policy=mpn, reshard_after_forward=True)
    fully_shard(model, mp_policy=mpn, reshard_after_forward=True)
    model.to_empty(device=device)
    for name in ('dec_s', 'dec_p'):
        decoder = getattr(model, name, None)
        if decoder is not None:
            decoder.model.rotary_emb = type(decoder.model.rotary_emb)(config=c06, device=device)
    log(f'[build] frozen BF16 storage, {sum(p.numel() for p in model.parameters()) / 1e9:.3f}B parameters')
    return model, {}


def _head_forward(self, h, b, role, teacher=None):
    hn = M.FO.rms(h, self.norm.weight, float(self.norm.variance_epsilon))
    if role == 'pose_teacher':
        return {'hn': hn}
    x = self.proj(hn.index_select(0, b['sel']))
    replacement = torch.cat((self.spec_out, self.nw_out, self.eob_out), 0)
    if role == 'audio_teacher':
        return {'hn': hn, 'x': x.detach(), 'W': self.lm_head.weight.detach().clone(),
                'R': replacement.detach().clone()}
    loss, kl, lp = chunked_text_objective(
        x, self.lm_head.weight, replacement, teacher['x'], teacher['W'], teacher['R'],
        b['y'], b['text_weight'], b['text_credit'], special_ids=M.SPEC,
        executable_vocab=M.TOWER_VOCAB, chunk_size=b['text_chunk_frames'],
        logits_callback=b.get('policy_probe_text'))
    return {'hn': hn, 'loss': loss, 'kl': kl, 'logp': lp}


def _tower_forward(self, ids, depth, hn, pid, ctx, rot):
    c = M.shift1(self.cond(hn), ctx, 0.)
    e = self.text_emb(ids)
    h = (e + self.code_emb(M.shift1(depth, ctx, -1)).to(e.dtype) + c.to(e.dtype))[None]
    pe = rot(h, pid)
    for layer in self.layers:
        h = layer(h, pe, ctx)
    hv = M.FO.rms(h[0], self.norm.weight, float(self.norm.variance_epsilon))
    return hv, (hv @ self.cb0_head.weight.to(hv.dtype).t()).float()


def _ppm_forward(self, b, rot, *, role, teacher=None):
    e = self.inp(b['ids'], b['depth'], b['present'], b.get('pdepth'))[None]
    if b.get('vlat') is not None:
        v = self.vgain(self.vadapter(b['vlat'])[0])
        e = e + torch.zeros_like(e[0]).index_copy(0, b['vpos'], v.to(e.dtype))[None]
    ctx = M.Ctx(b['segs'], e.device)
    h, pe = e, rot(e, b['pid'])
    for layer in self.layers:
        h = layer(h, pe, ctx)
    head = self.head(h[0], b, role, teacher=teacher)
    hn = head.pop('hn')
    out = {'text': head}
    if role != 'pose_teacher':
        out['hv_s'], out['logits_s'] = self.tower_s(b['ids'][1], b['depth'][1], hn, b['pid'], ctx, rot)
    if role != 'audio_teacher':
        out['hv_p'], out['logits_p'] = self.tower_p(b['ids'][1], b['pdepth'], hn, b['pid'], ctx, rot)
        out['zb_p'] = M.shift1(self.decp_bb(hn), ctx, 0.) * float(self.decp_bb_alpha)
    return out


def _depth_norm(module, value):
    """HF RMSNorm with the actor's FP32 upcast of rounded BF16 weights.

    Q/K projections are BF16, so HF rounds normalized activations to BF16
    BEFORE multiplying by the FP32 norm weight. Residual norms stay FP32.
    """
    original_dtype = value.dtype
    value = value.float()
    value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + module.variance_epsilon)
    return module.weight.float() * value.to(original_dtype)


def _depth_layer(self, h, cos, sin):
    # Called through the layer's __call__, so FSDP hooks gather its parameters.
    a = self.self_attn
    y = _depth_norm(self.input_layernorm, h)
    shape = (*h.shape[:2], -1, a.head_dim)
    q, k = _depth_norm(a.q_norm, a.q_proj(y).view(shape)), _depth_norm(a.k_norm, a.k_proj(y).view(shape))
    v = a.v_proj(y).view(shape)
    q, k = M.Q3.apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)
    dtype = torch.bfloat16 if h.is_cuda else h.dtype
    z = M.DA.attn32(q.to(dtype), k.to(dtype), v.to(dtype), a.scaling)
    h = h + a.o_proj(z.reshape(*h.shape[:2], -1))
    return h + self.mlp(_depth_norm(self.post_attention_layernorm, h))


def install_decoder_layers(model, *, checkpoint_depth_decoder=True):
    from torch.distributed.fsdp import FSDPModule
    for name in ('dec_s', 'dec_p'):
        decoder = getattr(model, name, None)
        if decoder is not None:
            decoder._opd_checkpoint_depth_decoder = bool(checkpoint_depth_decoder)
            for layer in decoder.model.layers:
                if isinstance(layer, FSDPModule):
                    state = layer._get_fsdp_state()
                    # Parameters remain BF16 and gradients reduce in FP32.
                    # Casting the residual/RoPE inputs would change the actor policy.
                    state._mp_policy = replace(state._mp_policy, cast_forward_inputs=False)
                layer.forward = types.MethodType(_depth_layer, layer)


def _decoder_inputs(self, hv, depth, flat, zb=None):
    """Match FusedDec.slot0/emb before any residual or D33 addition."""
    codes = depth.index_select(0, flat)
    slot0 = self.proj(hv.index_select(0, flat)).float()
    if zb is not None:
        slot0 = slot0 + zb.index_select(0, flat).float()
    cb0 = self.cb0_emb(codes[:, 0]).float()
    rest = self.model.embed_tokens(codes[:, 1:31] - 2048).float()
    x = torch.cat((slot0[:, None], cb0[:, None], rest), 1)
    labels = codes[:, 1:] - torch.arange(1, 32, device=codes.device) * 2048
    return x, labels


def _decoder_forward(self, hv, depth, flat, zb=None):
    x, labels = _decoder_inputs(self, hv, depth, flat, zb)
    cos, sin = self.model.rotary_emb(x, torch.arange(x.shape[1], device=x.device)[None])
    h = x
    for layer in self.model.layers:
        checkpointed = torch.is_grad_enabled() and getattr(self, '_opd_checkpoint_depth_decoder', True)
        h = M.checkpoint(layer, h, cos, sin, use_reentrant=False) if checkpointed else layer(h, cos, sin)
    h = _depth_norm(self.model.norm, h)
    logits = torch.einsum('ndh,dvh->ndv', h[:, 1:], self.heads.weight.view(31, 2048, -1)).float()
    return logits, labels


@contextlib.contextmanager
def resident_decoders(*modules):
    """Amu's exact residency optimization; gradient reduce-scatter is unchanged."""
    from torch.distributed.fsdp import FSDPModule
    saved = []
    for module in modules:
        for sub in module.modules():
            if isinstance(sub, FSDPModule):
                state = sub._get_fsdp_state()
                group = state._fsdp_param_group
                if group is not None:
                    saved.append((sub, state, group, group.post_forward_mesh_info,
                                  group.reshard_after_backward, state._auto_reshard_after_forward))
                    group.post_forward_mesh_info = None
                    group.reshard_after_backward = False
    try:
        yield
    finally:
        for sub, state, group, mesh, backward, auto in saved:
            group.post_forward_mesh_info = mesh
            group.reshard_after_backward = backward
            state._auto_reshard_after_forward = auto
            sub.reshard()


@contextlib.contextmanager
def decoder_gradient_sync(decoder, mode):
    """Reduce accumulated decoder gradients only on the aligned final chunk.

    Read existing FSDP flags for restoration, but change them only through the
    public API. Full local gradients remain resident until the final backward.
    This context never applies to teacher modules or temporal-network groups.
    """
    from torch.distributed.fsdp import FSDPModule
    if mode not in ('per_chunk', 'per_replay'):
        raise ValueError('Unknown decoder gradient synchronization mode')
    saved = []
    if mode == 'per_replay':
        for module in decoder.modules():
            if isinstance(module, FSDPModule):
                group = module._get_fsdp_state()._fsdp_param_group
                if group is not None:
                    if not group.reduce_grads:
                        raise RuntimeError('Per-replay decoder synchronization cannot nest inside disabled gradient sync')
                    saved.append((module, group.reduce_grads, group.all_reduce_grads))

    def before_chunk(last):
        for module, _, all_reduce in saved:
            module.set_requires_gradient_sync(bool(last), recurse=False)
            if last and not all_reduce:
                module.set_requires_all_reduce(False, recurse=False)

    try:
        yield before_chunk
    finally:
        for module, reduce, all_reduce in saved:
            module.set_requires_gradient_sync(reduce, recurse=False)
            module.set_requires_all_reduce(all_reduce, recurse=False)


def mark_last_chunk(chunks):
    """One-item lookahead preserves globally padded call counts without a list."""
    iterator = iter(chunks)
    previous = next(iterator, None)
    if previous is None:
        return
    for value in iterator:
        yield previous, False
        previous = value
    yield previous, True


def event_layout(length, device, segments=None):
    stream = torch.tensor([0] + [1] * 32 + [2] * 32, device=device)
    cb = torch.tensor([-1] + list(range(32)) * 2, device=device)
    segments = [length] if segments is None else list(segments)
    if sum(segments) != length or any(n <= 0 for n in segments):
        raise ValueError('Packed event segments must cover the complete timeline')
    rows = torch.repeat_interleave(torch.arange(len(segments), device=device),
                                   torch.tensor(segments, device=device))
    frames = torch.cat([torch.arange(n, device=device) for n in segments])
    return DecisionLayout(rows.repeat_interleave(65), frames.repeat_interleave(65),
                          stream.repeat(length), cb.repeat(length))


def event_masks(raw, device):
    masks = {k: torch.as_tensor(v, device=device, dtype=torch.bool)
             for k, v in raw['sampled_masks'].items()}
    if raw.get('dummy', False):
        masks = {k: torch.zeros_like(v) for k, v in masks.items()}
    return torch.cat((masks['text'][:, None], masks['audio'][:, None].expand(-1, 32),
                      masks['pose'][:, None].expand(-1, 32)), 1)


def update_normalization(bundles, device):
    """Global real-row and per-modality counts for one whole optimizer update.

    Reuse this across accumulation microbatches; variable lengths and dummy
    slots then normalize exactly as a single effective batch would.
    """
    totals = torch.zeros(4, device=device, dtype=torch.float64)
    for bundle in bundles:
        raw = bundle['student']
        totals[0] += not raw.get('dummy', False)
        totals[1:] += event_masks(raw, device)[:, [0, 1, 33]].sum(0)
    if dist.is_initialized():
        dist.all_reduce(totals)
    return totals


def loss_weights(raw, config, device, *, world=1, normalization=None):
    """Each modality is either a global position mean or a mean of row means."""
    masks = event_masks(raw, device)
    counts = masks[:, [0, 1, 33]].sum(0).double()
    real = torch.tensor(float(not raw.get('dummy', False)), device=device, dtype=torch.float64)
    if config.reduction == 'position_mean':
        denom = counts.clone() if normalization is None else normalization[1:]
        if normalization is None and dist.is_initialized():
            dist.all_reduce(denom)
        scale = world / denom.clamp_min(1)
    else:
        if normalization is not None:
            real = normalization[0]
        elif dist.is_initialized():
            dist.all_reduce(real)
        scale = world / (real.clamp_min(1) * counts.clamp_min(1))
    cb = torch.as_tensor(config.codebook_weights, device=device)
    w = torch.zeros(masks.shape, device=device, dtype=torch.float32)
    w[:, 0] = config.modality_weights['text'] * scale[0]
    w[:, 1:33] = config.modality_weights['audio'] * scale[1] * cb
    w[:, 33:] = config.modality_weights['pose'] * scale[2] * cb
    return w * masks, masks


def pad_replay_bundle(bundle, fixed_frames):
    """Append an unscored causal tail while preserving the native prefix.

    Input audio positions, text targets, sampled media indices and physical
    conversation lengths remain unchanged. Every packed row receives its own
    complete segment, so padding cannot connect neighboring histories.
    """
    if fixed_frames is None:
        return bundle
    raw = bundle['student']
    length = len(raw['pid'])
    if length > fixed_frames:
        raise ValueError(f'Replay row has {length} positions, exceeding fixed_sequence_frames={fixed_frames}')
    if length == fixed_frames:
        return bundle
    pad_id = int(raw['pad_id'])
    result = {}
    for role, source in bundle.items():
        if len(source['pid']) != length or list(source['segs']) != [length]:
            raise ValueError('Static padding requires one aligned complete row per replay view')
        out = dict(source)
        for key, fill in (('ids', pad_id), ('depth', -1), ('present', 0)):
            value = np.asarray(source[key])
            shape = (value.shape[0], fixed_frames, *value.shape[2:])
            padded = np.full(shape, fill, dtype=value.dtype)
            padded[:, :length] = value
            out[key] = padded
        for key, fill in (('pdepth', -1), ('cb0', -100), ('pcb0', -100)):
            if key in source:
                value = np.asarray(source[key])
                padded = np.full((fixed_frames, *value.shape[1:]), fill, dtype=value.dtype)
                padded[:length] = value
                out[key] = padded
        positions = np.asarray(source['pid'])
        out['pid'] = np.concatenate((positions,
            positions[-1] + 1 + np.arange(fixed_frames - length, dtype=positions.dtype)))
        out['segs'] = [fixed_frames]
        out['sampled_masks'] = {key: np.pad(np.asarray(mask, dtype=bool), (0, fixed_frames - length))
                                for key, mask in source['sampled_masks'].items()}
        result[role] = out
    return result


def prepare(bundle, latents, config, device, *, world, normalization=None):
    if isinstance(bundle, (list, tuple)):
        bundles = list(bundle)
        if not bundles or len(bundles) != len(latents):
            raise ValueError('Packed replay requires matching nonempty rows and audio features')
        if normalization is None:
            normalization = update_normalization(bundles, device)
        prepared = [prepare(row, features, config, device, world=world, normalization=normalization)
                    for row, features in zip(bundles, latents)]
        packed = {}
        lengths = [len(weights) for _, weights, _ in prepared]
        offsets = np.cumsum([0, *lengths[:-1]]).tolist()
        timeline = {'ids', 'depth', 'present'}
        positions = {'sel', 'text_positions', 'light', 'plight', 'vpos'}
        used = timeline | positions | {'pid', 'pdepth', 'y', 'text_weight', 'text_credit', 'vlat'}
        for role in bundles[0]:
            entries = [value[0][role] for value in prepared]
            out = {}
            for key in used & entries[0].keys():
                values = [entry[key] + offset if key in positions else entry[key]
                          for entry, offset in zip(entries, offsets)]
                out[key] = torch.cat(values, dim=1 if key in timeline | {'pid', 'vlat'} else 0)
            out.update(segs=lengths, text_chunk_frames=config.text_chunk_frames)
            packed[role] = out
        return packed, torch.cat([value[1] for value in prepared]), torch.cat([value[2] for value in prepared])
    bundle = pad_replay_bundle(bundle, config.fixed_sequence_frames)
    raw = bundle['student']
    weights, masks = loss_weights(raw, config, device, world=world, normalization=normalization)
    result = {}
    for role, source in bundle.items():
        b = {k: torch.as_tensor(np.asarray(v), device=device)
             for k, v in source.items() if isinstance(v, np.ndarray) and k != 'wav'}
        b.update(segs=list(source['segs']), text_chunk_frames=config.text_chunk_frames)
        b['pid'] = b['pid'].reshape(1, -1)
        b['text_positions'] = torch.as_tensor(raw['text_positions'], device=device)
        b['text_weight'] = weights[b['text_positions'], 0]
        b['text_credit'] = torch.zeros_like(b['text_weight'])
        if role != 'pose_teacher':
            prefix, n = map(int, source['vin'])
            if tuple(latents.shape) != (1, 4 * n, 1280):
                raise ValueError(f'Causal A feature shape mismatch: {tuple(latents.shape)}, n={n}')
            b['vlat'] = latents.to(device)
            b['vpos'] = torch.arange(prefix, prefix + n, device=device)
        result[role] = b
    return result, weights, masks


def chunk_indices(positions, size):
    """Same number/shape of FSDP calls on every rank; padded frames weigh zero."""
    count = torch.tensor((positions.numel() + size - 1) // size, device=positions.device)
    if dist.is_initialized():
        dist.all_reduce(count, op=dist.ReduceOp.MAX)
    if not positions.numel():
        raise ValueError('Every real/fallback row must contain generated media')
    for start in range(0, int(count) * size, size):
        actual = positions[start:start + size]
        flat = positions[:1].expand(size).clone()
        flat[:len(actual)] = actual
        yield flat, len(actual)


def _media_term(sl, tl, ids, weights, credit):
    kl, lp = categorical_kl(sl, tl), sampled_log_probs(sl, ids)
    return (weights * kl + credit * (lp - lp.detach())).sum(), kl.detach(), lp.detach()


@lru_cache(maxsize=4)
def load_silence_codes(path, sha256):
    """Load pinned Mimi-encoded digital silence, in local codebook IDs."""
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != sha256:
        raise ValueError(f'Pose CFG silence file SHA256 mismatch: {path}')
    codes = np.load(path, allow_pickle=False)
    if (codes.ndim != 2 or codes.shape[0] != 32 or not codes.shape[1]
            or not np.issubdtype(codes.dtype, np.integer)
            or np.any(codes < 0) or np.any(codes >= 2048)):
        raise ValueError('Pose CFG silence must have shape [32, frames] and local Mimi IDs')
    return codes


def silent_pose_view(batch, silence):
    """Native monoplex CFG: silence every coded audio run; keep text and pose.

    This includes reference audio and the prefill. Missing audio stays absent,
    and silence frame indices restart at each contiguous run and packed row.
    """
    depth = batch['depth'].clone()
    silence = torch.as_tensor(silence, device=depth.device, dtype=depth.dtype)
    offsets = torch.arange(32, device=depth.device) * 2048
    start = 0
    for length in batch['segs']:
        for channel in range(depth.shape[0]):
            positions = (depth[channel, start:start + length, 0] >= 0).nonzero().flatten()
            if not positions.numel():
                continue
            indices = torch.arange(positions.numel(), device=depth.device)
            run_start = torch.cat((torch.ones(1, dtype=torch.bool, device=depth.device),
                                   positions[1:] != positions[:-1] + 1))
            within_run = indices - torch.where(run_start, indices, 0).cummax(0).values
            if int(within_run.max()) >= silence.shape[1]:
                raise ValueError('Pose CFG silence table is shorter than an audio run')
            depth[channel, start + positions] = silence[:, within_run].T + offsets
        start += length
    return {**batch, 'depth': depth}


def guided_pose_logits(conditional, unconditional, scale):
    """Guide categorical logits, before the teacher probability normalization."""
    if scale == 1:
        return conditional
    return unconditional.float() + scale * (conditional.float() - unconditional.float())


def replay_update(student, audio_teacher, pose_teacher, bundle, latents, config, rot, device, *, world=1,
                  normalization=None, policy_probes=None, stage_profiler=None, prepared_teachers=None):
    """Accumulate all student gradients, with no parameter update inside replay."""
    stage = stage_profiler.span if stage_profiler is not None else lambda _: contextlib.nullcontext()
    with stage('prepare'):
        b, weights, sampled = prepare(bundle, latents, config, device, world=world, normalization=normalization)
    s = b['student']
    rows = bundle if isinstance(bundle, (list, tuple)) else [bundle]
    original_lengths = [len(row['student']['pid']) for row in rows]
    padded = original_lengths != list(s['segs'])

    def original_timeline(value):
        if not padded:
            return value
        return torch.cat([part[:length] for part, length in zip(value.split(s['segs']), original_lengths)])

    def original_positions(positions):
        if not padded:
            return positions
        ends = torch.as_tensor(np.cumsum(s['segs'])[:-1], device=device)
        padding_offsets = torch.as_tensor(np.cumsum([0, *s['segs'][:-1]]) -
                                          np.cumsum([0, *original_lengths[:-1]]), device=device)
        return positions - padding_offsets[torch.bucketize(positions, ends, right=True)]
    costs = torch.zeros_like(weights)
    replay_logp = torch.zeros_like(weights)
    pose_unconditional = None
    cfg_scale = config.pose_teacher_cfg_scale
    cache_teacher = config.teacher_depth_cache == 'cpu' or (prepared_teachers is not None and prepared_teachers.cache is not None)
    teacher_cache = {'s': [], 'p': []}

    def teacher_depth_logits(key, decoder, conditional, codes, flat):
        return _teacher_depth_logits(key, decoder, conditional, pose_unconditional, codes, flat, config)

    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == 'cuda', dtype=torch.bfloat16):
        if prepared_teachers is None:
            with stage('audio_teacher_temporal'):
                audio = audio_teacher(b['audio_teacher'], rot, role='audio_teacher')
            pose, pose_unconditional = pose_temporal(pose_teacher, b['pose_teacher'], rot, config, stage)
        else:
            if tuple(s['segs']) != prepared_teachers.segments:
                raise ValueError('Prepared teacher rows do not match this student replay')
            audio, pose, pose_unconditional = prepared_teachers.audio, prepared_teachers.pose, prepared_teachers.unconditional
        if policy_probes is not None:
            s['policy_probe_text'] = policy_probes.text
        try:
            with stage('student_detached_temporal'):
                reference = student(s, rot, role='student', teacher=audio['text'])
        finally:
            s.pop('policy_probe_text', None)
        costs[s['text_positions'], 0] = reference['text']['kl']
        replay_logp[s['text_positions'], 0] = reference['text']['logp']
        for key, teacher, tout, codes, positions, offset in (
            ('s', audio_teacher, audio, s['depth'][1], s['light'], 1),
            ('p', pose_teacher, pose, s['pdepth'], s['plight'], 33)):
            costs[:, offset] = categorical_kl(reference['logits_' + key], tout['logits_' + key])
            replay_logp[:, offset] = sampled_log_probs(reference['logits_' + key], codes[:, 0].clamp_min(0))
            if policy_probes is not None:
                policy_probes.cb0('audio' if key == 's' else 'pose', original_timeline(reference['logits_' + key]))
            decoder, td = getattr(student, 'dec_' + key), getattr(teacher, 'dec_' + key)
            precached = prepared_teachers is not None and prepared_teachers.cache is not None
            with stage('detached_depth_audio' if key == 's' else 'detached_depth_pose'), resident_decoders(decoder, *((td,) if not precached else ())):
                for index, (flat, valid) in enumerate(chunk_indices(positions, config.decoder_chunk_frames)):
                    sl, labels = decoder(reference['hv_' + key], codes, flat, zb=reference.get('zb_' + key))
                    if policy_probes is not None:
                        policy_probes.depth('audio' if key == 's' else 'pose', original_positions(flat[:valid]), sl[:valid])
                    tl = (prepared_teachers.depth_chunk(key, index, config.decoder_chunk_frames, device)
                          if precached else teacher_depth_logits(key, td, tout, codes, flat))
                    if cache_teacher and not precached:
                        # Preserve every FP32 logit, including padded frames. A
                        # pinned copy runs on the current CUDA stream; retrieval
                        # below is ordered after it on that same stream.
                        if tl.dtype != torch.float32:
                            raise RuntimeError('Teacher depth logits must be FP32 before caching')
                        cached = torch.empty(tl.shape, dtype=tl.dtype, device='cpu', pin_memory=tl.is_cuda)
                        cached.copy_(tl.detach(), non_blocking=tl.is_cuda)
                        teacher_cache[key].append(cached)
                    k, lp = categorical_kl(sl, tl), sampled_log_probs(sl, labels)
                    costs[flat[:valid], offset + 1:offset + 32] = k[:valid]
                    replay_logp[flat[:valid], offset + 1:offset + 32] = lp[:valid]
        del reference
    with stage('credit_assignment'):
        credit = causal_credit_weights((costs * weights).flatten(), event_layout(len(costs), device, s['segs']),
                                       lookback_frames=config.lookback_frames, window_mode=config.window_mode,
                                       loss_mask=sampled.flatten(), sampled_mask=sampled.flatten()).reshape_as(weights).float()
        s['text_credit'] = credit[s['text_positions'], 0]
        contributions = (costs.double() * weights).sum(0).cpu().tolist()
        metrics = {'loss/direct': sum(contributions), 'loss/text': contributions[0],
                   'loss/audio': sum(contributions[1:33]), 'loss/pose': sum(contributions[33:]),
                   'credit/absolute_sum': float(credit.abs().sum())}
        for stream, offset in (('audio', 1), ('pose', 33)):
            for codebook in range(32):
                metrics[f'loss/{stream}_cb{codebook}'] = contributions[offset + codebook]
    with torch.autocast(device_type=device.type, enabled=device.type == 'cuda', dtype=torch.bfloat16):
        with stage('student_gradient_temporal'):
            out = student(s, rot, role='student', teacher=audio['text'])
        loss = out['text']['loss']
        roots, gradients = [], []
        for key, teacher, tout, codes, positions, offset in (
            ('s', audio_teacher, audio, s['depth'][1], s['light'], 1),
            ('p', pose_teacher, pose, s['pdepth'], s['plight'], 33)):
            term, _, _ = _media_term(out['logits_' + key], tout['logits_' + key], codes[:, 0].clamp_min(0),
                                    weights[:, offset], credit[:, offset])
            loss = loss + term
            hv = out['hv_' + key].detach().requires_grad_(True)
            zb = out.get('zb_' + key)
            zd = None if zb is None else zb.detach().requires_grad_(True)
            decoder, td = getattr(student, 'dec_' + key), getattr(teacher, 'dec_' + key)
            # Cached teachers have no second-pass forwards or all-gathers.
            with stage('audio_depth_forward_backward' if key == 's' else 'pose_depth_forward_backward'), \
                    resident_decoders(decoder, *((td,) if not cache_teacher else ())), \
                    decoder_gradient_sync(decoder, config.decoder_gradient_sync) as before_chunk:
                chunks = mark_last_chunk(chunk_indices(positions, config.decoder_chunk_frames))
                for index, ((flat, valid), last) in enumerate(chunks):
                    before_chunk(last)
                    with torch.no_grad():
                        tl = (prepared_teachers.depth_chunk(key, index, config.decoder_chunk_frames, device)
                              if prepared_teachers is not None and prepared_teachers.cache is not None else
                              teacher_cache[key][index].to(device, non_blocking=device.type == 'cuda')
                              if cache_teacher else teacher_depth_logits(key, td, tout, codes, flat))
                    sl, labels = decoder(hv, codes, flat, zb=zd)
                    w = weights[flat, offset + 1:offset + 32].clone()
                    c = credit[flat, offset + 1:offset + 32].clone()
                    w[valid:] = 0
                    c[valid:] = 0
                    depth_loss, _, _ = _media_term(sl, tl, labels, w, c)
                    depth_loss.backward()
            roots.append(out['hv_' + key])
            gradients.append(torch.zeros_like(hv) if hv.grad is None else hv.grad)
            if zb is not None:
                roots.append(zb)
                gradients.append(torch.zeros_like(zd) if zd.grad is None else zd.grad)
        with stage('final_temporal_backward'):
            torch.autograd.backward([loss, *roots], [None, *gradients])
    if any(p.grad is not None for model in (audio_teacher, pose_teacher) for p in model.parameters()):
        raise RuntimeError('Frozen teacher received a gradient')
    return metrics, original_timeline(replay_logp.detach()), original_timeline(sampled)
