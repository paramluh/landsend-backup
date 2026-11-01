"""Bounded, optional diagnostics for cached actor versus learner replay."""
from __future__ import annotations

import numpy as np


def benchmark_prefix(seed, frames):
    """Explicit benchmark-only row prefix; never used by production configs."""
    from .data import sampling_masks
    if type(frames) is not int or frames <= int(seed['prefill_frames']) + 2:
        raise ValueError('Benchmark prefix must exceed the prefill plus media delay')
    original = int(seed['conversation_frames'])
    n = min(frames, original)
    prefix = int(np.asarray(seed['P']).reshape(-1)[0])
    total = prefix + n + 2
    out = dict(seed)
    for key in ('ids', 'depth', 'present'):
        out[key] = np.array(seed[key][:, :total], copy=True)
    out['pdepth'] = np.array(seed['pdepth'][:total], copy=True)
    out['pid'] = np.arange(total, dtype=np.int64)
    out['vin'] = np.asarray([prefix, n], np.int64)
    out['segs'] = [total]
    out['conversation_frames'] = n
    out['benchmark_original_frames'] = original
    out['wav'] = np.array(seed['wav'][:n * 1920], copy=True)
    out['sampled_masks'] = sampling_masks(prefix, n, int(seed['prefill_frames']))
    return out


class PolicyAgreement:
    """Retain only the first N sampled frames per stream and physical row.

    Statistics describe replay log p minus actor log p for the SAME sampled
    action and history. They are not categorical KL estimates. Prefill, forced
    terminal text and dummy rows are excluded by the learner's sampled mask.
    """

    def __init__(self, frames_per_row=128):
        if type(frames_per_row) is not int or frames_per_row <= 0:
            raise ValueError('Policy agreement frame count must be positive')
        self.frames_per_row = frames_per_row
        self._blocks = {stream: [] for stream in ('text', 'audio', 'pose')}

    def add(self, bundles, rollouts, replay_logp, sampled, *, row_offset=0):
        """Extract bounded slices from one packed learner microbatch to CPU."""
        import torch
        if len(bundles) != len(rollouts):
            raise ValueError('Policy agreement requires one rollout per replay row')
        total = sum(len(rollout.text) for rollout in rollouts)
        if tuple(replay_logp.shape) != (total, 65) or tuple(sampled.shape) != (total, 65):
            raise ValueError('Policy agreement expects a packed [positions,65] replay')
        offset = 0
        for index, (bundle, rollout) in enumerate(zip(bundles, rollouts)):
            raw = bundle['student']
            length = len(rollout.text)
            if raw.get('dummy', False):
                offset += length
                continue
            identity = {'row_slot': row_offset + index, 'row_key': str(raw.get('row_key', '')),
                        'item': list(map(int, raw['item'])) if 'item' in raw else None}
            for stream, column, width in (('text', 0, 1), ('audio', 1, 32), ('pose', 33, 32)):
                positions = np.flatnonzero(rollout.sampled_masks[stream])[:self.frames_per_row]
                if not len(positions):
                    continue
                selected = torch.as_tensor(positions + offset, device=replay_logp.device)
                replay = replay_logp[:, column:column + width].index_select(0, selected)
                valid = sampled[:, column:column + width].index_select(0, selected)
                replay = replay.detach().float().cpu().numpy().astype(np.float64)
                valid = valid.detach().cpu().numpy().astype(bool)
                actor = np.asarray(rollout.behavior_logprobs[stream], dtype=np.float64)[positions]
                if width == 1:
                    actor = actor[:, None]
                self._blocks[stream].append({'identity': identity, 'positions': positions,
                    'prefix': int(rollout.prefix), 'prefill': int(rollout.prefill_frames),
                    'actor': actor, 'replay': replay, 'valid': valid})
            offset += length

    def _statistics(self, stream, codebook=None):
        pieces, worst, worst_abs = [], None, -1.
        nonfinite = 0
        for block in self._blocks[stream]:
            actor, replay, valid = (block[key] for key in ('actor', 'replay', 'valid'))
            if codebook is not None:
                actor, replay, valid = (value[:, codebook:codebook + 1] for value in (actor, replay, valid))
            delta = replay - actor
            finite = np.isfinite(actor) & np.isfinite(replay)
            nonfinite += int((valid & ~finite).sum())
            selected = valid & finite
            pieces.append(delta[selected])
            if not selected.any():
                continue
            magnitude = np.where(selected, np.abs(delta), -1.)
            frame, column = np.unravel_index(magnitude.argmax(), magnitude.shape)
            value = float(magnitude[frame, column])
            if value <= worst_abs:
                continue
            worst_abs = value
            position = int(block['positions'][frame])
            model_frame = position - block['prefix']
            physical_frame = model_frame - (0 if stream == 'text' else 2)
            worst = {**block['identity'], 'stream': stream,
                'model_position': position, 'model_frame': model_frame,
                'physical_frame': physical_frame, 'generated_frame': physical_frame - block['prefill'],
                'codebook': None if stream == 'text' else int(column if codebook is None else codebook),
                'actor_logprob': float(actor[frame, column]), 'replay_logprob': float(replay[frame, column]),
                'signed_delta': float(delta[frame, column])}
        values = np.concatenate(pieces) if pieces else np.empty(0)
        result = {'count': int(len(values)), 'nonfinite_count': nonfinite,
                  'mean_signed': None, 'mean_abs': None, 'max_abs': None,
                  'p50_abs': None, 'p95_abs': None, 'p99_abs': None, 'worst': worst}
        if len(values):
            absolute = np.abs(values)
            result.update(mean_signed=float(values.mean()), mean_abs=float(absolute.mean()),
                          max_abs=float(absolute.max()))
            result.update(zip(('p50_abs', 'p95_abs', 'p99_abs'),
                              map(float, np.quantile(absolute, (.5, .95, .99)))))
        return result

    def report(self):
        streams = {}
        for stream in self._blocks:
            streams[stream] = self._statistics(stream)
            if stream != 'text':
                streams[stream]['codebooks'] = {str(cb): self._statistics(stream, cb) for cb in range(32)}
        return {'scope': 'first_sampled_frames_per_stream_per_row',
                'frames_per_row': self.frames_per_row, 'delta': 'replay_logprob_minus_actor_logprob',
                'quantiles': 'absolute_sampled_logprob_delta', 'streams': streams}


class CategoricalAgreement:
    """Compare sparse full distributions captured on the sampled actor history."""
    def __init__(self, bundles, rollouts, *, row_offset=0):
        self.requests, self.text_indices, self.captured = {}, {}, {}
        offset, text_offset = 0, 0
        for row, (bundle, rollout) in enumerate(zip(bundles, rollouts)):
            raw = bundle['student']
            text_positions = np.asarray(raw['text_positions'])
            if not raw.get('dummy', False):
                for probe in getattr(rollout, 'policy_probes', ()):
                    position, stream, cb = int(probe['position']), probe['stream'], int(probe['codebook'])
                    if not rollout.sampled_masks[stream][position]:
                        continue
                    key = (stream, position + offset, cb)
                    action = (rollout.text[position] if stream == 'text' else
                              getattr(rollout, stream)[position, cb] - 2048 * cb)
                    self.requests[key] = {**probe, 'action': int(action), 'row_slot': row_offset + row,
                        'row_key': str(raw.get('row_key', '')), 'prefix': int(rollout.prefix)}
                    if stream == 'text':
                        selected = int(np.searchsorted(text_positions, position))
                        if selected >= len(text_positions) or text_positions[selected] != position:
                            raise ValueError('Actor text probe is not a replay text decision')
                        self.text_indices[text_offset + selected] = key
            offset += len(rollout.text)
            text_offset += len(text_positions)

    def text(self, start, logits):
        for index, key in self.text_indices.items():
            if start <= index < start + len(logits):
                self.captured[key] = logits[index - start].detach().float().cpu()

    def cb0(self, stream, logits):
        for key in self.requests:
            if key[0] == stream and key[2] == 0:
                self.captured[key] = logits[key[1]].detach().float().cpu()

    def depth(self, stream, positions, logits):
        wanted = {key[1]: key for key in self.requests if key[0] == stream and key[2] > 0}
        if not wanted:
            return
        positions = positions.detach().cpu().tolist()
        for index, position in enumerate(positions):
            if position in wanted:
                for key in self.requests:
                    if key[0] == stream and key[1] == position and key[2] > 0:
                        self.captured[key] = logits[index, key[2] - 1].detach().float().cpu()

    def report(self):
        import torch
        records, missing = [], []
        for key, request in self.requests.items():
            identity = {'row_slot': request['row_slot'], 'row_key': request['row_key'],
                        'stream': key[0], 'model_position': int(request['position']),
                        'model_frame': int(request['position']) - request['prefix'], 'codebook': key[2]}
            if key not in self.captured:
                missing.append(identity)
                continue
            actor = torch.as_tensor(request['logits']).double()
            replay = self.captured[key].double()
            if actor.shape != replay.shape or actor.ndim != 1:
                raise ValueError('Actor/replay probe vocabularies do not match')
            la, lr = actor.log_softmax(-1), replay.log_softmax(-1)
            pa, pr = la.exp(), lr.exp()
            action = request['action']
            records.append({**identity, 'vocabulary': len(actor), 'sampled_action': action,
                'kl_actor_to_replay': float((pa * (la - lr)).sum()),
                'kl_replay_to_actor': float((pr * (lr - la)).sum()),
                'logit_mean_abs_difference': float((replay - actor).abs().mean()),
                'logit_max_abs_difference': float((replay - actor).abs().max()),
                'sampled_logprob_signed_delta': float(lr[action] - la[action])})
        return {'same_sampled_history': True, 'requested': len(self.requests),
                'captured': len(records), 'missing': missing, 'probes': records}
