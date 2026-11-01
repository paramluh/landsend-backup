"""Causal block refresh of already-committed conversational teacher outputs.

The six changed backbone positions and five changed speech-tower positions are
each processed as a short causal sequence against the existing KV prefix. No
text head or light sampling is performed here. The caller samples the one new
teacher position afterwards using the normal actor path.

Weights, normalization, RoPE and causal attention match FastPPM. GEMM/query
shapes differ from six single-position calls, so BF16 roundoff may differ; GPU
logit/KV parity must be checked before selecting this optimization for a run.
"""
from __future__ import annotations

import time
import json

import torch


def _capture(fn, device):
    with torch.cuda.device(device):
        stream = torch.cuda.Stream(device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                fn()
        torch.cuda.current_stream(device).wait_stream(stream)
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            fn()
        torch.cuda.synchronize(device)
    return graph


def _layer(engine, layer, hidden, cosine, sine, keys, values, lengths, *, batch, frames, fused):
    """FastPPM._layer_step with frames>1 and independent per-row cache prefixes."""
    attention = layer.self_attn
    before, after, mlp = layer.input_layernorm, layer.post_attention_layernorm, layer.mlp
    x = fused.rms(hidden, before.weight, float(before.variance_epsilon))
    q, k = fused.qk_prep(attention.q_proj(x), attention.k_proj(x),
                         attention.q_norm.weight, attention.k_norm.weight,
                         float(attention.q_norm.variance_epsilon), cosine, sine)
    dim = attention.head_dim
    value = attention.v_proj(x).view(batch, frames, -1, dim)
    attended = engine.fa.flash_attn_with_kvcache(
        q.view(batch, frames, -1, dim), keys, values,
        k=k.view(batch, frames, -1, dim), v=value, cache_seqlens=lengths,
        softmax_scale=attention.scaling, causal=True, num_splits=1,
    )
    hidden, x = fused.add_rms(hidden, attention.o_proj(attended.reshape(1, batch * frames, -1)),
                              after.weight, float(after.variance_epsilon))
    return hidden + mlp.down_proj(fused.silu_mul(mlp.gate_proj(x), mlp.up_proj(x)))


class CommittedBlockReplay:
    """One static block shape, with separately cached graphs per actor device."""
    def __init__(self, engine, frames, *, graphs=True):
        if engine.pose or engine.cfg_rows or engine.R != engine.NI:
            raise ValueError('Committed block replay is for an audio-only, non-CFG partner')
        if type(frames) is not int or frames < 2:
            raise ValueError('Committed block replay needs at least two positions')
        if getattr(engine, '_opd_sampler_backend', None) != 'inverse_cdf':
            raise ValueError('Use the natural, position-indexed conversational partner actor')
        if engine._opd_rewind_window < frames:
            raise ValueError('Rewind history must retain every refreshed block boundary')
        # Exact same kernel module used by the native actor; no alternate model import.
        import fused_triton
        self.fused, self.engine, self.frames = fused_triton, engine, frames
        self.use_graphs, self.graphs = bool(graphs), {}
        self.capture_seconds = 0.
        self.batch = engine.NI
        self.ar = {device: torch.arange(frames, device=device) for device in engine.sdevs}
        self.rows = {device: torch.arange(self.batch, device=device)[:, None] for device in engine.sdevs}
        self.tokens = {device: torch.empty(self.batch, frames, dtype=torch.long, device=device)
                       for device in set((engine.d0, engine.dl))}
        self.codes = {device: torch.empty(self.batch, frames, 32, dtype=torch.long, device=device)
                      for device in set((engine.d0, engine.dl))}
        shape = (1, self.batch * frames, engine.H)
        self.segment_input = {i: torch.empty(shape, dtype=torch.bfloat16, device=device)
                              for i, (device, _) in enumerate(engine.segs) if i}
        self.segment_output = {i: torch.empty(shape, dtype=torch.bfloat16, device=device)
                               for i, (device, _) in enumerate(engine.segs)}
        self.hn = torch.empty(self.batch, frames, engine.H, dtype=torch.bfloat16, device=engine.dl)

    def _positions(self, device):
        return self.engine.Tv[device][:self.batch, None] + self.ar[device][None]

    def _backbone(self, index):
        engine, count, batch = self.engine, self.frames, self.batch
        device, indices = engine.segs[index]
        positions = self._positions(device)
        lengths = positions[:, 0].to(torch.int32).contiguous()
        flat = positions.reshape(-1)
        cosine, sine = (table.index_select(1, flat) for table in engine.rope[device])
        if index == 0:
            rows = self.rows[device]
            ids = torch.stack((engine.idsA[rows, positions], self.tokens[device])).reshape(2, batch * count)
            codes = torch.stack((engine.depA[rows, positions], self.codes[device])).reshape(2, batch * count, 32)
            present = torch.stack((engine.presA[rows, positions], engine.presB[rows, positions])).reshape(2, batch * count)
            # NaturalCachedFastPPM's hook is inactive outside _seg/_prefill_row;
            # explicitly add the same adapter features once at every position.
            hidden = engine.m.inp(ids, codes, present)
            hidden = (hidden + engine._opd_audio[rows, positions].reshape(batch * count, engine.H).to(hidden.dtype))[None]
        else:
            hidden = self.segment_input[index]
        for layer_index in indices:
            hidden = _layer(engine, engine.bb.L[layer_index], hidden, cosine, sine,
                            engine.bbK[layer_index], engine.bbV[layer_index], lengths,
                            batch=batch, frames=count, fused=self.fused)
        self.segment_output[index].copy_(hidden)
        if index + 1 == len(engine.segs):
            norm = engine.m.head.norm
            self.hn.copy_(self.fused.rms(hidden[0], norm.weight, float(norm.variance_epsilon))
                          .view(batch, count, engine.H))

    def _tower(self):
        engine, count, batch = self.engine, self.frames - 1, self.batch
        device, tower = engine.dl, engine.m.tower_s
        positions = self._positions(device)[:, 1:]
        lengths = positions[:, 0].to(torch.int32).contiguous()
        flat = positions.reshape(-1)
        cosine, sine = (table.index_select(1, flat) for table in engine.rope_t)
        current_text = self.tokens[device][:, 1:].reshape(batch * count)
        previous_codes = self.codes[device][:, :-1].reshape(batch * count, 32)
        previous_hn = self.hn[:, :-1].reshape(batch * count, engine.H)
        hidden = (tower.text_emb(current_text) + tower.code_emb(previous_codes).to(torch.bfloat16)
                  + tower.cond(previous_hn).to(torch.bfloat16))[None]
        for index, layer in enumerate(engine.ts.L):
            hidden = _layer(engine, layer, hidden, cosine, sine, engine.tsK[index], engine.tsV[index],
                            lengths, batch=batch, frames=count, fused=self.fused)
        final = self.fused.rms(hidden[0], tower.norm.weight, float(tower.norm.variance_epsilon))
        last = final.view(batch, count, -1)[:, -1]
        engine.hv_s.copy_(last)
        engine.z0.copy_((last @ tower.cb0_head.weight.t()).float())

    def _run(self, name, device, fn):
        with torch.cuda.device(device):
            if not self.use_graphs:
                fn()
                return
            if name not in self.graphs:
                started = time.perf_counter()
                self.graphs[name] = _capture(fn, device)
                self.capture_seconds += time.perf_counter() - started
            self.graphs[name].replay()

    @torch.no_grad()
    def replay(self, start, stop):
        engine = self.engine
        if (type(start) is not int or type(stop) is not int or start < 0
                or stop - start != self.frames):
            raise ValueError('Replay interval differs from this static block shape')
        if engine._opd_pending_frame is not None or engine._opd_next_frame != start:
            raise ValueError('Rewind the partner to the block start before block replay')
        if engine._opd_input_mode is not None:
            raise ValueError('Block replay must not run inside another actor input hook')
        engine.set_f(start)
        before_capture = self.capture_seconds
        with torch.cuda.device(engine.dl):
            positions = self._positions(engine.dl)
            self.tokens[engine.dl].copy_(engine.laneB[self.rows[engine.dl], positions])
            self.codes[engine.dl].copy_(engine.depB[self.rows[engine.dl], positions])
        if engine.d0 != engine.dl:
            with torch.cuda.device(engine.d0):
                self.tokens[engine.d0].copy_(self.tokens[engine.dl])
                self.codes[engine.d0].copy_(self.codes[engine.dl])
        for index, (device, _) in enumerate(engine.segs):
            if index:
                with torch.cuda.device(device):
                    self.segment_input[index].copy_(self.segment_output[index - 1])
            self._run(f'backbone{index}', device, lambda index=index: self._backbone(index))
        self._run('tower', engine.dl, self._tower)
        # Preserve every per-position rewind boundary, including text-ring state.
        # These tiny state copies remain eager; the expensive layers are captured.
        for offset, frame in enumerate(range(start, stop)):
            engine.set_f(frame)
            with torch.cuda.device(engine.dl):
                engine._supply_text(self.tokens[engine.dl][:, offset])
                engine.hn_prev.copy_(self.hn[:, offset])
            engine._opd_next_frame, engine._opd_pending_frame = frame + 1, None
            engine._remember_frame(frame + 1)
        with torch.cuda.device(engine.dl):
            engine.pk[engine.dl][:, 0].copy_(self.tokens[engine.dl][:, -1])
            engine.pk[engine.dl][:, 1:].copy_(self.codes[engine.dl][:, -1])
            engine._opd_lane_force.copy_(self.tokens[engine.dl][:, -1])
        engine._opd_force = None
        return {'backbone_frames': self.frames, 'forced_tower_frames': self.frames - 1,
                'new_sampling_draws': 0, 'graphs': self.use_graphs,
                'capture_seconds_this_call': self.capture_seconds - before_capture}


def replay_committed_block(engine, start, stop, *, graphs=True):
    """Refresh [start,stop), leaving the actor ready to sample output at stop.

    Caller has already rewound the actor, updated all changed A features, and
    retained every committed text/audio output. One helper is cached per block
    length and execution mode; there is no growing-prefix model replay.
    """
    cache = getattr(engine, '_opd_partner_blocks', None)
    if cache is None:
        cache = engine._opd_partner_blocks = {}
    key = (stop - start, bool(graphs))
    if key not in cache:
        cache[key] = CommittedBlockReplay(engine, key[0], graphs=graphs)
    return cache[key].replay(start, stop)


def _synchronize(engine):
    for device in engine.sdevs:
        torch.cuda.synchronize(device)


def _committed_outputs(engine, frame):
    positions = engine.Pv[engine.dl][:engine.NI] + frame
    rows = engine.arI[engine.dl]
    return {'lane': engine.laneB[rows, positions], 'codes': engine.depB[rows, positions]}


def _sequential_refresh(engine, start, stop, graphs):
    for frame in range(start, stop):
        force = _committed_outputs(engine, frame)
        if frame != start:
            engine.sample_outputs(frame, graphs=graphs, force=force, sample_text=False)
        engine.advance_backbone(frame, graphs=graphs, force=force if frame == start else None)


def _changed_kv(engine, start, stop):
    """Only the bounded refreshed suffix, independently indexed for each row."""
    snapshots = {}
    for kind in ('bbK', 'bbV', 'tsK', 'tsV'):
        first = start + int(kind.startswith('ts'))
        for layer, cache in enumerate(getattr(engine, kind)):
            device = cache.device
            rows = torch.arange(engine.NI, device=device)[:, None]
            positions = engine.Pv[device][:engine.NI, None] + torch.arange(first, stop, device=device)[None]
            snapshots[f'{kind}.{layer}'] = cache[rows, positions].clone()
    return snapshots


def _next_logits(engine, frame, common_text=None):
    """Probe next text and cb0 logits using a common forced word, without a draw.

    The next tower KV position is uncommitted and will be overwritten after the
    caller's rewind. Its temporary lane token and scratch outputs are restored.
    No RNG, light decoder, waveform, or committed media array is advanced.
    """
    engine.set_f(frame)
    positions = engine.Tv[engine.dl][:engine.NI]
    rows = engine.arI[engine.dl]
    saved = {'lane': engine.laneB[rows, positions].clone(), 'tok_now': engine.tok_now.clone(),
             'pk': engine.pk[engine.dl].clone(), 'hv_s': engine.hv_s.clone(), 'z0': engine.z0.clone()}
    with torch.cuda.device(engine.dl):
        text = engine.text_logits(engine.hn_prev[:engine.NI])[:, :engine._opd_executable].float().clone()
        chosen = text.argmax(-1) if common_text is None else common_text
        try:
            engine.laneB[rows, positions] = chosen
            engine.tok_now.copy_(chosen)
            engine._speech(False)
            cb0 = engine.z0.float().clone()
        finally:
            engine.laneB[rows, positions] = saved['lane']
            engine.tok_now.copy_(saved['tok_now'])
            engine.pk[engine.dl].copy_(saved['pk'])
            engine.hv_s.copy_(saved['hv_s'])
            engine.z0.copy_(saved['z0'])
    return text, cb0, chosen


def _error(actual, expected):
    actual, expected = actual.float(), expected.float()
    delta = actual - expected
    statistics = torch.stack((delta.abs().max(), delta.square().mean().sqrt(),
                              expected.square().mean().sqrt(), torch.isfinite(actual).all().float()))
    maximum, rms, scale, finite = statistics.cpu().tolist()
    return {'max_abs': maximum, 'rms': rms, 'relative_rms': rms / max(scale, 1e-12),
            'reference_rms': scale, 'finite': bool(finite)}


def _distribution_error(actual, expected):
    result = _error(actual, expected)
    log_actual, log_expected = actual.float().log_softmax(-1), expected.float().log_softmax(-1)
    divergence = (log_expected.exp() * (log_expected - log_actual)).sum(-1)
    result.update(kl_reference_to_block_max=float(divergence.max()),
                  kl_reference_to_block_mean=float(divergence.mean()),
                  argmax_agreement=float((actual.argmax(-1) == expected.argmax(-1)).float().mean()))
    return result


@torch.no_grad()
def verify_block_replay(engine, start, stop, *, graphs=True, max_kl=2e-3,
                        max_relative_rms=.03, raise_on_failure=True):
    """Check actual teacher state/next distributions against forced sequential replay.

    The input features and retained tokens must already be corrected. This
    deliberately repeats only the six-position suffix, never a rollout. It
    measures a cold block call and a second warm call separately, then leaves
    the actor rewound to start for its normal selected-backend execution.
    """
    if stop <= start or engine.pose:
        raise ValueError('Block parity needs an audio-only teacher and a nonempty interval')
    if max_kl <= 0 or max_relative_rms <= 0:
        raise ValueError('Parity thresholds must be positive')
    if engine._opd_rewind_frames[start % engine._opd_rewind_window] != start:
        raise ValueError('Retain the actor boundary before verifying replay')
    timings = {}
    try:
        engine.rewind(start)
        _synchronize(engine)
        began = time.perf_counter()
        _sequential_refresh(engine, start, stop, graphs)
        _synchronize(engine)
        timings['sequential_seconds'] = time.perf_counter() - began
        reference_hn = engine.hn_prev.clone()
        reference_boundaries = {
            frame: engine._opd_rewind_states['hn_prev'][frame % engine._opd_rewind_window].clone()
            for frame in range(start + 1, stop + 1)}
        reference_kv = _changed_kv(engine, start, stop)
        reference_text, reference_audio, word = _next_logits(engine, stop)

        engine.rewind(start)
        _synchronize(engine)
        began = time.perf_counter()
        cold = replay_committed_block(engine, start, stop, graphs=graphs)
        _synchronize(engine)
        timings['block_cold_seconds'] = time.perf_counter() - began
        timings['block_capture_seconds'] = cold['capture_seconds_this_call']

        engine.rewind(start)
        _synchronize(engine)
        began = time.perf_counter()
        replay_committed_block(engine, start, stop, graphs=graphs)
        _synchronize(engine)
        timings['block_warm_seconds'] = time.perf_counter() - began
        timings['warm_speedup'] = timings['sequential_seconds'] / max(timings['block_warm_seconds'], 1e-12)
        hidden = _error(engine.hn_prev, reference_hn)
        boundaries = {str(frame): _error(
            engine._opd_rewind_states['hn_prev'][frame % engine._opd_rewind_window], value)
            for frame, value in reference_boundaries.items()}
        candidate_kv = _changed_kv(engine, start, stop)
        kv = {name: _error(candidate_kv[name], reference) for name, reference in reference_kv.items()}
        block_text, block_audio, _ = _next_logits(engine, stop, word)
        text, audio = _distribution_error(block_text, reference_text), _distribution_error(block_audio, reference_audio)
        state_errors = [hidden, *boundaries.values(), *kv.values()]
        passed = (all(value['finite'] and value['relative_rms'] <= max_relative_rms for value in state_errors)
                  and all(value['finite'] and value['kl_reference_to_block_max'] <= max_kl for value in (text, audio)))
        report = {'passed': passed, 'start': start, 'stop': stop,
            'method': 'same_corrected_features_and_committed_tokens_no_new_sampling',
            'thresholds': {'max_relative_rms': max_relative_rms, 'max_next_categorical_kl': max_kl},
            'timings': timings, 'hidden': hidden, 'boundaries': boundaries,
            'kv': kv, 'next_text': text, 'next_audio_cb0': audio,
            'new_light_draws': 0, 'new_waveform_frames': 0, 'restored_to_frame': start}
    finally:
        engine.rewind(start)
    if not report['passed'] and raise_on_failure:
        raise RuntimeError('Committed-block numerical parity failed: ' + json.dumps(report, allow_nan=False))
    return report
