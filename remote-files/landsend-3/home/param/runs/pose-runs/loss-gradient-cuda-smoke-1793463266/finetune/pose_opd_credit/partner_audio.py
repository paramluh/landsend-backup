"""Batched, GPU-resident audio for the speculative frozen conversational partner.

Both directions use the archived live frontend and causal Mimi decoder. An
optional fixed PCM delivery delay applies from the start of the conversation,
including its real prefill; native frontend windows and generated output clocks
remain unchanged. Generated Mimi waveforms undo the source speaker's recorded
output gain, then use the live input peak bound. Only the teacher's frontend
rewinds: published waveforms and Mimi decoder states are append-only.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .streaming import vox_stream as VS
from .streaming.vox_fast import VoxStreamFast
from .streaming.mimi_dec_stream import MimiDecStream


class AudioTimeline:
    """Append sender frames into the exact waveform heard by the receiver.

    ``available`` is the exclusive receiver-sample endpoint, including the fixed
    delivery delay. The leading delay is known zero audio. Delaying the entire
    prefix as well as generated chunks keeps their transition continuous.
    """
    def __init__(self, prefixes, frames, device, *, inverse_gains, delivery_delay_samples=0):
        if (type(delivery_delay_samples) is not int
                or not 0 <= delivery_delay_samples < VS.SPF24):
            raise ValueError('delivery_delay_samples must be an integer within one audio frame')
        self.delivery_delay_samples = delivery_delay_samples
        self.device = torch.device(device)
        self.batch = len(prefixes)
        self.frames = int(frames)
        lengths = [len(w) for w in prefixes]
        if any(n % VS.SPF24 for n in lengths):
            raise ValueError('A real audio prefill must contain complete 80 ms frames')
        self.prefills = np.asarray(lengths, np.int64) // VS.SPF24
        self.prefills_gpu = torch.as_tensor(self.prefills, device=device)
        self.available = np.asarray(lengths, np.int64) + self.delivery_delay_samples
        self.gains = torch.as_tensor(inverse_gains, dtype=torch.float32, device=device)[:, None]
        if (self.gains.shape != (self.batch, 1) or not torch.isfinite(self.gains).all()
                or (self.gains <= 0).any()):
            raise ValueError('One finite positive recorded inverse gain is required per speaker')
        self.buffer = torch.zeros(self.batch, VS.CTX24 + (frames + 8) * VS.SPF24 + VS.LOOK24
                                  + self.delivery_delay_samples,
                                  device=device, dtype=torch.float32)
        prefix_start = VS.CTX24 + self.delivery_delay_samples
        for i, wav in enumerate(prefixes):
            self.buffer[i, prefix_start:prefix_start + len(wav)].copy_(torch.as_tensor(wav, device=device))
        self.indices = torch.arange(VS.CTX24 + VS.SPF24 + VS.LOOK24, device=device)

    def append(self, frame, waveform):
        """Deliver one sender frame; preserve the shifted raw prefill exactly."""
        frame = int(frame)
        start = frame * VS.SPF24 + self.delivery_delay_samples
        end = start + VS.SPF24
        active = frame >= self.prefills
        if np.any(active & (self.available != start)):
            raise ValueError(f'Non-contiguous generated audio at frame {frame}: {self.available}')
        if waveform.shape != (self.batch, VS.SPF24):
            raise ValueError('Mimi must emit exactly one frame per stream')
        wave = (waveform.to(self.device) * self.gains).clamp(-.995, .995)
        target = self.buffer[:, VS.CTX24 + start:VS.CTX24 + end]
        target.copy_(torch.where((frame >= self.prefills_gpu)[:, None], wave, target))
        self.available = np.where(active, end, self.available)

    def require(self, frame):
        need = (int(frame) + 1) * VS.SPF24 + VS.LOOK24
        if np.any(self.available < need):
            raise ValueError(f'Student frontend frame {frame} would hear ungenerated audio: '
                             f'need {need}, have {self.available}')

    def waveform(self, instance, frames):
        """Return receiver-time PCM, including its initial delivery-delay zeros."""
        return self.buffer[instance, VS.CTX24:VS.CTX24 + frames * VS.SPF24].detach().cpu().numpy().copy()


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


def _block_frontend_embedding(vx, mel, count):
    """Convolve a causal block and retain the state before each 80 ms frame.

    conv1 consumes eight mel positions per model frame; conv2 consumes the
    same eight conv1 outputs with stride two. Snapshots include the preceding
    two mel positions and one conv1 output, exactly as sequential embed().
    """
    if mel.shape[-1] != count * VS.MPF:
        raise ValueError('A frontend block requires exactly eight mel frames per model frame')
    mel_inputs = torch.cat((vx.mel_prev, mel.to(vx.mel_prev.dtype)), -1)
    mel_states = mel_inputs.unfold(-1, 2, VS.MPF)[..., :count, :].permute(2, 0, 1, 3).contiguous()
    embedder = vx.enc.embedder
    conv1 = F.gelu(F.conv1d(mel_inputs, embedder.conv1.weight, embedder.conv1.bias))
    conv2_inputs = torch.cat((vx.c1_prev, conv1), -1)
    conv_states = conv2_inputs.unfold(-1, 1, VS.MPF)[..., :count, :].permute(2, 0, 1, 3).contiguous()
    conv2 = F.gelu(F.conv1d(conv2_inputs, embedder.conv2.weight, embedder.conv2.bias, stride=2))
    vx.mel_prev.copy_(mel_inputs[..., -2:])
    vx.c1_prev.copy_(conv2_inputs[..., -1:])
    return conv2.permute(0, 2, 1), mel_states, conv_states


def _capture_front_block(fn, vx, device):
    """Capture at the current cursor without advancing the live frontend.

    Warmups overwrite only the block's KV suffix, which is immediately
    overwritten again by the real execution. The committed KV prefix is never
    copied or changed. The small convolution histories/cursor are restored
    before each warmup, before capture, and before returning to the caller.
    """
    with torch.cuda.device(device):
        targets = (vx.kd, vx.mel_prev, vx.c1_prev)
        saved = tuple(value.clone() for value in targets)
        def restore():
            for target, value in zip(targets, saved):
                target.copy_(value)
        stream = torch.cuda.Stream(device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                restore()
                fn()
        torch.cuda.current_stream(device).wait_stream(stream)
        torch.cuda.synchronize(device)
        restore()
        stream.wait_stream(torch.cuda.current_stream(device))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            fn()
        torch.cuda.synchronize(device)
        restore()
    return graph


class StreamingFront:
    """Static batched frontend with a small convolution-state rewind ring.

    Temporal KV suffixes are overwritten at explicit absolute cache positions.
    They are never copied on rewind. The student's exact encoder outputs are
    recorded for both differentiable replay and audio-teacher scoring.
    """
    def __init__(self, front, engine, timeline, *, fa, fused_ops, frames, rewind=8, graphs=True,
                 cache_mode='paged'):
        self.engine, self.timeline = engine, timeline
        self.device, self.frames, self.rewind_window = engine.d0, int(frames), int(rewind)
        self.cache_frames = max(self.frames + 8, int(getattr(engine, 'n_fr', self.frames + 8)))
        self.vx = VoxStreamFast(front.encoder, engine.NI, self.cache_frames, self.device,
                               fa, fused_ops, front.fe, cache_mode=cache_mode,
                               rewind_frames=rewind, max_block_frames=rewind)
        self.cache_mode = cache_mode
        self.latents = torch.zeros(engine.NI, 4 * (frames + 8), self.vx.Hd,
                                   device=self.device, dtype=torch.bfloat16)
        self._mel = torch.zeros(rewind, *self.vx.mel_prev.shape, device=self.device, dtype=torch.bfloat16)
        self._conv = torch.zeros(rewind, *self.vx.c1_prev.shape, device=self.device, dtype=torch.bfloat16)
        self._saved = [-1] * rewind
        self._ar4 = torch.arange(4, device=self.device)
        self._row = torch.arange(engine.NI, device=self.device)
        self._graphs_enabled = bool(graphs)
        self._block_graphs = {}
        self._block_indices = {}
        self._block_frames = {}
        self._block_encoder_frames = {}
        self.graph = None
        if graphs:
            self.vx.kd.fill_(1)
            self.graph = _capture(lambda: self._run(False), self.device)
        self.vx.reset()

    def _run(self, first):
        frame = self.vx.kd.clone()
        slot = frame.remainder(self.rewind_window)
        self._mel.index_copy_(0, slot, self.vx.mel_prev[None])
        self._conv.index_copy_(0, slot, self.vx.c1_prev[None])
        window = self.timeline.buffer.index_select(1, frame * VS.SPF24 + self.timeline.indices)
        if first:
            lat = self.vx.encode(window, 1)
        else:
            lat = self.vx.step_graphable(window)
        self.latents.index_copy_(1, frame * 4 + self._ar4, lat)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            features = self.engine.m.vgain(self.engine.m.vadapter(lat))[:, 0]
        positions = self.engine.Pv[self.device][:self.engine.NI] + frame
        self.engine._opd_audio[self._row, positions] = features

    @torch.no_grad()
    def step(self, frame, *, speculative=False):
        frame = int(frame)
        if frame != self.vx.k:
            raise ValueError(f'Frontend is at {self.vx.k}, requested {frame}')
        if not speculative:
            self.timeline.require(frame)
        with torch.cuda.device(self.device):
            if frame == 0:
                self._run(True)
            elif self.graph is None:
                self._run(False)
                self.vx.k += 1
            else:
                self.graph.replay()
                self.vx.k += 1
        self._saved[frame % self.rewind_window] = frame

    def _run_block(self, count):
        frame = self.vx.kd.clone()
        window = self.timeline.buffer.index_select(1, frame * VS.SPF24 + self._block_indices[count])
        mel = self.vx.mel(window, count, False)
        embedding, mel_states, conv_states = _block_frontend_embedding(self.vx, mel, count)
        frames = frame + self._block_frames[count]
        slots = frames.remainder(self.rewind_window)
        self._mel.index_copy_(0, slots, mel_states)
        self._conv.index_copy_(0, slots, conv_states)
        # FA3 appends all24 K/V vectors at once and applies the same causal
        # sliding mask inside this block; no future encoder frame is visible.
        lat = self.vx.layers(embedding, VS.EPF * count)
        self.vx.kd.add_(count)
        self.latents.index_copy_(1, frame * VS.EPF + self._block_encoder_frames[count], lat)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            features = self.engine.m.vgain(self.engine.m.vadapter(lat))
        positions = self.engine.Pv[self.device][:self.engine.NI, None] + frames[None]
        self.engine._opd_audio[self._row[:, None], positions] = features

    @torch.no_grad()
    def step_block(self, start, count=6, *, speculative=True):
        """Advance a fixed causal block, retaining every internal rewind point.

        The initial reflected frontend frame is handled by step(0). Teacher
        refreshes start later, so each distinct block length uses one static
        CUDA graph. Count cannot exceed the convolution-state rewind ring.
        """
        start, count = int(start), int(count)
        if start < 1 or start != self.vx.k:
            raise ValueError(f'Frontend block must start at its current positive cursor {self.vx.k}, got {start}')
        if not 1 <= count <= self.rewind_window or start + count > self.frames + 8:
            raise ValueError('Frontend block exceeds its rewind window or allocated timeline')
        if not speculative:
            self.timeline.require(start + count - 1)
        with torch.cuda.device(self.device):
            if count not in self._block_indices:
                self._block_indices[count] = torch.arange(VS.CTX24 + count * VS.SPF24 + VS.LOOK24,
                                                          device=self.device)
                self._block_frames[count] = torch.arange(count, device=self.device)
                self._block_encoder_frames[count] = torch.arange(VS.EPF * count, device=self.device)
                self._block_graphs[count] = (_capture_front_block(lambda: self._run_block(count), self.vx, self.device)
                                             if self._graphs_enabled else None)
            graph = self._block_graphs[count]
            if graph is None:
                self._run_block(count)
            else:
                graph.replay()
            self.vx.k += count
        for frame in range(start, start + count):
            self._saved[frame % self.rewind_window] = frame

    @torch.no_grad()
    def rewind(self, frame):
        frame = int(frame)
        slot = frame % self.rewind_window
        if self._saved[slot] != frame:
            raise ValueError(f'Frontend rewind state for {frame} was not retained')
        with torch.cuda.device(self.device):
            self.vx.mel_prev.copy_(self._mel[slot])
            self.vx.c1_prev.copy_(self._conv[slot])
            self.vx.kd.fill_(frame)
        self.vx.k = frame

    def recorded(self, instance, frames):
        return self.latents[instance:instance + 1, :4 * frames].detach().cpu()

    def cache_report(self):
        return {'voxtral_cache_mode': self.cache_mode,
                'voxtral_cache_capacity_frames': self.cache_frames,
                'voxtral_logical_encoder_positions': self.vx.maxT,
                'voxtral_cache_encoder_positions': self.vx.cache_capacity_positions,
                'recorded_capacity_frames': self.latents.shape[1] // VS.EPF,
                'voxtral_kv_gib': sum(value.numel() * value.element_size()
                                      for value in self.vx.K + self.vx.V) / 2 ** 30}


class StreamingDecoder:
    """One graph and one causal Mimi state per batched output direction."""
    def __init__(self, mimi, batch, device, *, graphs=True):
        self.device = torch.device(device)
        self.decoder = MimiDecStream(mimi, batch, self.device)
        self.codes = torch.zeros(batch, 32, dtype=torch.long, device=self.device)
        self.offsets = torch.arange(32, device=self.device) * 2048
        self.output = torch.empty(batch, VS.SPF24, device=self.device)
        self.graph = _capture(self._run, self.device) if graphs else None
        self.decoder.reset()
        self.next_frame = 0

    def _run(self):
        self.output.copy_(self.decoder.step(self.codes))

    @torch.no_grad()
    def decode(self, frame, global_codes):
        if int(frame) != self.next_frame:
            raise ValueError(f'Mimi output is append-only: wanted {self.next_frame}, got {frame}')
        with torch.cuda.device(self.device):
            self.codes.copy_(global_codes.to(self.device) - self.offsets)
            if self.graph is None:
                self._run()
            else:
                self.graph.replay()
        self.next_frame += 1
        return self.output
