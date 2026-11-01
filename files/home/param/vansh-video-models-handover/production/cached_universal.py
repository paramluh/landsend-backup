"""
inference/pose_universal.py's sampler run frame -> band -> substep (same t, same windows, same trajectories) with the
left context read from per-layer K/V caches instead of being run through the backbone again: a step runs only
the moving frame's N = 120 tokens. See cached.md.

All caches live in one pool of per-layer (K rotated at the slot it is read from, V), bf16 as the full window
feeds FlashAttention:
- sink: slot 0, t = 0, no pose; one solo pass per clip.
- history: after band 0, one solo pass [sink | frame] with q and k at the history anchor (slot H). The hidden
  states are context-free, but present queries read the frame's K at the slot it currently occupies, which moves
  left at every commit, so the commit stores K rotated at every slot 1..H it will occupy.
- present (frame, band, s): the moving frame's own K/V at that step, rotated at its true slot 1 + H + band. Frame
  frame + j reads it at band band + j, same s, from the same slot with the same left context.
- pads: a startup hole is the learned pad token at t = 0 without pose or audio. History pads are context-free like
  history; a present pad at slot j sees only the sink and pads. Both are computed once per clip.

The moving frame at band b attends [sink | slots b+1 .. H+b | self] (H + 2 frames), TransformerLayer's present
rule. The code table slides as in training (common.sliding_code_table); a window's last real frame is the
wavefront's, min(k + P - 1 - b, L - 1), so the trajectories equal inference/pose_universal.py's for clips of any
length. The encoded table is cached per table start; before each band the moving frame's audio window (its K tokens
of the table and the null token) is copied into a static buffer that the band's steps read.
Not bit-identical to the full window: the GEMMs see 120 rows instead of the window's.

Speed (benchmark.md: whole-step compile + one CUDA graph is the fastest step): the step and the commit are pure
functions of static buffers (state, pose, context and write indices, audio window) that return the velocity and
the K/V to store; a frame's own entries are never in its context, so the pool writes can follow the pass. Each is
compiled (dynamic=False) and captured, with its pool writes (and the step's Euler update), as one CUDA
graph. Every step's indices come from a per-clip schedule on the GPU: the loop is device copies and replays.

Audio guidance (as vidcon-base's pose_vidcon.py): on a frame's first audio_guidance_steps steps (the noisiest),
v = v_noaudio + audio_guidance_scale * (v_audio - v_noaudio), where v_noaudio replaces the moving frame's audio
window with the learned no_audio_tokens (training's audio dropout), and its hiddens, if any, with the hidden drop
token. It reads the same cache, whose present K/V saw the
real audio, and stores nothing; scale 1.0 or 0 steps is off.

Batching (batch_size clips per process): every buffer has a leading clip dim and clip b owns pool entries
b * entries + e, so the schedule is shared and only the audio is per clip: each clip slides and encodes its own code
table into its own row of buf_audio. A batch runs to its longest clip; a shorter clip repeats its last frame's audio
past its end and keeps only its own frames, which never read later ones.
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault('TORCHINDUCTOR_FX_GRAPH_CACHE', '1')

import torch
import yaml
from flash_attn_interface import flash_attn_func
from safetensors.torch import save_file
from torch import nn
from torch._inductor.runtime import triton_heuristics

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataloading.sources import AUDIO_CODE_QUANTUM
from inference.common import (
    FPS, decode_wan22_vae, find_samples, load_sample, resolve_path, run_parallel, side_by_side, sliding_code_table,
)
from inference.loader import build_models
from inference.lora import cast_inference_weights, project_qkv
from models.pose import rope
from models.pose.audio import audio_latents_from_codes
from models.pose.wan_universal import WINDOW_OFFSET
from models.wan.model import sinusoidal_embedding_1d
from utils.common import AUTOCAST_DTYPE

TAG = 'cached'
SCRATCH, SINK = 0, 1  # SCRATCH takes the K/V of steps nothing reads (band P-1), so the step has one shape


def _first_config(self, *args, **kwargs):
    # the first candidate Triton config instead of the fastest, so kernels are identical on every node and process
    self.launchers = self.launchers[:1]
    if self.save_cache_hook:
        self.save_cache_hook(self.launchers[0].config, 0, triton_cache_hash=self.launchers[0].cache_hash)


triton_heuristics.CachingAutotuner.autotune_to_one_config = _first_config
torch.compiler.config.cache_key_tag = 'first-config'


def audio_start(f_last):
    return max(0, f_last - (AUDIO_CODE_QUANTUM // 2 - 1))


class CachedNet:
    def __init__(self, models, latent_hw, substeps, device, compile, guidance_scale=1.0, batch=1):
        layers = models.net.layers
        self.initial, self.blocks, self.final = layers[0], list(layers[1:-1]), layers[-1]
        self.mimi, self.num_codebooks = models.mimi, models.num_codebooks
        self.H, self.P, self.m = models.num_history, models.num_present, substeps
        self.R = self.initial.num_refs
        H, P, m, R = self.H, self.P, substeps, self.R
        self.num_ctx = R + 1 + H  # a moving frame's keys before itself: refs, sink, H frames
        attention = self.blocks[0].block.self_attn
        self.n, self.d = attention.num_heads, attention.head_dim
        patch = self.initial.patch_embedding.kernel_size
        self.grid = (1, latent_hw[0] // patch[1], latent_hw[1] // patch[2])
        self.N = self.grid[1] * self.grid[2]
        self.K = self.initial.audio_cross_attention_codes
        self.device, self.B = device, batch
        self.dt = -1.0 / (P * m)
        self.guidance_scale = guidance_scale

        # Autocast casts fp32 weights to bf16 on every call; casting the Linear/Conv weights of the fp32 modules
        # once gives the same bf16 operands. Their RMSNorm weights stay fp32, as autocast leaves them.
        hidden_blocks = [l.hidden_block for l in self.blocks if l.hidden_block is not None]
        for module in (*[l.audio_block for l in self.blocks if l.audio_block is not None], *hidden_blocks,
                       self.initial.pose_patch_embedding):
            cast_inference_weights(module)

        freqs = self.blocks[0].freqs
        freqs_i = rope.cached_freqs_i({}, (1 + H + P, *self.grid[1:]), freqs, WINDOW_OFFSET if R else 0)
        self.slot_freqs = freqs_i.unflatten(0, (1 + H + P, self.N))  # [slot, N, 1, c, 2]
        self.ref_freqs = rope.cached_freqs_i({}, (1, *self.grid[1:]), freqs)[None]
        self.anchor_freqs = self.slot_freqs[H:H + 1]
        self.history_freqs = self.slot_freqs[1:1 + H]  # a committed frame's K at every slot it will occupy
        self.present_freqs = self.slot_freqs[1 + H:]

        # Pool entries: scratch, sink, refs, history pads (slots 1..H), present pads (slots 0..P-2), history ring
        # (H + 1 frames x H slots), present ring (P frames x (P - 1) bands x m substeps).
        self.fixed = [*range(2, 2 + R), SINK]  # every non-ref frame's first keys, in training's order
        self.pad_history0 = 2 + R
        self.pad_present0 = self.pad_history0 + H
        self.history0 = self.pad_present0 + P - 1
        self.present0 = self.history0 + (H + 1) * H
        entries = self.present0 + P * (P - 1) * m
        shape = (len(self.blocks), batch * entries, self.N, self.n, self.d)
        self.pool_k = torch.empty(shape, dtype=torch.bfloat16, device=device)
        self.pool_v = torch.empty(shape, dtype=torch.bfloat16, device=device)
        self.offsets = torch.arange(batch, device=device)[:, None] * entries
        print(f'[{TAG}] K/V pool: {batch} x {entries} entries, {2 * self.pool_k.numel() * 2 / 2**30:.1f} GiB')

        t = torch.tensor([(b + 1) / P + s * self.dt for b in range(P) for s in range(m)], dtype=torch.float32)
        self.t_table = (t * 1000).to(device)  # the window's fp32 t, then * 1000, as pose_universal.py feeds it
        self.zero_t = torch.zeros(1, 1, device=device)

        C = self.initial.patch_embedding.in_channels
        self.buf_latent = torch.zeros(batch, C, *latent_hw, device=device)  # the moving frames, Euler-updated in place
        self.buf_pose = torch.zeros(batch, C, *latent_hw, device=device)
        # Clip-local entries shared by the batch; + offsets gives each clip's own.
        self.buf_ints = torch.zeros(self.num_ctx + 3, dtype=torch.long, device=device)  # ctx, write, band, s
        self.buf_store = torch.zeros(H, dtype=torch.long, device=device)  # the committing frame's history entries
        self.sink_ctx = torch.tensor(self.fixed, device=device) + self.offsets
        self.buf_audio = torch.zeros(batch, 1, self.K + 1, self.initial.dim, dtype=AUTOCAST_DTYPE, device=device)
        self.audio_cache = {}
        # With hidden blocks: per clip the chunk's Orpheus hiddens [P, 4096] (set per batch); the moving frame f
        # reads positions {2f - 1, 2f} from buf_hidden.
        self.hiddens = None
        self.buf_hidden = torch.zeros(batch, 1, 2, hidden_blocks[0].norm_kv_in.normalized_shape[0],
                                      dtype=torch.bfloat16, device=device) if hidden_blocks else None
        self.clips, self.token_offsets = torch.arange(batch, device=device)[:, None], torch.arange(self.K, device=device)
        self.moving = torch.compile(self._moving, dynamic=False) if compile else self._moving
        self.moving_noaudio = torch.compile(self._moving_noaudio, dynamic=False) if compile else self._moving_noaudio
        self.committing = torch.compile(self._commit, dynamic=False) if compile else self._commit
        self.graphs = None

    def history_entries(self, frame):
        return range(self.history0 + (frame % (self.H + 1)) * self.H, self.history0 + (frame % (self.H + 1) + 1) * self.H)

    def present_entry(self, frame, band, s):
        return self.present0 + ((frame % self.P) * (self.P - 1) + band) * self.m + s

    def _frame(self, latent, pose, pad, t, q_freqs, ctx, store_freqs, audio, head, embedding=None, hidden=None):
        """
        One frame per clip through the network against cached keys. latent / pose [B, C, h, w] (pose None: masked),
        pad: the learned pad token instead of the latent (bool, or per clip [B]), embedding: the patch embedding
        (default the transformer's), t [1, 1] already * 1000, q_freqs [1, N, 1, c, 2] for q and
        the frame's own k, ctx [B, num] pool entries read in slot order before self, audio [B, 1, K + 1, D] or None,
        hidden [B, 1, 2, 4096] (present frames only) or None.
        Returns (velocity [B, C, h, w] if head, K to store [layers, B, E, N, n, d] rotated at store_freqs
        [E, N, 1, c, 2] or at q_freqs (E = 1) when None, V [layers, B, N, n, d]).
        """
        initial, N, n, d, B = self.initial, self.N, self.n, self.d, latent.shape[0]
        ks, vs = [], []
        with torch.autocast('cuda', dtype=AUTOCAST_DTYPE, cache_enabled=False):
            embedding = initial.patch_embedding if embedding is None else embedding
            x = embedding(latent[:, :, None]).flatten(2).transpose(1, 2)
            if isinstance(pad, torch.Tensor):
                x = torch.where(pad[:, None, None], initial.pad_token.to(x.dtype), x)
            elif pad:
                x = initial.pad_token.to(x.dtype).expand_as(x)
            if pose is not None:
                x = x + initial.pose_patch_embedding(pose[:, :, None].to(x.dtype)).flatten(2).transpose(1, 2)
            e = initial.time_embedding(sinusoidal_embedding_1d(initial.freq_dim, t.flatten()).unflatten(0, t.shape))
            e0 = initial.time_projection(e).unflatten(2, (6, initial.dim))
            x = x.unflatten(1, (1, N))

            for i, layer in enumerate(self.blocks):
                block, attention = layer.block, layer.block.self_attn
                modulation = (block.modulation.unsqueeze(0) + e0).chunk(6, dim=2)
                normalized = (block.norm1(x) * (1 + modulation[1]) + modulation[0]).flatten(1, 2)
                q, k, v = project_qkv(attention, normalized)
                q = attention.norm_q(q).view(B, N, n, d)
                k = attention.norm_k(k).view(B, N, n, d)
                v = v.view(B, N, n, d).bfloat16()
                q = rope.apply_rope(q, q_freqs)
                k_self = rope.apply_rope(k, q_freqs).bfloat16()
                keys, values = k_self, v
                if ctx is not None:
                    keys = torch.cat((self.pool_k[i].index_select(0, ctx.flatten()).view(B, -1, n, d), k_self), dim=1)
                    values = torch.cat((self.pool_v[i].index_select(0, ctx.flatten()).view(B, -1, n, d), v), dim=1)
                output = flash_attn_func(q.bfloat16(), keys, values).to(q.dtype)
                if store_freqs is None:
                    ks.append(k_self[:, None])
                else:
                    E = store_freqs.shape[0]
                    ks.append(rope.apply_rope(k.repeat(1, E, 1, 1), store_freqs.flatten(0, 1)).bfloat16().view(B, E, N, n, d))
                vs.append(v)
                x = x + attention.o(output.flatten(2)).unflatten(1, (1, N)) * modulation[2]

                if audio is not None and layer.audio_block is not None:
                    x = x + layer.audio_block.forward_present(x, audio)
                if hidden is not None and layer.hidden_block is not None:
                    x = x + layer.hidden_block.forward_present(x, hidden)

                residual = block.ffn(block.norm2(x) * (1 + modulation[4]) + modulation[3])
                x = x + residual * modulation[5]

            ks, vs = torch.stack(ks), torch.stack(vs)
            if not head:
                return None, ks, vs
            final = self.final.head
            with torch.autocast('cuda', dtype=torch.float32, cache_enabled=False):
                modulation = (final.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
                x = final.head(final.norm(x) * (1 + modulation[1]) + modulation[0])
        return self.final.unpatchify(x.flatten(1, 2), self.grid)[:, :, 0].float(), ks, vs

    def _store(self, store, ks, vs):
        """store [B, E] pool entries (offsets included)."""
        self.pool_k.index_copy_(1, store.flatten(), ks.flatten(1, 2))
        self.pool_v.index_copy_(1, store.flatten(), vs.unsqueeze(2).expand_as(ks).flatten(1, 2))

    def _moving(self, noaudio=False):
        """The step every band/substep shares, a pure function of the static buffers (compiled)."""
        c, ints = self.num_ctx, self.buf_ints
        band, s = ints[c + 1:c + 2], ints[c + 2:c + 3]
        audio, hidden = self.buf_audio, self.buf_hidden
        if noaudio:  # training's dropped window: no-audio tokens replace audio's K tokens + null, the drop token hiddens
            audio = self.initial.audio_encoder.no_audio_tokens.to(self.buf_audio.dtype).expand(self.B, -1, -1, -1)
            if hidden is not None:
                hidden = self.initial.hidden_drop_token.to(hidden.dtype).expand(self.B, -1, -1, -1)
        t = self.t_table.index_select(0, band * self.m + s).view(1, 1)
        return self._frame(
            self.buf_latent, self.buf_pose, False, t, self.present_freqs.index_select(0, band),
            ints[:c] + self.offsets, None, audio, head=True, hidden=hidden,
        )

    def _moving_noaudio(self):
        return self._moving(noaudio=True)[0]

    def _commit(self):
        """The solo step [refs | sink | frame] at the history anchor; K for every history slot."""
        return self._frame(self.buf_latent, self.buf_pose, False, self.zero_t, self.anchor_freqs, self.sink_ctx,
                           self.history_freqs, None, head=False)

    def _step(self):
        v, ks, vs = self.moving()
        self._store(self.buf_ints[self.num_ctx:self.num_ctx + 1] + self.offsets, ks, vs)
        self.buf_latent.copy_(self.buf_latent + v * self.dt)

    def _guided_step(self):
        """v = v_noaudio + w (v_audio - v_noaudio); both read the same cache, only the audio pass's K/V is stored."""
        v, ks, vs = self.moving()
        v_noaudio = self.moving_noaudio()
        self._store(self.buf_ints[self.num_ctx:self.num_ctx + 1] + self.offsets, ks, vs)
        v = v_noaudio + self.guidance_scale * (v - v_noaudio)
        self.buf_latent.copy_(self.buf_latent + v * self.dt)

    def _commit_step(self):
        _, ks, vs = self.committing()
        self._store(self.buf_store + self.offsets, ks, vs)

    def step(self, guided=False):
        if self.graphs is None:
            return self._guided_step() if guided else self._step()
        self.graphs[2 if guided else 0].replay()

    def commit(self):
        if self.graphs is None:
            return self._commit_step()
        self.graphs[1].replay()

    def _capture(self, fn):
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):  # the first call compiles
                fn()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        return graph

    @torch.no_grad()
    def start_clip(self, sink, cuda_graph):
        """sink [B, C, h, w]: each clip's frame 0."""
        H, P, device = self.H, self.P, self.device
        idx = lambda values: torch.tensor(list(values), dtype=torch.long, device=device) + self.offsets
        self.audio_cache.clear()
        _, ks, vs = self._frame(sink, None, False, self.zero_t, self.slot_freqs[:1], None, None, None, head=False)
        self._store(idx([SINK]), ks, vs)
        pad_history = range(self.pad_history0, self.pad_history0 + H)
        _, ks, vs = self._frame(sink, None, True, self.zero_t, self.anchor_freqs, self.sink_ctx, self.history_freqs,
                                None, head=False)
        self._store(idx(pad_history), ks, vs)
        for j in range(P - 1):
            ctx = [*self.fixed, *pad_history[j:], *range(self.pad_present0 + max(0, j - H), self.pad_present0 + j)]
            _, ks, vs = self._frame(sink, None, True, self.zero_t, self.present_freqs[j:j + 1], idx(ctx), None, None,
                                    head=False)
            self._store(idx([self.pad_present0 + j]), ks, vs)
        if cuda_graph and self.graphs is None:
            self.buf_ints.copy_(torch.tensor([SINK] * self.num_ctx + [SCRATCH, 0, 0], device=device))
            self.buf_store.fill_(SCRATCH)
            start = time.time()
            self.graphs = [self._capture(self._step), self._capture(self._commit_step)]
            if self.guidance_scale != 1.0:
                self.graphs.append(self._capture(self._guided_step))
            print(f'[{TAG}] {len(self.graphs)} graphs captured in {time.time() - start:.0f}s', flush=True)

    def schedule(self, L):
        """Per step in sampling order: [ctx (R + 1 + H), write, band, s]; per frame: its history entries."""
        H, P, m = self.H, self.P, self.m
        rows = []
        for frame in range(1, L):
            for band in range(P - 1, -1, -1):
                first = frame - band
                history = []
                for slot in range(band + 1, H + 1):
                    f = first - (H + 1 - slot)
                    history.append(self.history_entries(f)[slot - 1] if f >= 1 else self.pad_history0 + slot - 1)
                for s in range(m):
                    present = [self.present_entry(first + j, j, s) if first + j >= 1 else self.pad_present0 + j
                               for j in range(max(0, band - H), band)]
                    write = self.present_entry(frame, band, s) if band < P - 1 else SCRATCH
                    rows.append([*self.fixed, *history, *present, write, band, s])
        commits = [list(self.history_entries(frame)) for frame in range(1, L)]
        return (torch.tensor(rows, dtype=torch.long, device=self.device),
                torch.tensor(commits, dtype=torch.long, device=self.device))

    def audio_f_last(self, frame, band, length):
        """The last real frame of the window the moving frame sees at band; past the clip's length, its last frame's."""
        return min(min(frame, length - 1) + self.P - 1 - band, length - 1)

    def audio_schedule(self, L, length):
        """Per band in sampling order: the moving frame's index in its code table (frame - a), for set_audio."""
        P = self.P
        rows = [min(frame, length - 1) - audio_start(self.audio_f_last(frame, band, length))
                for frame in range(1, L) for band in range(P - 1, -1, -1)]
        return torch.tensor(rows, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def set_audio(self, code_tables, codes, f_lasts, audio_frames):
        """
        Per clip: the encoded code table of the window whose last real frame is f_lasts[b] (cached per table starts),
        and the moving frame's K tokens (audio_frames [B]) and the null token -> buf_audio[b].
        """
        a = tuple(audio_start(f) for f in f_lasts)
        if a not in self.audio_cache:
            for old in [key for key in self.audio_cache if any(o < s - self.P for o, s in zip(key, a))]:
                del self.audio_cache[old]
            tables = [sliding_code_table(table, c, f)[0] for table, c, f in zip(code_tables, codes, f_lasts)]
            latents = audio_latents_from_codes(self.mimi, torch.cat(tables), self.num_codebooks)
            with torch.autocast('cuda', dtype=AUTOCAST_DTYPE):
                self.audio_cache[a] = self.initial.audio_encoder(latents)
        tokens = self.audio_cache[a]
        positions = self.K * (audio_frames[:, None] - 1) + self.token_offsets
        null = self.initial.audio_encoder.null_token[0].to(tokens.dtype).expand(self.B, -1, -1)
        self.buf_audio.copy_(torch.cat((tokens[self.clips, positions], null), dim=1)[:, None])


    def set_hidden(self, frame, lengths):
        """The moving frame's hiddens per clip; past a clip's length, its last frame's."""
        for b, (hiddens, length) in enumerate(zip(self.hiddens, lengths)):
            f = min(frame, length - 1)
            self.buf_hidden[b, 0].copy_(hiddens[2 * f - 1:2 * f + 1])


@torch.no_grad()
def generate(cached, clips, seeds, device, cuda_graph, guidance_steps=0, noise_frames=None, on_commit=None):
    """
    clips: up to cached.B (latents, pose, code_table, codes), one batch run to the longest (a short batch is filled
    with copies of its last clip). guidance_steps: the frame's first (noisiest) steps that run the audio-guided step.
    noise_frames: per clip, draw the noise for this many frames and keep the first L, so a truncated clip gets the
    full clip's noise (randn's layout is [C, L, ...]). on_commit(frame, latents [B, C, h, w]) runs as each frame is
    committed (its time counts in the timings). Returns each clip's latents [C, L, h, w].
    """
    H, P, m, N, B = cached.H, cached.P, cached.m, cached.N, cached.B
    guided = min(guidance_steps, P * m) if cached.guidance_scale != 1.0 else 0
    count, fill = len(clips), B - len(clips)
    noise_frames = noise_frames or [None] * count
    clips, seeds = clips + clips[-1:] * fill, seeds + seeds[-1:] * fill
    noise_frames = noise_frames + noise_frames[-1:] * fill
    lengths = [latents.shape[1] for latents, _, _, _ in clips]
    if min(lengths) < 2:
        raise RuntimeError(f'clip has {min(lengths)} conditioned frames, need >= 2')
    C, _, h, w = clips[0][0].shape
    L = max(lengths)
    noise = torch.zeros(B, C, L, h, w, device=device)
    pose = torch.zeros(B, C, L, h, w, device=device)
    for b, ((latents, clip_pose, _, _), seed, frames) in enumerate(zip(clips, seeds, noise_frames)):
        generator = torch.Generator(device=device).manual_seed(seed)
        l = lengths[b]
        noise[b, :, :l] = torch.randn(C, max(l, frames or l), h, w, generator=generator, device=device)[:, :l]
        pose[b, :, :l] = clip_pose.to(device)
    sink = torch.stack([latents[:, 0] for latents, _, _, _ in clips]).to(device)
    code_tables = [code_table.to(device) for _, _, code_table, _ in clips]
    codes = [c.to(device) for _, _, _, c in clips]
    cached.start_clip(sink, cuda_graph)
    rows, commits = cached.schedule(L)
    audio_frames = torch.stack([cached.audio_schedule(L, l) for l in lengths], dim=1)

    history = [sink]
    torch.cuda.synchronize()
    start = time.time()
    row = 0
    for frame in range(1, L):
        cached.buf_latent.copy_(noise[:, :, frame])
        cached.buf_pose.copy_(pose[:, :, frame])
        if cached.buf_hidden is not None:
            cached.set_hidden(frame, lengths)
        for band in range(P - 1, -1, -1):
            cached.set_audio(code_tables, codes, [cached.audio_f_last(frame, band, l) for l in lengths],
                             audio_frames[(frame - 1) * P + P - 1 - band])
            for s in range(m):
                cached.buf_ints.copy_(rows[row])
                row += 1
                cached.step((P - 1 - band) * m + s < guided)
        cached.buf_store.copy_(commits[frame - 1])
        cached.commit()
        history.append(cached.buf_latent.clone())
        if on_commit is not None:
            on_commit(frame, history[-1])
        if frame % 100 == 0:
            torch.cuda.synchronize()
            elapsed = time.time() - start
            print(f'[{TAG}] {frame}/{L - 1} frames, {elapsed:.0f}s, {1000 * elapsed / frame:.0f} ms/frame', flush=True)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    frames, passes = L - 1, P * m + guided + 1
    print(f'[{TAG}] {B} x {frames} frames in {elapsed:.1f}s: {1000 * elapsed / frames:.0f} ms/frame '
          f'({1000 * elapsed / (frames * B):.0f} per clip frame), '
          f'{1000 * elapsed / (frames * passes):.2f} ms/pass ({P * m} steps + {guided} no-audio + commit), '
          f'{B * frames * passes * N / elapsed:,.0f} query tokens/s, RTF {elapsed / (frames * 4 / FPS):.2f}', flush=True)
    generated = torch.stack(history, dim=2)
    return [generated[b, :, :lengths[b]] for b in range(count)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', nargs='?', default='cached_universal.yaml')
    parser.add_argument('--parallel', action='store_true', help='one worker per visible GPU')
    parser.add_argument('--shard', type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--num-shards', type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    config_path = Path(__file__).parent / args.config
    config = yaml.safe_load(open(config_path))
    base_dir = config_path.parent
    samples_dir = Path(resolve_path(config['samples_dir'], base_dir))
    output_dir = Path(resolve_path(config['output_dir'], base_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.shard in (None, 0):
        shutil.copy2(config_path, output_dir / config_path.name)

    nums = find_samples(samples_dir, config['max_samples'], config['sample_seed'])
    if args.parallel:
        print(f'[{TAG}] {len(nums)} samples in {samples_dir}')
        return run_parallel(__file__, args.config, len(nums))
    tag = TAG if args.shard is None else f'{TAG}:w{args.shard}'
    if args.shard is not None:
        nums = nums[args.shard::args.num_shards]

    if config['type'] != 'universal_wide':
        raise ValueError(f"{config['type']}: caching assumes universal_wide's context-free history")
    device = 'cuda'
    models = build_models(config, base_dir, device)
    substeps, num_steps = config['substeps'], config['num_steps']
    if num_steps != substeps * models.num_present:
        raise ValueError(f'num_steps must be substeps x num_present: {num_steps} != {substeps} x {models.num_present}')
    guidance_scale, guidance_steps = config['audio_guidance_scale'], config['audio_guidance_steps']
    batch = config.get('batch_size', 1)
    cached = None
    print(f'[{tag}] {len(nums)} samples, batch {batch}, {num_steps} steps per frame ({substeps} per band), '
          f'seed {config["seed"]}, audio guidance {guidance_scale} on the first {guidance_steps} steps, '
          f'compile {config["compile"]}, cuda_graph {config["cuda_graph"]}')

    for i in range(0, len(nums), batch):
        group, clips, audios = nums[i:i + batch], [], []
        for n in group:
            latents, pose, code_table, codes = load_sample(
                samples_dir, n, models.num_codebooks, config['max_latent_frames'], slide_audio=True)
            if cached is None:
                cached = CachedNet(models, latents.shape[2:], substeps, device, config['compile'], guidance_scale, batch)
            audio = models.mimi.decode(codes[None].to(device)).audio_values[0, 0].float().cpu().numpy()
            sample_path = samples_dir / f'samples_{n}.mp4'
            if not sample_path.exists():
                decode_wan22_vae(models.vae, latents[None], str(sample_path), fps=FPS, audio_np=audio, tag=tag)
            clips.append((latents, pose, code_table, codes))
            audios.append(audio)
        outputs = generate(cached, clips, [config['seed'] + n for n in group], device, config['cuda_graph'],
                           guidance_steps)
        for n, (latents, _, _, _), audio, generated in zip(group, clips, audios, outputs):
            print(f'[{tag}] samples_{n}: {latents.shape[1]} conditioned frames -> {generated.shape[1]} latent frames')
            sample_path, out_path = samples_dir / f'samples_{n}.mp4', output_dir / f'sample_{n}.mp4'
            save_file({'latents': generated.cpu()}, str(output_dir / f'sample_{n}_latents.safetensors'))
            decode_wan22_vae(models.vae, generated[None], str(out_path), fps=FPS, audio_np=audio, tag=tag)
            side_by_side(sample_path, out_path, output_dir / f'sample_{n}_sxs.mp4', tag=tag)


if __name__ == '__main__':
    main()
