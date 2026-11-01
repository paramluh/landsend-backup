"""
Resolving-mode inference over [sink | H clean history | 1 present], frame by frame. Per frame the context runs once
(sink attends itself, each history frame the sink and itself at slot H) and keeps per-layer K at slots 0..H and V;
each denoising step runs only the present, at slot H + 1, against them.
Reference-conditioned models cache the independent reference K/V once per clip and prepend it to history and
present attention, with the trained window RoPE offset. Reference buffers retain their addresses for CUDA graphs.
"""

import copy

import torch
from flash_attn_interface import flash_attn_func

from dataloading.sources import AUDIO_CODE_QUANTUM
from models.pose import rope
from models.pose.audio import audio_latents_from_codes
from models.pose.wan_universal import WINDOW_OFFSET
from models.wan.model import sinusoidal_embedding_1d


def context_frames(f, H):
    """
    Present frame f's window ofi: sink 0, history f-H..f-1, frames below 1 as pads (-1).
    """
    return [0] + [j if j >= 1 else -1 for j in range(f - H, f)]


class ResolverNet:
    def __init__(self, models):
        layers = models.net.layers
        self.initial, self.blocks, self.final = layers[0], list(layers[1:-1]), copy.copy(layers[-1])
        self.final.num_refs = 0
        self.mimi, self.num_codebooks, self.H = models.mimi, models.num_codebooks, models.num_history
        assert models.num_present == 1 and not self.initial.hidden
        self.num_refs = self.initial.num_refs
        self.reference_context = None
        attention = self.blocks[0].block.self_attn
        self.n, self.d = attention.num_heads, attention.head_dim
        self.freqs, self.rope_cache = self.blocks[0].freqs, {}

    def embed(self, latents, pose, ofi, t):
        """
        InitialLayer's tokens and time embeddings.

        latents, pose: [B, C, F, h, w]
        ofi: [B, F], -1 for pads
        t: [B, F], * 1000
        -> x [B, F, N, D], e, e0
        """
        initial = self.initial
        x = initial.patch_embedding(latents)
        self.grid = tuple(x.shape[2:])
        F, N = self.grid[0], self.grid[1] * self.grid[2]
        x = x.flatten(2).transpose(1, 2)
        x = torch.where((ofi < 0).repeat_interleave(N, dim=1)[:, :, None], initial.pad_token.to(x.dtype), x)
        pose_tokens = initial.pose_patch_embedding(pose.to(x.dtype)).flatten(2).transpose(1, 2).unflatten(1, (F, N))
        x = x.unflatten(1, (F, N)) + pose_tokens * (ofi >= 1).to(pose_tokens.dtype)[:, :, None, None]
        e = initial.time_embedding(sinusoidal_embedding_1d(initial.freq_dim, t.flatten()).unflatten(0, t.shape))
        return x, e, initial.time_projection(e).unflatten(2, (6, initial.dim))

    def slot_freqs(self, N):
        offset = WINDOW_OFFSET if self.num_refs else 0
        return rope.cached_freqs_i(self.rope_cache, (self.H + 2, *self.grid[1:]), self.freqs, offset).unflatten(0, (self.H + 2, N))

    @torch.no_grad()
    def set_refs(self, latents, keep):
        """Cache [B, C, R, h, w] references and a boolean [B, R] mask inside the caller's inference autocast."""
        if not self.num_refs:
            raise ValueError('This resolver has no reference conditioning')
        if latents.ndim != 5 or latents.shape[2] != self.num_refs:
            raise ValueError('Reference latents must have shape [batch, channels, references, height, width]')
        if keep.shape != (latents.shape[0], self.num_refs) or keep.dtype != torch.bool:
            raise ValueError('Reference keep mask must be boolean [batch, references]')
        initial = self.initial
        embedded = initial.ref_patch_embedding(latents)
        batch, channels, frames, height, width = embedded.shape
        tokens = height * width
        hidden = embedded.flatten(3).permute(0, 2, 3, 1)
        hidden = torch.where(keep[:, :, None, None], hidden, initial.pad_token.to(hidden.dtype))
        times = torch.zeros((batch, frames), device=latents.device)
        time_embedding = initial.time_embedding(
            sinusoidal_embedding_1d(initial.freq_dim, times.flatten()).unflatten(0, times.shape))
        projected = initial.time_projection(time_embedding).unflatten(2, (6, initial.dim))
        frequencies = rope.cached_freqs_i(self.rope_cache, (1, height, width), self.freqs)
        cached_keys, cached_values = [], []
        for layer in self.blocks:
            block, attention = layer.block, layer.block.self_attn
            modulation = (block.modulation.unsqueeze(0) + projected).chunk(6, dim=2)
            normalized = (block.norm1(hidden) * (1 + modulation[1]) + modulation[0]).flatten(0, 1)
            query = attention.norm_q(attention.q(normalized)).view(batch * frames, tokens, self.n, self.d)
            key = attention.norm_k(attention.k(normalized)).view(batch * frames, tokens, self.n, self.d)
            value = attention.v(normalized).view(batch * frames, tokens, self.n, self.d).bfloat16()
            query = rope.apply_rope(query, frequencies).bfloat16()
            key = rope.apply_rope(key, frequencies).bfloat16()
            attended = flash_attn_func(query, key, value).unflatten(0, (batch, frames))
            hidden = hidden + attention.o(attended.flatten(3)) * modulation[2]
            hidden = hidden + block.ffn(block.norm2(hidden) * (1 + modulation[4]) + modulation[3]) * modulation[5]
            cached_keys.append(key.unflatten(0, (batch, frames)).flatten(1, 2))
            cached_values.append(value.unflatten(0, (batch, frames)).flatten(1, 2))
        reference_context = torch.stack(cached_keys), torch.stack(cached_values)
        if self.reference_context is None:
            self.reference_context = reference_context
        else:
            for existing, updated in zip(self.reference_context, reference_context):
                if existing.shape != updated.shape:
                    raise ValueError('Reference geometry changed after initialization')
                existing.copy_(updated)

    def context(self, latents, pose, ofi):
        """
        The clean context [sink | history], run once per frame.

        -> K at reference positions and slots 0..H, V: each [layers, B, (R + 1 + H) N, n, d] bf16
        """
        x, _, e0 = self.embed(latents, pose, ofi, torch.zeros(ofi.shape, device=ofi.device))
        if self.num_refs and self.reference_context is None:
            raise ValueError('Reference-conditioned resolver requires set_refs before sampling')
        B, F, N, _ = x.shape
        H, n, d = self.H, self.n, self.d
        freqs = self.slot_freqs(N)
        ks, vs = [], []
        for layer_index, layer in enumerate(self.blocks):
            block, attention = layer.block, layer.block.self_attn
            modulation = (block.modulation.unsqueeze(0) + e0).chunk(6, dim=2)
            normalized = (block.norm1(x) * (1 + modulation[1]) + modulation[0]).flatten(1, 2)
            q = attention.norm_q(attention.q(normalized)).view(B, F, N, n, d)
            k = attention.norm_k(attention.k(normalized)).view(B, F, N, n, d)
            v = attention.v(normalized).view(B, F, N, n, d).bfloat16()
            k_true = rope.apply_rope(k.flatten(1, 2), freqs[:F].flatten(0, 1)).bfloat16()
            sink = flash_attn_func(rope.apply_rope(q[:, 0], freqs[0]).bfloat16(), k_true[:, :N], v[:, 0])
            history_q = rope.apply_rope(q[:, 1:].flatten(0, 1), freqs[H]).bfloat16()
            history_k = rope.apply_rope(k[:, 1:].flatten(0, 1), freqs[H]).bfloat16()
            sink_k = k_true[:, None, :N].expand(B, H, N, n, d).flatten(0, 1)
            sink_v = v[:, :1].expand(B, H, N, n, d).flatten(0, 1)
            history_keys = torch.cat((sink_k, history_k), 1)
            history_values = torch.cat((sink_v, v[:, 1:].flatten(0, 1)), 1)
            if self.num_refs:
                reference_keys, reference_values = (cached[layer_index] for cached in self.reference_context)
                history_reference_keys = reference_keys[:, None].expand(-1, H, -1, -1, -1).flatten(0, 1)
                history_reference_values = reference_values[:, None].expand(-1, H, -1, -1, -1).flatten(0, 1)
                history_keys = torch.cat((history_reference_keys, history_keys), 1)
                history_values = torch.cat((history_reference_values, history_values), 1)
            history = flash_attn_func(history_q, history_keys, history_values)
            output = torch.cat((sink[:, None], history.unflatten(0, (B, H))), dim=1)
            x = x + attention.o(output.flatten(3)) * modulation[2]
            x = x + block.ffn(block.norm2(x) * (1 + modulation[4]) + modulation[3]) * modulation[5]
            ks.append(torch.cat((reference_keys, k_true), 1) if self.num_refs else k_true)
            vs.append(torch.cat((reference_values, v.flatten(1, 2)), 1) if self.num_refs else v.flatten(1, 2))
        return torch.stack(ks), torch.stack(vs)

    def audio(self, codes, f, tokens=None):
        """
        Present frame f's audio: the code table from 2a, a = max(0, f - 374), the present at f - a.

        codes: [B, codebooks, >= 2f + 2]
        tokens: that table already encoded, or None
        -> [B, 1, K + 1, D]
        """
        a = max(0, f - (AUDIO_CODE_QUANTUM // 2 - 1))
        if tokens is None:
            table = codes[:, :, 2 * a:2 * a + AUDIO_CODE_QUANTUM]
            assert table.shape[2] == AUDIO_CODE_QUANTUM, table.shape
            with torch.autocast('cuda', enabled=False):
                latents = audio_latents_from_codes(self.mimi, table, self.num_codebooks)
            tokens = self.initial.audio_encoder(latents)
        B = codes.shape[0]
        ofi = torch.tensor([[0] * (1 + self.H) + [f - a]] * B, device=codes.device)
        return self.initial._window_audio(tokens, ofi, torch.arange(B, device=codes.device))

    def velocity(self, x_t, pose, f, t, context, audio):
        """
        Present frame f's velocity against the context.

        x_t, pose: [B, C, h, w]
        t: 0-d, * 1000
        context: from self.context
        -> v [B, C, h, w]
        """
        B = x_t.shape[0]
        ofi = torch.full((B, 1), f, device=x_t.device)
        x, e, e0 = self.embed(x_t[:, :, None], pose[:, :, None], ofi, t.expand(B, 1))
        N, n, d = x.shape[2], self.n, self.d
        freqs = self.slot_freqs(N)[self.H + 1]
        ctx_k, ctx_v = context
        for i, layer in enumerate(self.blocks):
            block, attention = layer.block, layer.block.self_attn
            modulation = (block.modulation.unsqueeze(0) + e0).chunk(6, dim=2)
            normalized = (block.norm1(x) * (1 + modulation[1]) + modulation[0]).flatten(1, 2)
            q = rope.apply_rope(attention.norm_q(attention.q(normalized)).view(B, N, n, d), freqs).bfloat16()
            k = rope.apply_rope(attention.norm_k(attention.k(normalized)).view(B, N, n, d), freqs).bfloat16()
            v = attention.v(normalized).view(B, N, n, d).bfloat16()
            output = flash_attn_func(q, torch.cat((ctx_k[i], k), 1), torch.cat((ctx_v[i], v), 1))
            x = x + attention.o(output.flatten(2)).unflatten(1, (1, N)) * modulation[2]
            if layer.audio_block is not None:
                x = x + layer.audio_block.forward_present(x, audio)
            x = x + block.ffn(block.norm2(x) * (1 + modulation[4]) + modulation[3]) * modulation[5]
        return self.final((x.flatten(1, 2), e, None, (1, *self.grid[1:]), None, None))[:, :, 0].float()

    def frame(self, out, pose, codes, f, x, ts, tokens=None):
        """
        Resolves present frame f into out[:, :, f]: the context [sink | out f-H..f-1] once, then Euler steps along ts.

        out, pose: [B, C, L, h, w]
        x: [B, C, h, w], at ts[0]
        ts: t, not * 1000
        tokens: as in audio
        """
        ofi = context_frames(f, self.H)
        index = [max(j, 0) for j in ofi]
        context = self.context(out[:, :, index], pose[:, :, index], torch.tensor([ofi] * out.shape[0], device=out.device))
        audio = self.audio(codes, f, tokens)
        for k in range(len(ts) - 1):
            x = x + (ts[k + 1] - ts[k]) * self.velocity(x, pose[:, :, f], f, ts[k] * 1000, context, audio)
        out[:, :, f] = x
