"""
After 2× upsample, Mimi tokens are 25 Hz. `drop` discards the leading encoder
outputs so token 0 aligns with latent frame 1.
"""

import torch
import torch.nn as nn
from flash_attn import flash_attn_func


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        n = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return n.type_as(x) * self.weight


class FrameCrossAttentionBlock(nn.Module):
    """
    Frame-by-frame cross-attention.

    Q: [B, P, spatial_tokens, dim]
    KV: [B, P, conditioning_tokens, dim]
    """

    def __init__(self, dim, num_heads, initialization, eps=1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.norm_x = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.norm_a = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, dim * 2)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.out_proj = nn.Linear(dim, dim)
        if initialization == 'zero':
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)
        elif initialization != 'random':
            raise ValueError(f'unknown initialization: {initialization!r}')

    def forward_present(self, x, audio_features):
        B, P, spatial_size, D = x.shape
        n, hd = self.num_heads, self.head_dim
        N_tokens = audio_features.shape[2]

        q = self.norm_q(self.q_proj(self.norm_x(x)))
        kv = self.kv_proj(self.norm_a(audio_features))
        k, v = kv.chunk(2, dim=-1)
        k = self.norm_k(k)
        q = q.reshape(B * P, spatial_size, n, hd)
        k = k.reshape(B * P, N_tokens, n, hd)
        v = v.reshape(B * P, N_tokens, n, hd)
        out_dtype = q.dtype
        out = flash_attn_func(q.bfloat16(), k.bfloat16(), v.bfloat16()).to(out_dtype)
        return self.out_proj(out.reshape(B, P, spatial_size, D))


def audio_latents_from_codes(mimi, codes, num_codebooks):
    """Mimi codes [B, >=num_codebooks, T] -> frozen 25 Hz latents [B, 2T, in_dim], the MimiAudioEncoder input."""
    with torch.no_grad():
        latents = mimi.quantizer.decode(codes[:, :num_codebooks].long())
        latents = mimi.upsample(latents)
    return latents.transpose(1, 2)


class MimiAudioEncoder(nn.Module):
    def __init__(self, transformer, in_dim, hidden_dim, drop, num_window_tokens):
        super().__init__()
        self.transformer = transformer
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.drop = drop
        self.null_token = nn.Parameter(torch.zeros(1, 1, 1, hidden_dim))
        self.no_audio_tokens = nn.Parameter(torch.zeros(1, 1, num_window_tokens, hidden_dim))

    def forward(self, emb25):
        h = self.transformer(hidden_states=emb25)[0]
        return self.proj(h)[:, self.drop:]


class AudioAdapter(nn.Module):
    def __init__(self, hidden_dim, num_attn_heads, num_layers, injection_offset, injection_interval,
                 initialization, max_blocks=None):
        super().__init__()
        layers = list(range(injection_offset, num_layers, injection_interval))
        if max_blocks is not None:
            layers = layers[:max_blocks]  # cap count by dropping the deepest injection layers
        self.injection_layers = layers
        self.blocks = nn.ModuleList([
            FrameCrossAttentionBlock(hidden_dim, num_attn_heads, initialization)
            for _ in self.injection_layers
        ])
        self._layer_to_block = {layer: i for i, layer in enumerate(self.injection_layers)}

    def get_block(self, transformer_block_idx):
        i = self._layer_to_block.get(transformer_block_idx)
        return self.blocks[i] if i is not None else None
