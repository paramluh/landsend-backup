"""
Cross-attention over frozen Orpheus hidden states (vidcon-llm models/pose/hidden.py), per present frame: frame f
reads only its own gathered hiddens, positions {2f - 1, 2f} (the producers of the Mimi frames the audio path gives
it), plus one learned null token. Sink and history never query. A window whose audio is dropped reads the learned
drop token instead of its hiddens.
"""

import torch
import torch.nn as nn
from flash_attn import flash_attn_func

from .audio import RMSNorm


class PresentHiddenCrossAttention(nn.Module):
    """
    Q: present tokens [B, P, N, dim]; KV: hiddens [B, P, 2, kv_dim] + the null token, each projected straight to
    projection_tokens K/V tokens at dim (the null's averaged to one). initialization='zero' zeroes out_proj.
    """

    def __init__(self, dim, num_heads, kv_dim, initialization, projection_tokens=1, eps=1e-6):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.projection_tokens = projection_tokens

        self.norm_q_in = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.norm_kv_in = nn.LayerNorm(kv_dim, elementwise_affine=False, eps=eps)
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(kv_dim, dim * 2 * projection_tokens)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.out_proj = nn.Linear(dim, dim)
        if initialization == 'zero':
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)
        elif initialization != 'random':
            raise ValueError(f'unknown initialization: {initialization!r}')
        # Random, not zero: norm_kv_in at ~0 scales the gradient by 1/sqrt(eps).
        self.null_token = nn.Parameter(torch.randn(1, 1, 1, kv_dim))

    def forward_present(self, x, hiddens):
        B, P, N, D = x.shape
        n, hd = self.num_heads, self.head_dim

        null = self.null_token.to(hiddens.dtype).expand(B, P, 1, -1)
        kv = self.kv_proj(self.norm_kv_in(torch.cat([hiddens, null], dim=2)))
        kv = kv.unflatten(-1, (self.projection_tokens, 2, D))
        kv = torch.cat([kv[:, :, :-1].flatten(2, 3), kv[:, :, -1:].mean(dim=3)], dim=2)
        k, v = kv.unbind(dim=3)
        k = self.norm_k(k)

        q = self.norm_q(self.q_proj(self.norm_q_in(x)))
        q = q.reshape(B * P, N, n, hd)
        k = k.reshape(B * P, -1, n, hd)
        v = v.reshape(B * P, -1, n, hd)
        out_dtype = q.dtype
        out = flash_attn_func(q.bfloat16(), k.bfloat16(), v.bfloat16()).to(out_dtype)
        return self.out_proj(out.reshape(B, P, N, D))


class HiddenAdapter(nn.Module):
    """drop_token [1, 1, 2, in_dim]: replaces a present frame's two hiddens wherever the window's audio is dropped."""

    def __init__(self, hidden_dim, num_attn_heads, num_layers, in_dim, injection_offset, injection_interval,
                 initialization, projection_tokens=1, max_blocks=None):
        super().__init__()
        self.drop_token = nn.Parameter(torch.randn(1, 1, 2, in_dim))
        layers = list(range(injection_offset, num_layers, injection_interval))
        if max_blocks is not None:
            layers = layers[:max_blocks]
        self.injection_layers = layers
        self.blocks = nn.ModuleList([
            PresentHiddenCrossAttention(hidden_dim, num_attn_heads, in_dim, initialization, projection_tokens)
            for _ in self.injection_layers
        ])
        self._layer_to_block = {layer: i for i, layer in enumerate(self.injection_layers)}

    def get_block(self, transformer_block_idx):
        i = self._layer_to_block.get(transformer_block_idx)
        return self.blocks[i] if i is not None else None
