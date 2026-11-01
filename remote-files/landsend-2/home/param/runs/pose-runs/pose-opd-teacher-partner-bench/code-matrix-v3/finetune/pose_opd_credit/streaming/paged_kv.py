"""A bounded physical KV ring behind FA3's absolute logical page table.

The model still uses absolute positions and its original sliding-window mask.
Logical pages older than the attention window alias reusable physical pages;
they are never read by a valid query. Rewind reserves additional old positions,
so replacing the speculative suffix needs no window copies or page migration.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class PageRingLayout:
    logical_positions: int
    attention_window: int
    rewind_positions: int
    append_positions: int
    page_size: int = 256

    def __post_init__(self):
        if (self.logical_positions < 1 or self.attention_window < 1
                or self.rewind_positions < 0 or self.append_positions < 1):
            raise ValueError('Positive logical/window/append sizes and nonnegative rewind are required')
        if self.page_size < 256 or self.page_size % 256:
            raise ValueError('Pinned FA3 requires page sizes divisible by 256')

    @property
    def pages_per_row(self):
        # Conservative bound includes a complete future append as well as the
        # retained rewind suffix. At each query every visible key's distance
        # from any previously written position is strictly below this capacity.
        return math.ceil((self.attention_window + self.rewind_positions + self.append_positions) / self.page_size)

    @property
    def capacity_positions(self):
        return self.pages_per_row * self.page_size

    @property
    def logical_pages(self):
        return math.ceil(self.logical_positions / self.page_size)

    def page_table(self, batch, device):
        if batch < 1:
            raise ValueError('Paged KV requires a positive batch size')
        logical = torch.arange(self.logical_pages, dtype=torch.int32, device=device)
        rows = torch.arange(batch, dtype=torch.int32, device=device)
        return (rows[:, None] * self.pages_per_row + logical[None].remainder(self.pages_per_row)).contiguous()

    def cache_shape(self, batch, heads, head_dim):
        return (batch * self.pages_per_row, self.page_size, heads, head_dim)


def allocate_cache(*, batch, positions, heads, head_dim, layers, device,
                   mode='contiguous', attention_window=750, rewind_positions=32, append_positions=32):
    """Allocate only the selected layout; never first allocate a full history."""
    if mode not in ('contiguous', 'paged'):
        raise ValueError(f'Unknown Voxtral KV cache mode: {mode}')
    layout, table = None, None
    if mode == 'paged':
        layout = PageRingLayout(positions, attention_window, rewind_positions, append_positions)
        table = layout.page_table(batch, device)
        shape = layout.cache_shape(batch, heads, head_dim)
    else:
        shape = (batch, positions, heads, head_dim)
    keys = [torch.zeros(shape, dtype=torch.bfloat16, device=device) for _ in range(layers)]
    values = [torch.zeros(shape, dtype=torch.bfloat16, device=device) for _ in range(layers)]
    return keys, values, table, layout
