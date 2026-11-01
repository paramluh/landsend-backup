"""Explicit teacher audio-table conditioning for complete or trimmed clips."""

import torch

from dataloading.sources import AUDIO_CODE_QUANTUM
from inference.common import sliding_code_table
from models.pose.audio import audio_latents_from_codes
from utils.common import AUTOCAST_DTYPE


@torch.no_grad()
def resolver_tokens(resolver, code_tables, codes, frame, cache):
    """Encode the full retained table, separately from waveform-length codes.

    A trimmed video can have 64 latent frames, 128 waveform codes, and a full
    750-code conditioning table. The separate teacher needs that entire table,
    exactly as the stabilizer does. Cache only by table start within one clip.
    """
    start = max(0, frame - (AUDIO_CODE_QUANTUM // 2 - 1))
    if start not in cache:
        if len(code_tables) != len(codes):
            raise ValueError('Resolver conditioning batch lengths differ')
        tables = [sliding_code_table(table, code, frame)[0] for table, code in zip(code_tables, codes)]
        if not tables or any(table.shape[-1] != AUDIO_CODE_QUANTUM for table in tables):
            raise ValueError('Resolver conditioning requires full retained audio tables')
        table = torch.cat(tables)
        with torch.autocast(table.device.type, enabled=False):
            latents = audio_latents_from_codes(resolver.mimi, table, resolver.num_codebooks)
        with torch.autocast(table.device.type, dtype=AUTOCAST_DTYPE, cache_enabled=False):
            cache[start] = resolver.initial.audio_encoder(latents)
        for old in list(cache):
            if old < start:
                del cache[old]
    return cache[start]
