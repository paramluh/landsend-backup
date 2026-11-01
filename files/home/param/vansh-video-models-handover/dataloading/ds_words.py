import os

import torch
from safetensors import safe_open

from dataloading.ds_aloe import AloeChunkSource, AloeEvalClips, ShuffledDenseWindows


class WordsChunkSource(AloeChunkSource):
    """hiddens_root: mirrors root_dir; <hiddens_root>/<paths>/llm_hiddens.safetensors `hiddens` bf16 [P, 4096] are the
    chunk's Orpheus hiddens (precompute/orpheus.py), position p the producer of Mimi frame p."""

    def __init__(self, root_dir, *args, hiddens_root, **kwargs):
        super().__init__(root_dir, *args, **kwargs)
        self.hiddens_paths = [
            os.path.join(hiddens_root, os.path.relpath(os.path.dirname(clip.video_path), root_dir), 'llm_hiddens.safetensors')
            for clip in self.clips
        ]

    def load_hiddens(self, clip_index, frames):
        """frames [n] chunk-level latent frames (< 1: no frame) -> [n, 2, 4096]: frame f's positions {2f - 1, 2f},
        zeros for no frame."""
        first = 2 * frames.clamp(min=1) - 1
        lo, hi = int(first.min()), int(first.max()) + 2
        with safe_open(self.hiddens_paths[clip_index], framework='pt') as handle:
            rows = handle.get_slice('hiddens')[lo:hi]
        hiddens = rows[(first - lo)[:, None] + torch.arange(2)]
        return hiddens * (frames >= 1)[:, None, None].to(hiddens.dtype)


class WordsDenseWindows(ShuffledDenseWindows):
    """llm_hiddens [B, P, 2, 4096]: the hiddens of each window's present frames."""

    def __getitem__(self, index):
        batch = super().__getitem__(index)
        epoch, microbatch = divmod(index, len(self))
        first = (microbatch * self.world_size + self.rank) * self.batch_size
        order = self._order(epoch)
        windows = [self._window(epoch, int(order[first + offset])) for offset in range(self.batch_size)]
        batch['llm_hiddens'] = torch.stack([self.source.load_hiddens(clip, w[1 + self.num_history:]) for clip, w in windows])
        return batch


class WordsEvalClips(AloeEvalClips):
    """llm_hiddens [B, H + P, 2, 4096]: the hiddens of frames 1..H + P."""

    def __getitem__(self, index):
        batch = super().__getitem__(index)
        frames = torch.arange(1, self.prefix)
        batch['llm_hiddens'] = torch.stack([self.source.load_hiddens(clip, frames) for clip in batch['source_id'].tolist()])
        return batch
