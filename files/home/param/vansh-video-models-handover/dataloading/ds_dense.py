import numpy as np
import torch

from dataloading.sources import AUDIO_CODE_QUANTUM
from utils.common import is_main_process


SHUFFLE_SEED = 43


class DenseWindows:
    """
    Item = one rank microbatch of B windows [sink | H history | P present], each
    window a [C, 1+H+P, h, w] slice of one clip. Clip c contributes exactly
    floor((T_c - 1) / P) windows per epoch; a per-(epoch, clip) seeded
    startup_window in {1..P} is the present-frame count of the first window
    (frames 1..startup_window, right-aligned, no history); full groups of P follow.
    history_dropout[d-1] = P(drop the d oldest available history frames of a
    window); the remaining mass keeps history intact.
    """

    def __init__(self, source, num_history, num_present, history_dropout, rank, world_size, batch_size,
                 accumulation_steps):
        self.source = source
        self.num_history = num_history
        self.num_present = num_present
        assert len(history_dropout) == num_history and sum(history_dropout) <= 1, history_dropout
        self._drop_cdf = np.cumsum(history_dropout)
        self.rank = rank
        self.world_size = world_size
        self.batch_size = batch_size
        window_counts = (source.lengths - 1) // num_present
        self._prefix = np.zeros(len(window_counts) + 1, dtype=np.int64)
        np.cumsum(window_counts, out=self._prefix[1:])
        quantum = batch_size * world_size * accumulation_steps
        self._windows_per_epoch = int(self._prefix[-1]) // quantum * quantum
        if not self._windows_per_epoch:
            raise RuntimeError('Dense window stream is smaller than one global accumulation step')
        if is_main_process():
            print(f'DenseWindows: {len(source.clips)} clips, {self._windows_per_epoch} windows/epoch, {len(self)} microbatches/epoch')

    def __len__(self):
        return self._windows_per_epoch // (self.batch_size * self.world_size)

    def _window(self, epoch, window_index):
        H, P = self.num_history, self.num_present
        clip = int(np.searchsorted(self._prefix, window_index, side='right') - 1)
        ordinal = window_index - int(self._prefix[clip])
        rng = np.random.default_rng(np.random.SeedSequence([SHUFFLE_SEED, epoch, clip]))
        # Present-frame count of the clip's first window, uniform in {1..P}; later windows are full and follow it.
        startup_window = int(rng.integers(1, P + 1))
        if ordinal == 0:
            start, count = 1, startup_window
        else:
            start, count = 1 + startup_window + (ordinal - 1) * P, P
        slot = P - count
        ofi = torch.full((1 + H + P,), -1, dtype=torch.long)
        ofi[0] = 0
        history = torch.arange(max(1, start - H), start)
        drop = int(np.searchsorted(self._drop_cdf, rng.random(ordinal + 1)[-1], side='right')) + 1
        if drop <= H:
            history = history[drop:]
        ofi[1 + H - len(history):1 + H] = history
        ofi[1 + H + slot:1 + H + slot + count] = torch.arange(start, start + count)
        return clip, ofi

    @staticmethod
    def _gather(values, ofi):
        window = values.new_zeros(values.shape[0], len(ofi), *values.shape[2:])
        valid = ofi >= 0
        window[:, valid] = values[:, ofi[valid]]
        return window

    def __getitem__(self, index):
        # Epoch stream of _windows_per_epoch windows laid out [microbatch][rank][B].
        # Dataset index = epoch * len(self) + microbatch.
        epoch, microbatch = divmod(index, len(self))
        first = (microbatch * self.world_size + self.rank) * self.batch_size
        windows = [self._window(epoch, first + offset) for offset in range(self.batch_size)]

        rows = {clip: row for row, clip in enumerate(dict.fromkeys(clip for clip, _ in windows))}
        sources = {clip: self.source.load(clip) for clip in rows}
        ofi = torch.stack([ofi for _, ofi in windows])
        codes = torch.zeros(self.batch_size, next(iter(sources.values())).codes.shape[0], AUDIO_CODE_QUANTUM, dtype=torch.long)
        for clip, row in rows.items():
            source_codes = sources[clip].codes
            length = min(source_codes.shape[1], AUDIO_CODE_QUANTUM)
            codes[row, :, :length] = source_codes[:, :length]
        return {
            'latents_low': torch.stack([self._gather(sources[clip].latents, w) for clip, w in windows]),
            'pose_latents_low': torch.stack([self._gather(sources[clip].pose_latents, w) for clip, w in windows]),
            'original_frame_indices': ofi,
            'present_valid': ofi[:, 1 + self.num_history:] >= 0,
            'window_source': torch.tensor([rows[clip] for clip, _ in windows]),
            'codes': codes,
        }


class EvalClips:
    """Item = B whole clips, right-padded to the longest; eval builds its own windows from these."""

    def __init__(self, source, rank, world_size, batch_size):
        self.source = source
        self.rank = rank
        self.world_size = world_size
        self.batch_size = batch_size
        if is_main_process():
            print(f'EvalClips: {len(source.clips)} clips, {len(self)} microbatches')

    def __len__(self):
        return len(self.source.clips) // (self.batch_size * self.world_size)

    @staticmethod
    def _pad(x, n):
        return torch.cat([x, x.new_zeros(x.shape[0], n - x.shape[1], *x.shape[2:])], dim=1)

    def __getitem__(self, index):
        first = (index * self.world_size + self.rank) * self.batch_size
        clips = list(range(first, first + self.batch_size))
        items = [self.source.load(clip) for clip in clips]
        t_max = max(item.latents.shape[1] for item in items)
        codes = torch.zeros(self.batch_size, items[0].codes.shape[0], AUDIO_CODE_QUANTUM, dtype=torch.long)
        for row, item in enumerate(items):
            length = min(item.codes.shape[1], AUDIO_CODE_QUANTUM)
            codes[row, :, :length] = item.codes[:, :length]
        return {
            'latents_low': torch.stack([self._pad(item.latents, t_max) for item in items]),
            'pose_latents_low': torch.stack([self._pad(item.pose_latents, t_max) for item in items]),
            'source_id': torch.tensor(clips),
            'codes': codes,
        }
