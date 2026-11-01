import numpy as np
import torch

from dataloading.ds_dense import SHUFFLE_SEED

RESOLVING_SLOT_STREAM = 5


class BimodalWindows:
    """
    Item = (stabilising microbatch, resolving microbatch). Each stabilising window w [sink | H | P] gives one resolving
    window [sink | num_history | 1]: slots [0, 1+H+j-num_history .. 1+H+j] of w, for a valid present slot j drawn
    uniformly from (SHUFFLE_SEED, epoch, w, RESOLVING_SLOT_STREAM). Pads stay pads; codes and ids are w's.
    """

    def __init__(self, windows, num_history):
        assert 1 <= num_history <= windows.num_history, num_history
        self.windows = windows
        self.num_history = num_history

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        data, h = self.windows, self.num_history
        batch = data[index]
        epoch, microbatch = divmod(index, len(data))
        first = (microbatch * data.world_size + data.rank) * data.batch_size
        window_ids = data._order(epoch)[first:first + data.batch_size]
        j = torch.tensor([
            int(np.random.default_rng(np.random.SeedSequence([SHUFFLE_SEED, epoch, int(w), RESOLVING_SLOT_STREAM]))
                .choice(np.flatnonzero(valid.numpy())))
            for w, valid in zip(window_ids, batch['present_valid'])
        ])[:, None]
        slots = torch.cat((torch.zeros_like(j), 1 + data.num_history + j + torch.arange(-h, 1)), dim=1)  # [B, 1 + h + 1]
        frames = lambda x: torch.gather(x, 2, slots[:, None, :, None, None].expand(-1, x.shape[1], -1, *x.shape[3:]))
        resolving = {
            'latents_low': frames(batch['latents_low']),
            'pose_latents_low': frames(batch['pose_latents_low']),
            'original_frame_indices': torch.gather(batch['original_frame_indices'], 1, slots),
            'present_valid': torch.gather(batch['present_valid'], 1, j),
            **{key: batch[key] for key in ('window_source', 'codes', 'chunk_index', 'person_index', 'bad_audio') if key in batch},
        }
        return batch, resolving
