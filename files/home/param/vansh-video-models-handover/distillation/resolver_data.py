"""Full-GT windows and exact global batching for resolver distillation."""
import numpy as np
import torch
import torch.distributed as dist

from dataloading.ds_aloe import ShuffledDenseWindows
from dataloading.ds_dense import SHUFFLE_SEED


def rank_row_units(global_targets, nodes, local_world, local_rank):
    if global_targets % (4 * nodes):
        raise ValueError('Global target batch must divide into four-target rows on each node')
    per_node = global_targets // (4 * nodes)
    if per_node < local_world:
        raise ValueError('Every rank must have at least one four-target row')
    counts = [per_node // local_world + (r < per_node % local_world) for r in range(local_world)]
    return counts[local_rank], sum(counts[:local_rank]), per_node


class ResolverWindows(ShuffledDenseWindows):
    """[sink | six GT frames | four targets], disjoint variable-sized rank batches.

    The six GT frames let even the target in the highest-noise stabilizer slot
    have its complete real history. No startup padding substitutes for history.
    """
    def __init__(self, source, local_rank, local_world, global_targets, nodes, groups, min_over=None,
                 history_dropout=(0., 0., 0.)):
        if len(history_dropout) != 3 or any(p < 0 for p in history_dropout) or sum(history_dropout) > 1:
            raise ValueError('Expected fine-tuning parent history dropout over three slots')
        self.parent_drop_cdf = np.cumsum(history_dropout)
        units, offset, total = rank_row_units(global_targets, nodes, local_world, local_rank)
        self.node_batch_rows = total * groups
        self.local_offset = offset * groups
        self.local_batch_rows = units * groups
        super().__init__(source, 6, 4, [0.] * 6, 0, 1, self.local_batch_rows, 1)
        counts = np.maximum(0, (source.lengths - 7) // 4)
        self._prefix = np.zeros(len(counts) + 1, dtype=np.int64)
        np.cumsum(counts, out=self._prefix[1:])
        self._windows_per_epoch = int(self._prefix[-1]) // self.node_batch_rows * self.node_batch_rows
        if min_over is not None:
            count = torch.tensor(self._windows_per_epoch)
            dist.all_reduce(count, op=dist.ReduceOp.MIN, group=min_over)
            self._windows_per_epoch = int(count)
        if not self._windows_per_epoch:
            raise ValueError('No full resolver batch in smallest shard')
        self._perm_epoch = None

    def __len__(self):
        return self._windows_per_epoch // self.node_batch_rows

    def _order(self, epoch):
        full = super()._order(epoch).reshape(-1, self.node_batch_rows)
        return full[:, self.local_offset:self.local_offset + self.local_batch_rows].reshape(-1)

    def _window(self, epoch, window_index):
        clip = int(np.searchsorted(self._prefix, window_index, side='right') - 1)
        ordinal = window_index - int(self._prefix[clip])
        start = 7 + ordinal * 4
        ofi = torch.cat((torch.zeros(1, dtype=torch.long), torch.arange(start - 6, start + 4)))
        # Same per-(epoch,clip) stream and oldest-history categorical rule as
        # DenseWindows. BimodalWindows resolves a view of this parent mask;
        # it does not independently drop each resolver's two history slots.
        rng = np.random.default_rng(np.random.SeedSequence([SHUFFLE_SEED, epoch, clip]))
        rng.integers(1, 5)  # Fine-tuning consumes its startup-window draw first.
        drop = int(np.searchsorted(self.parent_drop_cdf, rng.random(ordinal + 1)[-1], side='right')) + 1
        if drop <= 3:
            ofi[4:4+drop] = -1
        return clip, ofi


def learning_rate_factor(update, total, warmup, final_ratio=.1):
    """LR actually used for zero-based update: warmup, then cosine to exact floor."""
    import math
    if total < 1 or not 0 < final_ratio <= 1 or not 0 <= warmup < total:
        raise ValueError('Invalid LR schedule')
    update = min(max(update, 0), total - 1)
    if warmup and update < warmup:
        return (update + 1) / warmup
    progress = (update - warmup) / max(1, total - 1 - warmup)
    return final_ratio + (1 - final_ratio) * .5 * (1 + math.cos(math.pi * progress))
