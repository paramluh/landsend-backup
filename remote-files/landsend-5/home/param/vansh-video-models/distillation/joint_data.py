"""One complete, deterministic epoch for joint stabilizer/resolver distillation.

Every rank must read the SAME full-corpus CSV and have access to every listed
clip and its separate actor-reference bank. Node-local CSV shards are rejected:
reducing to the smallest shard, dropping a last batch, and cycling a short shard
are not full-epoch semantics.

A usable clip length L contributes floor((L - 7) / 4) complete four-target
windows, clamped at zero. Targets start at 7, 11, ...; each raw row contains
[sink | six preceding frames | four targets]. The six preceding frames are GT
only for the resolver objective. The no-GT stabilizer helper must discard their
visual values and generate its own six-frame prefix and four targets. Reference
images always come from property_1..property_9 in the actor bank, never from a
target or another temporal frame of the loaded clip.

One dataset item pairs one complete global stabilizer batch with one complete
global resolver batch. Both objectives use the same selected window IDs and
reference images. The default global_batch_size=160 counts target frames per
objective, hence 40 four-target rows for EACH objective. Rank microbatches are
accumulated until that global batch is complete. The final group uses explicit
zero weights for padding, so every eligible window contributes exactly once
PER objective. Each objective loss is normalized separately by its actual
global_objective_rows count. global_valid_rows counts unique selected windows,
not the sum of their two objective uses.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from dataloading.ds_aloe import AloeChunkSource, REF_DROP_STREAM, REF_PICK_STREAM
from dataloading.ds_dense import SHUFFLE_SEED
from dataloading.sources import NUM_REFS


OBJECTIVES = ('stabilizer', 'resolver')
DATA_HISTORY = 6
TARGETS_PER_ROW = 4
PLAN_VERSION = 2


def _sha256_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _gather(frame0, body, body_start, ofi):
    """Gather an unshifted source window; audio OFIs are shifted separately."""
    values = torch.cat((frame0, body), dim=1)
    indices = ofi.clone()
    indices[ofi >= 1] = ofi[ofi >= 1] - body_start + 1
    gathered = values.new_zeros(values.shape[0], len(ofi), *values.shape[2:])
    valid = ofi >= 0
    gathered[:, valid] = values[:, indices[valid]]
    return gathered


class JointEpochDataset:
    """Prebatched data, compatible with DataLoader(..., batch_size=None).

    ``data[group_index]`` / ``data.group(group_index)`` returns a sequence in
    ``data.objective_schedule`` order: all stabilizer microbatches, then all
    resolver microbatches. Each objective has ``data.objective_microbatches``
    microbatches. They contain the raw tensors, ``objective``, ``sample_weight[B]``, and
    ``global_valid_rows`` / ``global_objective_rows`` / ``global_microbatch_rows``.
    The global counts are identical on every rank, including ranks whose entire
    local tail batch is padding. DDP callers still execute every forward/backward
    and multiply per-row losses by sample_weight before summing. Normalize each
    objective by global_objective_rows[objective], never by a microbatch count
    that may be zero in the padded tail. Shared epoch_window_id values and
    deterministic reference picks identify corresponding rows across objectives.

    An epoch is a fixed constructor argument, not inferred by modulo indexing.
    Out-of-range access fails instead of silently starting another epoch. Save
    ``signature`` and the NEXT group index in the checkpoint and compare the
    signature on resume. Worker count/prefetch do not affect the plan or refs.
    """

    def __init__(self, source, rank, world_size, batch_rows, *, seed=SHUFFLE_SEED,
                 epoch=0, reference_dropout=None, resolver_history_dropout=(.1, .1, 0.),
                 csv_sha256=None, global_batch_size=160):
        if NUM_REFS != 9:
            raise ValueError('Joint distillation requires the nine-reference actor bank')
        if world_size < 1 or not 0 <= rank < world_size or batch_rows < 1:
            raise ValueError('Invalid global rank/world size/local row batch')
        if (type(global_batch_size) is not int or global_batch_size < TARGETS_PER_ROW
                or global_batch_size % TARGETS_PER_ROW):
            raise ValueError('global_batch_size counts targets per objective and must be a positive multiple of four')
        if seed < 0 or epoch < 0:
            raise ValueError('Epoch and seed must be nonnegative')
        if not source.bank:
            raise ValueError('A separate nine-reference actor bank is required')
        self.source = source
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.batch_rows = int(batch_rows)
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.global_batch_size = global_batch_size
        self.global_batch_rows = self.global_batch_size // TARGETS_PER_ROW
        self.global_microbatch_rows = self.world_size * self.batch_rows
        if self.global_batch_rows % self.global_microbatch_rows:
            raise ValueError(
                f'global_batch_size={self.global_batch_size} means {self.global_batch_rows} rows per objective; '
                f'that must divide exactly by world_size * micro_batch_size_per_gpu '
                f'({self.world_size} * {self.batch_rows} = {self.global_microbatch_rows}). '
                'Choose a divisible topology or a smaller per-GPU microbatch.')
        self.objective_microbatches = self.global_batch_rows // self.global_microbatch_rows
        self.objective_schedule = [objective for objective in OBJECTIVES
                                   for _ in range(self.objective_microbatches)]
        self.microbatches_per_group = len(self.objective_schedule)
        self.group_capacity = self.global_batch_rows

        counts = np.maximum(0, (np.asarray(source.lengths, dtype=np.int64) - 7) // 4)
        self._prefix = np.zeros(len(counts) + 1, dtype=np.int64)
        np.cumsum(counts, out=self._prefix[1:])
        self.eligible_rows = int(self._prefix[-1])
        if self.eligible_rows == 0:
            raise ValueError('Full-corpus CSV contains no complete four-target window with six predecessors')
        self.num_groups = (self.eligible_rows + self.group_capacity - 1) // self.group_capacity
        self._order = np.random.default_rng(
            np.random.SeedSequence([self.seed, self.epoch])).permutation(self.eligible_rows)

        drop_counts, probabilities = reference_dropout or ([0], [1.0])
        if (not len(drop_counts) or len(drop_counts) != len(probabilities)
                or any(int(n) != n or not 0 <= n <= NUM_REFS for n in drop_counts)
                or any(not np.isfinite(p) or p < 0 for p in probabilities)
                or abs(sum(probabilities) - 1.0) > 1e-6):
            raise ValueError('Invalid reference dropout counts/probabilities')
        self._ref_drop_counts = np.asarray(drop_counts, dtype=np.int64)
        self._ref_drop_cdf = np.cumsum(probabilities)
        if (len(resolver_history_dropout) != 3
                or any(not np.isfinite(p) or p < 0 for p in resolver_history_dropout)
                or sum(resolver_history_dropout) > 1):
            raise ValueError('Resolver history dropout requires three nonnegative probabilities summing to at most one')
        self._resolver_history_drop_cdf = np.cumsum(resolver_history_dropout)

        records = [
            {'video': clip.video_path, 'pose': clip.pose_path, 'mimi': clip.mimi_path,
             'length': int(clip.length), 'uuid': clip.uuid}
            for clip in source.clips
        ]
        if len({record['video'] for record in records}) != len(records):
            raise ValueError('Full-corpus CSV contains duplicate clip paths; explicit deduplication is required')
        self.signature = {
            'plan_version': PLAN_VERSION,
            'csv_sha256': csv_sha256,
            'ordered_source_sha256': _sha256_json(records),
            'reference_bank': str(source.bank),
            'reference_drop_counts': self._ref_drop_counts.tolist(),
            'reference_drop_p': list(probabilities),
            'resolver_history_dropout': list(resolver_history_dropout),
            'resolver_history_dropout_seed': SHUFFLE_SEED,
            'seed': self.seed,
            'epoch': self.epoch,
            'world_size': self.world_size,
            'batch_rows_per_rank': self.batch_rows,
            'global_batch_size_targets_per_objective': self.global_batch_size,
            'global_batch_rows_per_objective': self.global_batch_rows,
            'global_microbatch_rows': self.global_microbatch_rows,
            'objectives': list(OBJECTIVES),
            'objective_schedule': list(self.objective_schedule),
            'objective_microbatches': self.objective_microbatches,
            'microbatches_per_group': self.microbatches_per_group,
            'paired_window_ids_across_objectives': True,
            'paired_reference_draws_across_objectives': True,
            'epoch_coverage': 'every_eligible_window_once_per_objective',
            'global_valid_rows_semantics': 'unique_windows_not_sum_over_objectives',
            'eligible_rows': self.eligible_rows,
            'num_groups': self.num_groups,
            'targets_per_row': TARGETS_PER_ROW,
            'source_history_frames': DATA_HISTORY,
            'tail_normalization': 'separate_actual_global_objective_rows',
            'excluded_initial_context_frames_per_clip': DATA_HISTORY,
            'incomplete_target_frames': int(np.maximum(0, np.asarray(source.lengths) - 7).sum()
                                            - TARGETS_PER_ROW * self.eligible_rows),
        }
        self.signature['plan_sha256'] = _sha256_json(self.signature)

    def __len__(self):
        return self.num_groups

    def _window(self, window_id, *, resolver=False):
        if not 0 <= window_id < self.eligible_rows:
            raise IndexError(window_id)
        clip = int(np.searchsorted(self._prefix, window_id, side='right') - 1)
        ordinal = int(window_id - self._prefix[clip])
        start = 7 + TARGETS_PER_ROW * ordinal
        ofi = torch.cat((torch.zeros(1, dtype=torch.long), torch.arange(start - DATA_HISTORY, start + 4)))
        if resolver:
            # Exactly the existing ResolverWindows parent-window stream. It is
            # seeded by the fixed pretraining shuffle seed, independently of
            # this epoch planner's permutation seed; one startup integer draw
            # precedes the ordinal-th categorical history draw.
            rng = np.random.default_rng(np.random.SeedSequence([SHUFFLE_SEED, self.epoch, clip]))
            rng.integers(1, 5)
            drop = int(np.searchsorted(self._resolver_history_drop_cdf,
                                       rng.random(ordinal + 1)[-1], side='right')) + 1
            if drop <= 3:
                ofi[4:4 + drop] = -1
        return clip, ofi

    def _reference_keep(self, window_id, present):
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, self.epoch, int(window_id), REF_DROP_STREAM]))
        choice = min(int(np.searchsorted(self._ref_drop_cdf, rng.random(), side='right')),
                     len(self._ref_drop_counts) - 1)
        drop = int(self._ref_drop_counts[choice])
        available = np.flatnonzero(present.numpy())
        # Match the existing nine-reference trainer: unavailable properties
        # already count as dropped, and only present properties can be sampled.
        count = min(len(available), max(0, drop - (NUM_REFS - len(available))))
        keep = present.clone()
        if count:
            keep[torch.from_numpy(rng.choice(available, count, replace=False))] = False
        return keep

    def _load(self, windows, row_ids, valid, *, evaluation=False, targets=4):
        loaded = [self.source.load_window(clip, ofi) for clip, ofi in windows]
        references = []
        keeps = []
        for row_id, (clip, _), item in zip(row_ids, windows, loaded):
            if evaluation:
                ref, present = self.source.load_eval_refs(clip, item[0])
                keep = present
            else:
                ref, present = self.source.load_refs(clip, item[0], np.random.default_rng(
                    np.random.SeedSequence([self.seed, self.epoch, int(row_id), REF_PICK_STREAM])))
                keep = self._reference_keep(row_id, present)
            if ref.shape[1] != NUM_REFS or present.shape != (NUM_REFS,):
                raise ValueError('Actor bank did not return nine reference properties')
            references.append(ref)
            keeps.append(keep)
        batch = {
            'latents_low': torch.stack([_gather(item[0], item[1], item[4], ofi)
                                       for item, (_, ofi) in zip(loaded, windows)]),
            'pose_latents_low': torch.stack([_gather(item[2], item[3], item[4], ofi)
                                            for item, (_, ofi) in zip(loaded, windows)]),
            'original_frame_indices': torch.stack([item[6] for item in loaded]),
            'source_frame_indices': torch.stack([ofi for _, ofi in windows]),
            'present_valid': torch.ones(len(windows), targets, dtype=torch.bool),
            'window_source': torch.arange(len(windows)),
            'codes': torch.stack([item[5] for item in loaded]),
            'chunk_index': torch.tensor([self.source.chunk_ids[clip] for clip, _ in windows], dtype=torch.int32),
            'person_index': torch.tensor([self.source.person_ids[clip] for clip, _ in windows], dtype=torch.int32),
            'refs': torch.stack(references),
            'ref_keep': torch.stack(keeps),
            'sample_weight': torch.tensor(valid, dtype=torch.float32),
            'epoch_window_id': torch.tensor([int(row_id) if weight else -1
                                             for row_id, weight in zip(row_ids, valid)], dtype=torch.long),
        }
        if self.source.bad_audio is not None:
            batch['bad_audio'] = torch.from_numpy(self.source.bad_audio[[clip for clip, _ in windows]])
        return batch

    def group(self, index):
        if not 0 <= index < len(self):
            raise IndexError(f'Group {index} is outside the fixed epoch of {len(self)} groups')
        first = int(index) * self.group_capacity
        ids = self._order[first:first + self.group_capacity]
        count = len(ids)
        # Partition one selected global batch across accumulation, ranks and
        # local rows. Reuse exactly this partition for both objectives; selecting
        # a second set of IDs would partition the epoch between objectives.
        layout = np.full(self.group_capacity, -1, dtype=np.int64)
        layout[:count] = ids
        layout = layout.reshape(self.objective_microbatches, self.global_microbatch_rows)
        valid_per_microbatch = (layout >= 0).sum(axis=1).tolist()
        objective_counts = {objective: count for objective in OBJECTIVES}
        batches = []
        lo = self.rank * self.batch_rows
        for objective in OBJECTIVES:
            for microbatch in range(self.objective_microbatches):
                local = layout[microbatch, lo:lo + self.batch_rows]
                valid = (local >= 0).astype(np.float32).tolist()
                # Padding has real geometry and participates in DDP, but weight
                # zero. It never advances row counters or metric denominators.
                row_ids = np.maximum(local, 0).tolist()
                batch = self._load([self._window(row_id, resolver=objective == 'resolver')
                                    for row_id in row_ids], row_ids, valid)
                batch.update(objective=objective, kind=objective, group_index=int(index),
                             microbatch_index=len(batches), objective_microbatch_index=microbatch,
                             global_valid_rows=count, global_objective_rows=dict(objective_counts),
                             global_microbatch_rows=int(valid_per_microbatch[microbatch]))
                batches.append(batch)
        return batches

    def __getitem__(self, index):
        return self.group(index)

    def _eval_clips(self, num_targets):
        if num_targets < 1:
            raise ValueError('Evaluation requires a positive target horizon')
        clips = np.flatnonzero(np.asarray(self.source.lengths) >= 7 + num_targets)
        if not len(clips):
            raise ValueError(f'No complete evaluation clip for six source context frames plus {num_targets} targets')
        return clips

    def eval_num_batches(self, num_targets=8):
        clips = self._eval_clips(num_targets)
        return (len(clips) + self.global_microbatch_rows - 1) // self.global_microbatch_rows

    def eval_batch(self, index=0, num_targets=8):
        """Fixed actor refs and canonical [sink | six context | N target] rows.

        The no-GT evaluator uses poses/audio for all six context positions
        and N target positions, generating all 6+N corresponding visual frames.
        No temporal GT value is necessary for that rollout. This method returns
        them only to keep the same canonical input schema as the resolver data.
        No valid clip is repeated within the evaluation schedule; padding is zero
        weighted. Evaluation consumes no training window IDs or RNG state.
        """
        clips = self._eval_clips(num_targets)
        batches = (len(clips) + self.global_microbatch_rows - 1) // self.global_microbatch_rows
        if not 0 <= index < batches:
            raise IndexError(f'Evaluation batch {index} is outside {batches} batches')
        first = (int(index) * self.world_size + self.rank) * self.batch_rows
        indices = [first + offset for offset in range(self.batch_rows)]
        valid = [float(i < len(clips)) for i in indices]
        selected = [int(clips[i]) if i < len(clips) else int(clips[0]) for i in indices]
        ofi = torch.arange(7 + num_targets, dtype=torch.long)
        batch = self._load([(clip, ofi.clone()) for clip in selected], selected, valid,
                           evaluation=True, targets=num_targets)
        batch['global_valid_rows'] = min(self.global_microbatch_rows,
                                         len(clips) - int(index) * self.global_microbatch_rows)
        batch['evaluation'] = True
        return batch

    def eval_row(self, index=0, future_frames=8):
        """Alias for a canonical batched evaluation row, with N future targets."""
        return self.eval_batch(index=index, num_targets=future_frames)


def build_joint_data(config, rank, world_size):
    """Build the supported, replicated-full-corpus source and immutable plan."""
    if config.get('node_sharded'):
        raise ValueError('Joint full-epoch training requires a common full-corpus CSV; node_sharded is unsupported')
    if 'windows_per_epoch' in config:
        raise ValueError('windows_per_epoch would truncate the requested full epoch')
    if 'buckets' in config or 'llm_hiddens_root' in config:
        raise ValueError('Joint data currently supports the Aloe CSV + actor-reference-bank source only')
    refs = config.get('refs')
    if not refs or not refs.get('bank'):
        raise ValueError('Set refs.bank to the separate nine-reference actor bank root')
    bank = Path(refs['bank']) / refs.get('train_split', 'train')
    if not bank.is_dir():
        raise ValueError(f'Nine-reference training bank is not a directory: {bank}')
    csv = Path(config['clipnames_root']) / config['train_csv']
    if csv.suffix != '.csv':
        csv = Path(str(csv) + '.csv')
    source = AloeChunkSource(config['root_dir'], config['clipnames_root'], config['train_csv'],
                             str(bank), pose_file=config['pose_latents'], mimi_file=config['mimi_codes'],
                             bad_audio=config.get('bad_audio', False))
    return JointEpochDataset(source, rank, world_size, config['micro_batch_size_per_gpu'],
                             seed=config.get('seed', SHUFFLE_SEED), epoch=config.get('epoch', 0),
                             reference_dropout=(refs.get('drop_counts', [0]), refs.get('drop_p', [1.0])),
                             resolver_history_dropout=config.get('resolver_history_dropout', [.1, .1, 0.]),
                             csv_sha256=hashlib.sha256(csv.read_bytes()).hexdigest(),
                             global_batch_size=config.get('global_batch_size', 160))
