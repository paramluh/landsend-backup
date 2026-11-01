"""A full epoch of paired, single-target full-GT distillation examples.

Each training row is [sink | six GT predecessors | one target], with nine actor
references loaded separately. Every eligible target appears exactly once in the
student epoch for EACH objective. Stabilizer and resolver use the same row IDs
and reference draws. Temporal history dropout is deliberately absent here: the
rollout owns one sticky mask shared by its student, teacher, and critic queries.

Critic updates use extra, deterministic batches from the same shuffled corpus.
They do not advance the student epoch. Within one round, all critic batches and
the student batch have disjoint target IDs; the constructor rejects corpora too
small to satisfy that contract. Critic examples may recur in other rounds.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from dataloading.ds_aloe import AloeChunkSource, REF_DROP_STREAM, REF_PICK_STREAM
from dataloading.ds_dense import SHUFFLE_SEED
from dataloading.sources import NUM_REFS
from distillation.joint_data import JointEpochDataset, _sha256_json


OBJECTIVES = ('stabilizer', 'resolver')
DATA_HISTORY = 6
TARGETS_PER_ROW = 1
PLAN_VERSION = 1


def concatenate_full_gt_batches(batches):
    """Combine local microbatches of ONE objective and ONE optimizer update.

    This changes execution grouping, not the epoch plan. In particular, the
    objective's global denominator is copied, never summed. Cross-update pools
    should concatenate prepared device rows and retain these raw dictionaries
    separately so each critic update keeps its own normalization and identity.
    """
    batches = list(batches)
    if not batches:
        raise ValueError('Cannot concatenate an empty full-GT batch list')
    first = batches[0]
    keys = set(first)
    if any(set(batch) != keys for batch in batches):
        raise ValueError('Full-GT microbatch fields differ')
    variable = {'microbatch_index', 'objective_microbatch_index', 'global_microbatch_rows'}
    result = {}
    for key, value in first.items():
        if isinstance(value, torch.Tensor):
            for batch in batches:
                item = batch[key]
                if (not isinstance(item, torch.Tensor) or item.ndim == 0
                        or item.shape[0] != len(batch['sample_weight'])):
                    raise ValueError(f'Full-GT tensor {key} is not row-aligned')
            result[key] = torch.cat([batch[key] for batch in batches], dim=0)
        elif key not in variable:
            if any(batch[key] != value for batch in batches[1:]):
                raise ValueError(f'Cannot combine different full-GT {key}')
            result[key] = dict(value) if isinstance(value, dict) else value
    result['global_microbatch_rows'] = sum(int(batch['global_microbatch_rows']) for batch in batches)
    result['microbatch_index'] = int(first['microbatch_index'])
    result['objective_microbatch_index'] = int(first['objective_microbatch_index'])
    if 'window_source' in result:
        result['window_source'] = torch.arange(len(result['sample_weight']),
                                              device=result['window_source'].device)
    return result


def split_full_gt_batch(batch, batch_rows):
    """Split a local objective batch without losing global tail counts.

    ``global_row_valid_counts`` contains the number of valid rows across ranks
    at each local position. Local weights alone cannot recover these counts:
    a rank may contain only padding while other ranks contain real examples.
    """
    if type(batch_rows) is not int or batch_rows < 1:
        raise ValueError('Full-GT microbatch rows must be a positive integer')
    count = len(batch['sample_weight'])
    if count == 0 or count % batch_rows:
        raise ValueError('Full-GT local objective rows must divide evenly into microbatches')
    if 'global_row_valid_counts' not in batch:
        raise ValueError('Full-GT regrouping requires global_row_valid_counts')
    if int(batch['global_row_valid_counts'].sum()) != int(batch['global_microbatch_rows']):
        raise ValueError('Full-GT global row counts disagree with microbatch metadata')
    output = []
    for index, start in enumerate(range(0, count, batch_rows)):
        item = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                if value.ndim == 0 or value.shape[0] != count:
                    raise ValueError(f'Full-GT tensor {key} is not row-aligned')
                item[key] = value[start:start + batch_rows]
            else:
                item[key] = dict(value) if isinstance(value, dict) else value
        item['global_microbatch_rows'] = int(item['global_row_valid_counts'].sum())
        item['microbatch_index'] = index
        item['objective_microbatch_index'] = index
        if 'window_source' in item:
            item['window_source'] = torch.arange(batch_rows, device=item['window_source'].device)
        output.append(item)
    return output


def regroup_full_gt_group(raw_group, batch_rows_by_objective):
    """Keep a paired optimizer update intact while changing each phase's MBS."""
    if set(batch_rows_by_objective) != set(OBJECTIVES):
        raise ValueError('Full-GT regrouping requires stabilizer and resolver microbatch sizes')
    raw_group = list(raw_group)
    if not raw_group:
        raise ValueError('Cannot regroup an empty full-GT optimizer update')
    # Reject a mixed or reordered update before any tensor copies are made.
    kinds = [batch['objective'] for batch in raw_group]
    if any(kind not in OBJECTIVES for kind in kinds) or kinds != sorted(kinds, key=OBJECTIVES.index):
        raise ValueError('Full-GT objective order changed')
    for field in ('group_index', 'phase', 'critic_update_index', 'global_valid_rows', 'global_objective_rows'):
        if any(batch.get(field) != raw_group[0].get(field) for batch in raw_group[1:]):
            raise ValueError(f'Full-GT optimizer update contains mixed {field}')
    output = []
    for objective in OBJECTIVES:
        combined = concatenate_full_gt_batches(batch for batch in raw_group if batch['objective'] == objective)
        for batch in split_full_gt_batch(combined, batch_rows_by_objective[objective]):
            batch['microbatch_index'] = len(output)
            output.append(batch)
    return output


class FullGTEpochDataset(JointEpochDataset):
    """Prebatched rows for DataLoader(..., batch_size=None).

    ``group(index)`` returns all stabilizer microbatches, followed by all
    resolver microbatches, in ``objective_schedule`` order.
    ``critic_group(index, update)`` returns the same layout for the zero-based
    critic update. ``data[index]`` returns a dictionary with ``student`` holding
    the student group and ``critic`` holding all critic groups, so DataLoader
    workers can prefetch the whole round. The two objectives have identical
    source rows and refs. Their CPU tensors are shared and must be treated as
    immutable; objective preparation may create independent device tensors.

    ``global_batch_size`` counts single target frames, hence also source rows,
    per objective. The final student batch uses zero-weight padding, and each
    objective must be normalized by its actual ``global_objective_rows`` count.
    Critic batches are always full. All ranks execute the complete schedule,
    even when a rank's student tail contains only padding.

    Only the common actor-reference loading, source-window gathering, and
    canonical sensitivity-evaluation helpers are inherited. Training planning,
    row geometry, and history-mask behavior do not use the joint four-target
    dataset. Save ``signature`` plus the next student group index on resume.
    """

    def __init__(self, source, rank, world_size, batch_rows, *, seed=SHUFFLE_SEED,
                 epoch=0, reference_dropout=None,
                 stabilizer_history_dropout=(0., .1, .2, .2),
                 csv_sha256=None,
                 global_batch_size=160, critic_updates=4, eligibility_history=6):
        if NUM_REFS != 9:
            raise ValueError('Full-GT distillation requires the nine-reference actor bank')
        if (type(world_size) is not int or type(rank) is not int
                or type(batch_rows) is not int or world_size < 1
                or not 0 <= rank < world_size or batch_rows < 1):
            raise ValueError('Invalid global rank/world size/local row batch')
        if type(global_batch_size) is not int or global_batch_size < 1:
            raise ValueError('global_batch_size must be a positive number of targets per objective')
        if type(critic_updates) is not int or critic_updates < 1:
            raise ValueError('critic_updates must be a positive integer')
        if type(eligibility_history) is not int or eligibility_history < DATA_HISTORY:
            raise ValueError('eligibility_history must be an integer of at least six')
        if type(seed) is not int or type(epoch) is not int or seed < 0 or epoch < 0:
            raise ValueError('Epoch and seed must be nonnegative integers')
        if not source.bank:
            raise ValueError('A separate nine-reference actor bank is required')

        self.source = source
        self.rank = rank
        self.world_size = world_size
        self.batch_rows = batch_rows
        self.seed = seed
        self.epoch = epoch
        self.critic_updates = critic_updates
        self.eligibility_history = eligibility_history
        self.global_batch_size = global_batch_size
        self.global_batch_rows = global_batch_size
        self.global_microbatch_rows = world_size * batch_rows
        if self.global_batch_rows % self.global_microbatch_rows:
            raise ValueError(
                f'global_batch_size={global_batch_size} single-target rows per objective '
                f'must divide exactly by world_size * micro_batch_size_per_gpu '
                f'({world_size} * {batch_rows} = {self.global_microbatch_rows}). '
                'Choose a divisible topology or a smaller per-GPU microbatch.')
        self.objective_microbatches = self.global_batch_rows // self.global_microbatch_rows
        self.objective_schedule = [objective for objective in OBJECTIVES
                                   for _ in range(self.objective_microbatches)]
        self.microbatches_per_group = len(self.objective_schedule)
        self.group_capacity = self.global_batch_rows

        # Frame zero is the supplied sink. With six predecessors, the earliest
        # target is source frame seven; eligibility_history=16 instead gives the
        # recovered opposite-limits experiment's target eligibility (f >= 17),
        # while still loading only six predecessors into each full-GT row.
        counts = np.maximum(0, np.asarray(source.lengths, dtype=np.int64)
                            - eligibility_history - 1)
        self._prefix = np.zeros(len(counts) + 1, dtype=np.int64)
        np.cumsum(counts, out=self._prefix[1:])
        self.eligible_rows = int(self._prefix[-1])
        minimum_rows = (critic_updates + 1) * self.global_batch_rows
        if self.eligible_rows < minimum_rows:
            raise ValueError(
                f'Full-GT corpus has {self.eligible_rows} eligible target rows, but '
                f'{minimum_rows} are required for one student batch and {critic_updates} '
                'disjoint critic batches. Use a larger full-corpus CSV or reduce '
                'global_batch_size/critic_updates; rows are never silently reused '
                'within a round.')
        self.num_groups = (self.eligible_rows + self.group_capacity - 1) // self.group_capacity
        self._order = np.random.default_rng(
            np.random.SeedSequence([seed, epoch])).permutation(self.eligible_rows)

        drop_counts, probabilities = reference_dropout or ([0], [1.0])
        if (not len(drop_counts) or len(drop_counts) != len(probabilities)
                or any(int(n) != n or not 0 <= n <= NUM_REFS for n in drop_counts)
                or any(not np.isfinite(p) or p < 0 for p in probabilities)
                or abs(sum(probabilities) - 1.0) > 1e-6):
            raise ValueError('Invalid reference dropout counts/probabilities')
        self._ref_drop_counts = np.asarray(drop_counts, dtype=np.int64)
        self._ref_drop_cdf = np.cumsum(probabilities)
        if (len(stabilizer_history_dropout) != 4
                or any(not np.isfinite(p) or not 0 <= p <= 1
                       for p in stabilizer_history_dropout)):
            raise ValueError('Stabilizer history dropout requires four independent probabilities in [0, 1]')
        records = [
            {'video': clip.video_path, 'pose': clip.pose_path, 'mimi': clip.mimi_path,
             'length': int(clip.length), 'uuid': clip.uuid}
            for clip in source.clips
        ]
        if len({record['video'] for record in records}) != len(records):
            raise ValueError('Full-corpus CSV contains duplicate clip paths; explicit deduplication is required')
        self.signature = {
            'dataset': 'full_gt_single_target_nine_reference',
            'plan_version': PLAN_VERSION,
            'csv_sha256': csv_sha256,
            'ordered_source_sha256': _sha256_json(records),
            'reference_bank': str(source.bank),
            'reference_drop_counts': self._ref_drop_counts.tolist(),
            'reference_drop_p': list(probabilities),
            'reference_pick_stream': REF_PICK_STREAM,
            'reference_drop_stream': REF_DROP_STREAM,
            'reference_rng_policy': 'seed_epoch_canonical_target_id_stream',
            'stabilizer_history_dropout': list(stabilizer_history_dropout),
            'temporal_dropout_owner': 'rollout_sticky_mask_not_dataset',
            'temporal_dropout_policy': 'same_frame_identity_mask_for_all_queries_of_each_sample',
            'seed': seed,
            'epoch': epoch,
            'shuffle_policy': 'numpy_default_rng_seedsequence_seed_epoch_permutation',
            'world_size': world_size,
            'batch_rows_per_rank': batch_rows,
            'global_batch_size_targets_per_objective': global_batch_size,
            'global_batch_rows_per_objective': self.global_batch_rows,
            'global_microbatch_rows': self.global_microbatch_rows,
            'objectives': list(OBJECTIVES),
            'objective_schedule': list(self.objective_schedule),
            'objective_microbatches': self.objective_microbatches,
            'microbatches_per_group': self.microbatches_per_group,
            'paired_window_ids_across_objectives': True,
            'paired_reference_draws_across_objectives': True,
            'epoch_coverage': 'every_eligible_target_once_per_student_objective',
            'global_valid_rows_semantics': 'unique_target_rows_not_sum_over_objectives',
            'eligible_rows': self.eligible_rows,
            'num_groups': self.num_groups,
            'targets_per_row': TARGETS_PER_ROW,
            'source_history_frames': DATA_HISTORY,
            'eligibility_history_frames': eligibility_history,
            'first_eligible_target': eligibility_history + 1,
            'tail_normalization': 'separate_actual_global_objective_rows',
            'student_tail_policy': 'zero_weight_padding_no_repeat_contribution',
            'critic_updates': critic_updates,
            'critic_policy': 'successive_circular_blocks_after_student_block_in_epoch_shuffle',
            'critic_batches_full': True,
            'critic_disjoint_from_student_and_other_critic_updates_within_round': True,
            'critic_sampling_advances_student_epoch': False,
            'minimum_eligible_rows': minimum_rows,
        }
        self.signature['plan_sha256'] = _sha256_json(self.signature)

    def _window(self, window_id, *, resolver=False):
        """Canonical, unmasked source row; resolver uses the identical geometry."""
        if not 0 <= window_id < self.eligible_rows:
            raise IndexError(window_id)
        clip = int(np.searchsorted(self._prefix, window_id, side='right') - 1)
        ordinal = int(window_id - self._prefix[clip])
        target = self.eligibility_history + 1 + ordinal
        ofi = torch.cat((torch.zeros(1, dtype=torch.long),
                         torch.arange(target - DATA_HISTORY, target + 1)))
        return clip, ofi

    def _check_group(self, index):
        if not isinstance(index, (int, np.integer)) or not 0 <= index < len(self):
            raise IndexError(f'Group {index} is outside the fixed epoch of {len(self)} groups')

    def _batches(self, ids, index, *, critic_update=None):
        count = len(ids)
        layout = np.full(self.group_capacity, -1, dtype=np.int64)
        layout[:count] = ids
        layout = layout.reshape(self.objective_microbatches, self.global_microbatch_rows)
        valid_per_microbatch = (layout >= 0).sum(axis=1).tolist()
        objective_counts = {objective: count for objective in OBJECTIVES}
        batches = []
        lo = self.rank * self.batch_rows
        loaded = []
        for microbatch in range(self.objective_microbatches):
            local = layout[microbatch, lo:lo + self.batch_rows]
            valid = (local >= 0).astype(np.float32).tolist()
            row_ids = np.maximum(local, 0).tolist()
            loaded.append(self._load([self._window(row_id) for row_id in row_ids],
                                     row_ids, valid, targets=TARGETS_PER_ROW))
        for objective in OBJECTIVES:
            for microbatch in range(self.objective_microbatches):
                # Paired objectives load their common source assets once. Only
                # their dictionaries differ; temporal masks are sampled later.
                batch = dict(loaded[microbatch])
                batch.update(objective=objective, kind=objective, group_index=int(index),
                             microbatch_index=len(batches), objective_microbatch_index=microbatch,
                             global_valid_rows=count, global_objective_rows=dict(objective_counts),
                             global_microbatch_rows=int(valid_per_microbatch[microbatch]),
                             global_row_valid_counts=torch.from_numpy(
                                 (layout[microbatch].reshape(self.world_size, self.batch_rows) >= 0)
                                 .sum(axis=0)),
                             phase='student' if critic_update is None else 'critic')
                if critic_update is not None:
                    batch['critic_update_index'] = int(critic_update)
                batches.append(batch)
        return batches

    def group(self, index):
        """The unique student batch, paired across stabilizer and resolver."""
        self._check_group(index)
        first = int(index) * self.group_capacity
        ids = self._order[first:first + self.group_capacity]
        return self._batches(ids, index)

    def critic_group(self, index, update):
        """Extra full critic batch; update is in [0, critic_updates).

        A round reserves a block of global_batch_size positions for its student
        batch, followed by one full block per critic update. The positions wrap
        around the immutable shuffled corpus. Since the corpus has at least the
        size of all these blocks combined, target IDs never overlap within the
        round. A partial final student batch simply leaves its padding positions
        unused; they are not silently reused by a critic batch in that round.
        """
        self._check_group(index)
        if not isinstance(update, (int, np.integer)) or not 0 <= update < self.critic_updates:
            raise IndexError(f'Critic update {update} is outside [0, {self.critic_updates})')
        first = (int(index) + int(update) + 1) * self.group_capacity
        positions = (first + np.arange(self.group_capacity)) % self.eligible_rows
        return self._batches(self._order[positions], index, critic_update=update)

    def __getitem__(self, index):
        """Prefetch all five source batches without loading files on the GPU loop."""
        return {'student': self.group(index),
                'critic': [self.critic_group(index, update)
                           for update in range(self.critic_updates)]}


def build_full_gt_data(config, rank, world_size):
    """Build a common full-corpus source, never per-node shards or capped epochs."""
    if config.get('node_sharded'):
        raise ValueError('Full-GT full-epoch training requires a common full-corpus CSV; node_sharded is unsupported')
    if 'windows_per_epoch' in config:
        raise ValueError('windows_per_epoch would truncate the requested full epoch')
    if 'buckets' in config or 'llm_hiddens_root' in config:
        raise ValueError('Full-GT data supports the Aloe CSV + actor-reference-bank source only')
    if 'resolver_history_dropout' in config:
        raise ValueError('Full-GT uses stabilizer_history_dropout as one frame-identity mask policy '
                         'for both objectives; remove resolver_history_dropout')
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
    return FullGTEpochDataset(
        source, rank, world_size, config['micro_batch_size_per_gpu'],
        seed=config.get('seed', SHUFFLE_SEED), epoch=config.get('epoch', 0),
        reference_dropout=(refs.get('drop_counts', [0]), refs.get('drop_p', [1.0])),
        stabilizer_history_dropout=config.get('stabilizer_history_dropout', [0., .1, .2, .2]),
        csv_sha256=hashlib.sha256(csv.read_bytes()).hexdigest(),
        global_batch_size=config.get('global_batch_size', 160),
        critic_updates=config.get('critic_updates', 4),
        eligibility_history=config.get('eligibility_history', 6))
