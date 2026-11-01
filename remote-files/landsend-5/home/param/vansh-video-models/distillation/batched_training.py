"""Batch execution helpers; optimizer-update boundaries remain with the caller.

Prepared rows and retained full-GT contexts contain batch-leading tensors and
constant scalar settings. Concatenation/slicing keeps all conditioning aligned
with each generated target, including noise, reference masks and dropout.
"""

from contextlib import contextmanager

import torch


OBJECTIVES = ('stabilizer', 'resolver')


def concatenate_batch_trees(values):
    values = list(values)
    if not values:
        raise ValueError('Cannot concatenate an empty batch tree')
    first = values[0]
    if isinstance(first, torch.Tensor):
        if first.ndim == 0:
            raise ValueError('Batch tensors must have a leading row dimension')
        return torch.cat(values, dim=0) if len(values) > 1 else first
    if isinstance(first, dict):
        if any(set(value) != set(first) for value in values):
            raise ValueError('Batch tree dictionary keys differ')
        return {key: concatenate_batch_trees(value[key] for value in values) for key in first}
    if isinstance(first, tuple):
        if any(len(value) != len(first) for value in values):
            raise ValueError('Batch tree tuple lengths differ')
        return tuple(concatenate_batch_trees(value[index] for value in values)
                     for index in range(len(first)))
    if any(value != first for value in values[1:]):
        raise ValueError('Batch tree scalar settings differ')
    return first


def slice_batch_tree(value, start, stop):
    if isinstance(value, torch.Tensor):
        return value[start:stop]
    if isinstance(value, dict):
        return {key: slice_batch_tree(item, start, stop) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(slice_batch_tree(item, start, stop) for item in value)
    return value


def regroup_prepared_rows(raw_group, prepared, batch_rows_by_objective):
    """Regroup already-prepared rows, preserving their audio augmentation."""
    if len(raw_group) != len(prepared):
        raise ValueError('Raw and prepared microbatch counts differ')
    result = []
    for kind in OBJECTIVES:
        rows = concatenate_batch_trees(rows for raw, rows in zip(raw_group, prepared)
                                       if raw['objective'] == kind)
        count, size = len(rows['latents']), batch_rows_by_objective[kind]
        if type(size) is not int or size < 1 or count % size:
            raise ValueError('Each objective must divide into complete local microbatches')
        result.extend(slice_batch_tree(rows, start, start + size)
                      for start in range(0, count, size))
    return result


def generate_pooled_targets(groups, generate, batch_rows):
    """Generate once per objective, then restore each critic update's rows.

``groups`` contains one list of ``(objective, prepared_rows)`` per optimizer
update. All updates use the same fixed student; no endpoint is reused for a
different source row. Output has the same update/microbatch nesting. A cap
below the combined size makes several complete generation batches per mode.
"""
    if type(batch_rows) is not int or batch_rows < 1:
        raise ValueError('Rollout pool batch rows must be a positive integer')
    result = [[None] * len(group) for group in groups]
    with torch.no_grad():
        for kind in OBJECTIVES:
            entries = [(update, index, rows)
                       for update, group in enumerate(groups)
                       for index, (objective, rows) in enumerate(group) if objective == kind]
            if not entries:
                raise ValueError(f'Rollout pool lacks {kind}')
            rows = concatenate_batch_trees(entry[2] for entry in entries)
            count = len(rows['latents'])
            size = min(batch_rows, count)
            if count % size:
                raise ValueError('Rollout pool must divide into complete local generation batches')
            outputs = []
            for start in range(0, count, size):
                outputs.append(generate(slice_batch_tree(rows, start, start + size), kind))
            pool = concatenate_batch_trees(outputs)
            start = 0
            for update, index, original in entries:
                stop = start + len(original['latents'])
                result[update][index] = slice_batch_tree(pool, start, stop)
                start = stop
            del rows, outputs, pool
    return result


@contextmanager
def objective_checkpoint(net, mode):
    """Change checkpoint policy without replacing the shared compiled callable."""
    if mode not in ('block', 'call', 'none'):
        raise ValueError(f'Unsupported objective checkpoint mode: {mode}')
    previous = net.checkpoint
    net.checkpoint = mode
    try:
        yield
    finally:
        net.checkpoint = previous
