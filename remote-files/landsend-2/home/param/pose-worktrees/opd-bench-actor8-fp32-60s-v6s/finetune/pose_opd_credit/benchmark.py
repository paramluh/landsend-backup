"""Explicit unsaved benchmark helpers; no production trajectory reuse."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def cache_identity(identity, *, update, rank):
    """Bind trajectories while allowing learner batch/gradient-sync changes."""
    method = dict(identity['method'])
    method['rollout_batch_per_rank'] = method.get('rollout_batch_per_rank') or method['physical_microbatch']
    del method['physical_microbatch']
    for key in ('decoder_gradient_sync', 'decoder_chunk_frames', 'text_chunk_frames'):
        method.pop(key, None)
    fields = ('source_sha256', 'plan_digest', 'dataset', 'models', 'world', 'actor',
              'model_configs', 'voxtral', 'asset_config_sha256', 'benchmark_prefix_frames')
    return json.loads(_json({**{key: identity.get(key) for key in fields}, 'method': method,
        'policy_probes': identity.get('policy_agreement_frames') is not None,
        'update': int(update), 'rank': int(rank), 'format': 'pose-opd-rollout-cache-v1'}))


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path, value, rank):
    temporary = path.with_name(path.name + f'.rank{rank}.tmp')
    with temporary.open('w') as stream:
        stream.write(_json(value) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def save_rollout_rank(root, identity, payload, rollouts):
    """Write completed CPU rollouts; caller globally agrees receipts afterwards."""
    import torch
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rank = identity['rank']
    path = root / f'rank-{rank:05d}.pt'
    marker = root / f'rank-{rank:05d}.complete.json'
    if path.exists() or marker.exists() or (root / 'COMPLETE').exists():
        raise ValueError(f'Refusing to overwrite an existing rollout cache: {root}')
    if len(payload) != len(rollouts) or len(payload) != identity['method']['effective_batch_per_rank']:
        raise ValueError('Rollout cache must contain the complete effective per-rank batch')
    if any(features.device.type != 'cpu' for _, features in payload):
        raise ValueError('Rollout caches require CPU audio features')
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        torch.save({'identity': identity, 'payload': payload, 'rollouts': rollouts}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    receipt = {'rank': rank, 'file': path.name, 'bytes': path.stat().st_size,
               'sha256': _digest(path), 'identity_sha256': hashlib.sha256(_json(identity).encode()).hexdigest()}
    _atomic_json(marker, receipt, rank)
    return receipt


def commit_rollout_cache(root, identity, receipts):
    """Call on every rank AFTER successful global receipt agreement.

    Each node may have its own local directory. Identical global manifests are
    therefore installed by every rank, using rank-specific atomic temporary files.
    """
    world = identity['world']
    if len(receipts) != world or [value['rank'] for value in receipts] != list(range(world)):
        raise ValueError('Cache completion requires one ordered receipt from every rank')
    common = {key: value for key, value in identity.items() if key != 'rank'}
    for rank, receipt in enumerate(receipts):
        expected = hashlib.sha256(_json({**common, 'rank': rank}).encode()).hexdigest()
        if receipt['identity_sha256'] != expected:
            raise ValueError('Rollout cache ranks disagree on trajectory identity')
    _atomic_json(Path(root) / 'COMPLETE', {'identity': common, 'receipts': receipts}, identity['rank'])


def validate_rollout_cache(root, identity, *, verify_bytes=False):
    root = Path(root)
    if not (root / 'COMPLETE').is_file():
        raise ValueError('Rollout cache has no globally agreed COMPLETE manifest')
    doc = json.loads((root / 'COMPLETE').read_text())
    common = {key: value for key, value in identity.items() if key != 'rank'}
    if doc['identity'] != common:
        raise ValueError('Rollout cache changes source, models, data, plan or sampling policy')
    rank = identity['rank']
    receipts = doc['receipts']
    if len(receipts) != identity['world'] or [value['rank'] for value in receipts] != list(range(identity['world'])):
        raise ValueError('Rollout cache completion manifest is incomplete')
    receipt = receipts[rank]
    marker = root / f'rank-{rank:05d}.complete.json'
    if not marker.is_file() or json.loads(marker.read_text()) != receipt:
        raise ValueError('Rollout cache rank receipt is missing or inconsistent')
    if receipt['file'] != f'rank-{rank:05d}.pt':
        raise ValueError('Unexpected rollout cache shard name')
    expected = hashlib.sha256(_json(identity).encode()).hexdigest()
    if receipt['identity_sha256'] != expected:
        raise ValueError('Rollout cache rank identity mismatch')
    path = root / receipt['file']
    if not path.is_file() or path.stat().st_size != receipt['bytes']:
        raise ValueError('Rollout cache shard missing or truncated')
    if verify_bytes and _digest(path) != receipt['sha256']:
        raise ValueError('Rollout cache shard SHA256 mismatch')
    return path


def load_rollout_rank(root, identity):
    import torch
    path = validate_rollout_cache(root, identity, verify_bytes=True)
    # Explicit user-selected, hash-verified local benchmark artifact contains
    # numpy arrays and our Rollout dataclass, not just a tensor state dictionary.
    doc = torch.load(path, map_location='cpu', weights_only=False)
    if doc['identity'] != identity:
        raise ValueError('Rollout cache payload identity mismatch')
    payload, rollouts = doc['payload'], doc['rollouts']
    if len(payload) != len(rollouts) or len(payload) != identity['method']['effective_batch_per_rank']:
        raise ValueError('Rollout cache batch is incomplete')
    for (row, features), rollout in zip(payload, rollouts):
        n, prefix = int(row['conversation_frames']), int(row['P'][0])
        if (tuple(features.shape) != (1, 4 * n, 1280) or len(rollout.text) != prefix + n + 2
                or rollout.prefix != prefix or rollout.conversation_frames != n):
            raise ValueError('Rollout cache row timeline or audio features are inconsistent')
    return payload, rollouts


def gradient_statistics(parameters, device):
    """Small global checksums, not a proof of elementwise gradient equality."""
    import torch
    import torch.distributed as dist
    totals = torch.zeros(3, dtype=torch.float64, device=device)
    maximum = torch.zeros((), dtype=torch.float64, device=device)
    for parameter in parameters:
        if parameter.grad is None:
            continue
        value = parameter.grad.detach()
        value = value.to_local() if hasattr(value, 'to_local') else value
        totals[0] += value.sum(dtype=torch.float64)
        totals[1] += value.float().square().sum(dtype=torch.float64)
        totals[2] += value.numel()
        if value.numel():
            maximum = torch.maximum(maximum, value.abs().max().double())
    if dist.is_initialized():
        dist.all_reduce(totals)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    return {'sum': float(totals[0]), 'l2': float(totals[1].sqrt()),
            'max_abs': float(maximum), 'elements': int(totals[2]),
            'comparison': 'summary_checksums_not_elementwise_equality'}


def replay_candidates(config, sizes, optimizer):
    """Clear candidate gradients without advancing weights or optimizer state.

    The caller replays the same bundles on each yielded configuration and may
    step the optimizer only after this iterator finishes, using the last grads.
    """
    from dataclasses import replace
    for size in sizes or (config.physical_microbatch,):
        if sizes is not None:
            optimizer.zero_grad(set_to_none=True)
        yield replace(config, physical_microbatch=size).validate()
