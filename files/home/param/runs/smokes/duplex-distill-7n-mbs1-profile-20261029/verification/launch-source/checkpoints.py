"""Load pinned production weights and save resumable node-local training state.

A torch DCP directory may be *read* at a different world size because all of its
files have been staged on every node. New saves deliberately use per-rank files:
DCP's replicated-tensor deduplication and rank-0-only metadata are inappropriate
for node-local disks. Each node gets its own manifest, replicated states and
completion marker. Resume requires the exact original world and host mapping.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import socket
import stat
import time
import warnings

import torch
import torch.distributed as dist


_LAYER = re.compile(r'^(layers|tower_[sp]\.layers|dec_[sp]\.model\.layers)\.(\d+)\.')


def _local(value):
    return value.to_local() if hasattr(value, 'to_local') else value


def _canonical(name):
    return name.replace('_checkpoint_wrapped_module.', '')


def _state(model):
    from torch.distributed.checkpoint.state_dict import get_model_state_dict
    return {_canonical(k): v for k, v in get_model_state_dict(model).items()}


def _rows(tensor):
    if not hasattr(tensor, 'to_local'):
        return 0, tensor.shape[0]
    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
    if any(p.is_shard() and p.dim != 0 for p in tensor.placements):
        raise ValueError('Production rank exports support only dimension-zero sharding')
    shape, offset = compute_local_shape_and_global_offset(tensor.shape, tensor.device_mesh, tensor.placements)
    return (int(offset[0]) if offset else 0), (int(shape[0]) if shape else 0)


def _rank_world_hosts():
    rank, world = (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)
    host = socket.gethostname().split('.')[0]
    hosts = [None] * world
    if dist.is_initialized():
        dist.all_gather_object(hosts, host)
    else:
        hosts[0] = host
    return rank, world, hosts


def _barrier():
    if dist.is_initialized():
        dist.barrier()


def _all_objects(value):
    if not dist.is_initialized():
        return [value]
    result = [None] * dist.get_world_size()
    dist.all_gather_object(result, value)
    return result


def _check_keys(expected, available, smoke):
    missing = sorted(set(expected) - set(available))
    extra = sorted(set(available) - set(expected))
    depth = {}
    for name in expected:
        match = _LAYER.match(name)
        if match:
            depth[match[1]] = max(depth.get(match[1], 0), int(match[2]) + 1)
    allowed = []
    if smoke:
        for name in extra:
            match = _LAYER.match(name)
            if match and match[1] in depth and int(match[2]) >= depth[match[1]]:
                allowed.append(name)
    bad = sorted(set(extra) - set(allowed))
    if missing or bad:
        raise ValueError(f'Checkpoint/model keys differ: missing={missing[:10]}, unexpected={bad[:10]}')
    return allowed


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _atomic_json(path, doc):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(doc, indent=2, default=str) + '\n')
    os.replace(tmp, path)


@torch.no_grad()
def _restore_exact_gain(state, path, meta):
    if 'vgain.theta' not in state:
        return None
    side = path / 'vgain_theta_fp32.json'
    if side.is_file():
        doc = json.loads(side.read_text())
        theta, source = float(doc['vgain_theta_fp32']), str(side)
    elif 'vgain_theta_fp32' in meta:
        theta, source = float(meta['vgain_theta_fp32']), str(path / 'meta.json')
    else:
        raise ValueError(f'{path}: production model has vgain.theta but lacks exact fp32 gain metadata')
    local = _local(state['vgain.theta'])
    if local.numel():
        if not torch.isfinite(torch.tensor(theta)) or float((local.float() - theta).abs().max()) >= 0.04:
            raise ValueError(f'{path}: exact gain is inconsistent with saved bf16 model value')
        local.fill_(theta)
    return {'vgain_theta_fp32': theta, 'source': source}


@torch.no_grad()
def _finite_check(state, dev):
    bad = torch.zeros((), device=dev, dtype=torch.int64)
    for name, value in state.items():
        # Production's norm='none' adapter deliberately persists NaN as the
        # unused target-RMS sentinel (train_samsep_posedup.py:1264/1325).
        if name == 'vadapter.target_rms':
            continue
        local = _local(value)
        bad.add_((~torch.isfinite(local)).any().to(device=dev, dtype=torch.int64))
    if dist.is_initialized():
        dist.all_reduce(bad)
    if int(bad):
        raise ValueError('Checkpoint loaded nonfinite tensors')


@torch.no_grad()
def load_checkpoint(model, path, dev, *, smoke=False):
    """Load production DCP or ppm-local-bf16-shards-v1 weights in place.

    DCPs reshard through torch's distributed checkpoint loader. Rank exports
    concatenate source dim-0 shards in rank order and load only this rank's
    overlapping rows, without constructing full tensors. Only explicitly
    requested smoke loads may skip *deeper layer* keys; every other key and
    global shape must match. Exact fp32 Voxtral gains override bf16 exports.
    """
    path = Path(path)
    state = _state(model)
    t0 = time.monotonic()
    rank, world, hosts = _rank_world_hosts()
    if (path / '.metadata').is_file():
        import torch.distributed.checkpoint as dcp
        if not (path / 'COMPLETE').is_file():
            raise ValueError(f'Incomplete DCP: {path}')
        reader = dcp.FileSystemReader(path)
        metadata = reader.read_metadata().state_dict_metadata
        skipped = _check_keys(state, metadata, smoke)
        for name, tensor in state.items():
            if tuple(metadata[name].size) != tuple(tensor.shape):
                raise ValueError(f'DCP shape mismatch for {name}: {metadata[name].size} != {tensor.shape}')
        before = {k: _local(v).data_ptr() for k, v in state.items()}
        dcp.load(state, storage_reader=reader)
        for name, tensor in state.items():
            if _local(tensor).data_ptr() != before[name]:
                raise RuntimeError(f'DCP load replaced rather than filled model storage: {name}')
        meta = json.loads((path / 'train_cfg.json').read_text()) if (path / 'train_cfg.json').is_file() else {}
        kind = 'dcp'
    else:
        meta = json.loads((path / 'meta.json').read_text())
        if meta.get('format') != 'ppm-local-bf16-shards-v1':
            raise ValueError(f'Unknown checkpoint format: {meta.get("format")}')
        node_local = meta.get('storage_layout') == 'node-local-shards'
        if node_local:
            if not (path / 'NODE_COMPLETE').is_file():
                raise ValueError(f'Incomplete node-local export: {path}')
            if world != meta['world'] or hosts != meta['hosts']:
                raise ValueError('Gather all node-local export rank files before loading at a different world/host map')
        elif not (path / 'COMPLETE').is_file():
            raise ValueError(f'Incomplete rank export: {path}')
        source_world = int(meta['world'])
        if source_world < 1:
            raise ValueError('Source checkpoint world must be positive')
        sources = {}
        def source(i):
            if i not in sources:
                sources[i] = torch.load(path / f'rank{i}.pt', map_location='cpu', mmap=True, weights_only=True)
            return sources[i]
        replicas = torch.load(path / 'rep.pt', map_location='cpu', mmap=True, weights_only=True)
        available = set(source(rank if node_local else 0)) | set(replicas)
        skipped = _check_keys(state, available, smoke)
        for name, tensor in state.items():
            local = _local(tensor)
            if name in replicas:
                saved = replicas[name]
                if tuple(saved.shape) != tuple(tensor.shape):
                    raise ValueError(f'Replicated source shape mismatch: {name}')
                if hasattr(tensor, 'to_local') and tensor.ndim:
                    offset, count = _rows(tensor)
                    saved = saved[offset:offset + count]
                local.copy_(saved.to(device=local.device, dtype=local.dtype))
                continue
            if node_local:
                saved = source(rank)[name]
                if tuple(saved.shape) != tuple(local.shape):
                    raise ValueError(f'Node-local source shape mismatch: {name}')
                local.copy_(saved.to(device=local.device, dtype=local.dtype))
                continue
            if tensor.ndim == 0:
                raise ValueError(f'Scalar {name} unexpectedly sharded in source checkpoint')
            total = tensor.shape[0]
            old_chunk = (total + source_world - 1) // source_world
            offset, count = _rows(tensor)
            covered = 0
            if count:
                for old_rank in range(offset // old_chunk, min(source_world, (offset + count + old_chunk - 1) // old_chunk)):
                    low, high = old_rank * old_chunk, min((old_rank + 1) * old_chunk, total)
                    shard = source(old_rank)[name]
                    if tuple(shard.shape) != (high - low, *tensor.shape[1:]):
                        raise ValueError(f'Source rank {old_rank} shape mismatch for {name}: {shard.shape}')
                    a, b = max(offset, low), min(offset + count, high)
                    if a < b:
                        local[a - offset:b - offset].copy_(shard[a - low:b - low].to(device=local.device, dtype=local.dtype))
                        covered += b - a
            if covered != count:
                raise ValueError(f'Source shards did not cover all local rows of {name}')
        sources.clear()
        del replicas
        kind = meta['format']
    gain = _restore_exact_gain(state, path, meta)
    _finite_check(state, dev)
    return {'path': str(path), 'format': kind, 'step': meta.get('step'),
            'run': meta.get('run', meta.get('run_tag')), 'world': world,
            'source_world': meta.get('world', 'DCP'), 'tensors_loaded': len(state),
            'smoke': bool(smoke), 'skipped_deeper_layers': skipped,
            'exact_gain': gain, 'seconds': time.monotonic() - t0}


def _pack_value(value):
    if isinstance(value, torch.Tensor):
        return {'__tensor__': True, 'data': _local(value).detach().cpu().contiguous().clone(),
                'dtensor': hasattr(value, 'to_local'), 'shape': tuple(value.shape),
                'stride': tuple(value.stride()), 'device_type': _local(value).device.type}
    if isinstance(value, dict):
        return {k: _pack_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_pack_value(v) for v in value)
    return value


def _unpack_value(value, parameter):
    if isinstance(value, dict) and value.get('__tensor__') is True:
        target_device = _local(parameter).device if value['device_type'] != 'cpu' else torch.device('cpu')
        tensor = value['data'].to(target_device)
        if value['dtensor']:
            from torch.distributed.tensor import DTensor
            if not hasattr(parameter, 'device_mesh'):
                raise ValueError('Saved DTensor optimizer state now has an unsharded parameter')
            tensor = DTensor.from_local(tensor, parameter.device_mesh, parameter.placements, run_check=False,
                                        shape=torch.Size(value['shape']), stride=tuple(value['stride']))
        return tensor
    if isinstance(value, dict):
        return {k: _unpack_value(v, parameter) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_unpack_value(v, parameter) for v in value)
    return value


def _optimizer_state(model, optimizers, *, sharded):
    names = {id(p): _canonical(n) for n, p in model.named_parameters()}
    result = {}
    for name, optimizer in optimizers.items():
        groups = []
        states = {}
        for group in optimizer.param_groups:
            groups.append({**{k: v for k, v in group.items() if k != 'params'},
                           'params': [names[id(p)] for p in group['params']]})
            for p in group['params']:
                if hasattr(p, 'to_local') == sharded and p in optimizer.state:
                    states[names[id(p)]] = _pack_value(optimizer.state[p])
        result[name] = {'param_groups': groups, 'state': states}
    return result


def _tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return _local(value).numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(v) for v in value)
    return 0


def _rng_state():
    import numpy as np
    np_state = np.random.get_state()
    return {'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            'python': random.getstate(),
            'numpy': (np_state[0], np_state[1].tolist(), np_state[2], np_state[3], np_state[4])}


def _restore_rng(state):
    import numpy as np
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'])
    random.setstate(state['python'])
    ns = state['numpy']
    np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4]))


def _save_file(obj, path):
    torch.save(obj, path)
    return {'file': Path(path).name, 'bytes': Path(path).stat().st_size, 'sha256': _sha256(path)}


def _safe_checkpoint_root(run_dir):
    """Retention never follows symlinks or traverses outside this run."""
    run = Path(run_dir).absolute()
    if '..' in run.parts or any(part.is_symlink() for part in (run, *run.parents)):
        raise ValueError('Checkpoint retention refuses symlink/parent-traversal run paths')
    root = run / 'checkpoints'
    if root.is_symlink() or root.resolve() != run.resolve() / 'checkpoints':
        raise ValueError('Checkpoint retention refuses a symlink checkpoint root')
    return root


def _validate_resume_directory(root, directory, host, world, hosts):
    directory = Path(directory).absolute()
    match = re.fullmatch(r'step-(\d{6,})', directory.name)
    if (not match or directory.name != f'step-{int(match[1]):06d}'
            or directory.parent != root or directory.is_symlink()
            or directory.resolve().parent != root.resolve()):
        raise ValueError(f'Unsafe checkpoint step directory: {directory}')
    resume = directory / 'resume'
    if resume.is_symlink() or not resume.is_dir():
        raise ValueError(f'Unsafe or absent resume directory: {resume}')
    # Check every entry before opening metadata, so a malicious metadata link
    # cannot make validation read or remove an unrelated file.
    if any(entry.is_symlink() or not entry.is_file() for entry in resume.iterdir()):
        raise ValueError(f'Resume directory contains symlinks or nested entries: {resume}')
    meta = json.loads((resume / 'train_state.json').read_text())
    step = int(match[1])
    local_ranks = [rank for rank, value in enumerate(hosts) if value == host]
    if (meta.get('format') != 'pose-duplex-distill-resume-v1'
            or meta.get('storage_layout') != 'node-local-shards'
            or meta.get('step') != step or meta.get('host') != host
            or meta.get('world') != world or meta.get('hosts') != hosts
            or meta.get('local_ranks') != local_ranks
            or meta.get('run_dir') != str(root.parent.resolve())
            or (resume / 'COMPLETE').read_text().strip() != str(step)):
        raise ValueError(f'Resume manifest does not belong to this completed run: {resume}')
    entries = [entry for entry in meta['files'] if entry['section'] == 'resume' and entry['host'] == host]
    names = [entry['file'] for entry in entries]
    expected = {'rep.pt', *[f'rank{rank}.pt' for rank in local_ranks]}
    if set(names) != expected or len(names) != len(expected):
        raise ValueError(f'Resume manifest has unexpected or duplicate filenames: {resume}')
    if {entry.name for entry in resume.iterdir()} != expected | {'COMPLETE', 'train_state.json'}:
        raise ValueError(f'Resume directory has files not owned by the checkpoint: {resume}')
    for entry in entries:
        file = resume / entry['file']
        if (not isinstance(entry['bytes'], int) or entry['bytes'] <= 0
                or file.stat().st_size != entry['bytes']
                or not re.fullmatch(r'[0-9a-f]{64}', entry['sha256'])):
            raise ValueError(f'Resume file size/hash manifest invalid: {file}')
    return meta, entries


def _validate_preserved_export(directory, meta, host):
    export = directory / 'export'
    if export.is_symlink() or not export.is_dir():
        raise ValueError('Cannot prune a resume without its separate weight export')
    if any(entry.is_symlink() or not entry.is_file() for entry in export.iterdir()):
        raise ValueError('Weight export contains unsafe entries')
    saved = json.loads((export / 'meta.json').read_text())
    complete = json.loads((export / 'NODE_COMPLETE').read_text())
    if (saved.get('format') != 'ppm-local-bf16-shards-v1'
            or saved.get('storage_layout') != 'node-local-shards'
            or any(saved.get(key) != meta[key] for key in ('step', 'world', 'hosts', 'host', 'run_dir'))
            or complete.get('step') != meta['step'] or complete.get('host') != host):
        raise ValueError('Separate weight export does not match the old resume')
    entries = [entry for entry in saved['files'] if entry['host'] == host]
    expected = {'rep.pt', *[f'rank{rank}.pt' for rank in meta['local_ranks']]}
    if {entry['file'] for entry in entries} != expected or len(entries) != len(expected):
        raise ValueError('Separate weight export has an invalid file manifest')
    for entry in entries:
        if (export / entry['file']).stat().st_size != entry['bytes']:
            raise ValueError('Separate weight export is incomplete')


def _remove_flat_resume(resume, names):
    """Remove only the validated checkpoint files through a no-follow dir fd."""
    fd = os.open(resume, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        if set(os.listdir(fd)) != set(names):
            raise ValueError('Resume files changed after validation')
        for name in names:
            if Path(name).name != name or not stat.S_ISREG(os.stat(name, dir_fd=fd, follow_symlinks=False).st_mode):
                raise ValueError('Resume contains a non-regular file')
        # A partially pruned old directory must never advertise completeness.
        for name in ['COMPLETE', *sorted(set(names) - {'COMPLETE'})]:
            os.unlink(name, dir_fd=fd)
    finally:
        os.close(fd)
    resume.rmdir()


def _write_latest_marker(root, marker):
    temporary = root / f'.LATEST_RESUME.{os.getpid()}.{time.time_ns()}.tmp'
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(marker, stream, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, root / 'LATEST_RESUME.json')
    finally:
        if temporary.exists():
            temporary.unlink()


def _commit_latest_resume(run_dir, directory, rank, world, hosts, keep_latest):
    """Verify the new save on all ranks, commit all node markers, then prune."""
    host = hosts[rank]
    leader = rank == hosts.index(host)
    error = None
    try:
        root = _safe_checkpoint_root(run_dir)
        meta, entries = _validate_resume_directory(root, directory, host, world, hosts)
        owned = {f'rank{rank}.pt'} | ({'rep.pt'} if leader else set())
        for entry in entries:
            if entry['file'] in owned and _sha256(Path(directory) / 'resume' / entry['file']) != entry['sha256']:
                raise ValueError(f'New resume checksum mismatch: {entry["file"]}')
    except Exception as exc:
        error = f'{host} rank{rank}: {exc}'
    errors = [value for value in _all_objects(error) if value]
    if errors:
        raise RuntimeError('New checkpoint verification failed; previous resumes retained: ' + '; '.join(errors))
    marker_error = None
    if leader:
        try:
            _write_latest_marker(root, {'format': 'pose-duplex-distill-latest-resume-v1',
                'step': meta['step'], 'resume_path': str(Path(directory).absolute() / 'resume'),
                'relative_resume_path': f'{Path(directory).name}/resume',
                'host': host, 'world': world, 'hosts': hosts,
                'manifest_sha256': _sha256(Path(directory) / 'resume' / 'train_state.json'),
                'verified_unix': time.time()})
        except Exception as exc:
            marker_error = f'{host}: latest marker not committed; old resumes retained: {exc}'
    marker_errors = [value for value in _all_objects(marker_error) if value]
    # No node may prune until every node has committed the new latest marker.
    if marker_errors:
        if rank == 0:
            warnings.warn('; '.join(marker_errors), RuntimeWarning)
        return {'committed': False, 'pruned': [], 'warnings': marker_errors}
    pruned, issues = [], []
    if leader and keep_latest:
        for old in sorted(root.iterdir()):
            match = re.fullmatch(r'step-(\d{6,})', old.name)
            if not match or int(match[1]) >= meta['step'] or not (old / 'resume').exists():
                continue
            try:
                previous, files = _validate_resume_directory(root, old, host, world, hosts)
                _validate_preserved_export(old, previous, host)
                _remove_flat_resume(old / 'resume', {entry['file'] for entry in files} | {'COMPLETE', 'train_state.json'})
                pruned.append({'host': host, 'resume_path': str(old / 'resume'), 'step': previous['step']})
            except Exception as exc:
                issues.append(f'{host}: retaining old checkpoint {old.name}: {exc}')
    reports = _all_objects({'pruned': pruned, 'warnings': issues})
    all_issues = sum((report['warnings'] for report in reports), [])
    if all_issues and rank == 0:
        warnings.warn('; '.join(all_issues), RuntimeWarning)
    return {'committed': True, 'marker': str(root / 'LATEST_RESUME.json'),
            'pruned': sum((report['pruned'] for report in reports), []), 'warnings': all_issues}


@torch.no_grad()
def save_checkpoint(model, optimizers, run_dir, step, cfg):
    """Atomic same-world resume state plus portable bf16 rank-weight exports.

    ``optimizers`` is a name->torch.optim.Optimizer mapping. Full parameter and
    optimizer-state dtypes, optimizer group settings, rank RNG states, cfg and
    cfg['_runtime_state'] are preserved. Replicated model/optimizer tensors are
    saved once PER NODE. Every node receives completion markers only after all
    ranks have successfully written files. After global verification, each node
    atomically records LATEST_RESUME.json. If keep_latest_optimizer_state is set,
    only older validated resume directories in this run are removed; their bf16
    exports remain. Pruning errors retain the old directory and warn.
    """
    rank, world, hosts = _rank_world_hosts()
    host = hosts[rank]
    leader = rank == hosts.index(host)
    final = Path(run_dir) / 'checkpoints' / f'step-{int(step):06d}'
    tmp = final.with_name(final.name + '.tmp')
    state = _state(model)
    model_bytes = sum(_tensor_bytes(v) for v in state.values() if hasattr(v, 'to_local') or leader)
    optimizer_bytes = sum(_tensor_bytes(st) for opt in optimizers.values() for p, st in opt.state.items()
                          if hasattr(p, 'to_local') or leader)
    export_bytes = sum(_local(v).numel() * (2 if v.is_floating_point() else v.element_size())
                       for v in state.values() if hasattr(v, 'to_local') or leader)
    sizes = _all_objects(model_bytes + optimizer_bytes + export_bytes)
    errors = []
    if leader:
        final.parent.mkdir(parents=True, exist_ok=True)
        if final.exists() or tmp.exists():
            errors.append(f'{host}: checkpoint path already exists: {final} or {tmp}')
        needed = sum(n for n, h in zip(sizes, hosts) if h == host) * 1.10 + float(cfg.get('checkpoint_reserve_gb', 2)) * 1e9
        free = shutil.disk_usage(final.parent).free
        if free < needed:
            errors.append(f'{host}: checkpoint requires {needed / 1e9:.1f} GB free; has {free / 1e9:.1f} GB')
    all_errors = sum(_all_objects(errors), [])
    if all_errors:
        raise RuntimeError('; '.join(all_errors))
    if leader:
        (tmp / 'resume').mkdir(parents=True)
        (tmp / 'export').mkdir()
    _barrier()
    entries = []
    error = None
    try:
        sharded = {k: _local(v).detach().cpu().contiguous().clone() for k, v in state.items() if hasattr(v, 'to_local')}
        full = {'model': sharded, 'optimizers': _optimizer_state(model, optimizers, sharded=True), 'rng': _rng_state()}
        entry = _save_file(full, tmp / 'resume' / f'rank{rank}.pt');entry.update(host=host, rank=rank, section='resume');entries.append(entry)
        del full
        export = {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in sharded.items()}
        entry = _save_file(export, tmp / 'export' / f'rank{rank}.pt');entry.update(host=host, rank=rank, section='export');entries.append(entry)
        del export, sharded
        if leader:
            replica = {k: v.detach().cpu().contiguous().clone() for k, v in state.items() if not hasattr(v, 'to_local')}
            full = {'model': replica, 'optimizers': _optimizer_state(model, optimizers, sharded=False)}
            entry = _save_file(full, tmp / 'resume' / 'rep.pt');entry.update(host=host, section='resume');entries.append(entry)
            del full
            export = {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in replica.items()}
            entry = _save_file(export, tmp / 'export' / 'rep.pt');entry.update(host=host, section='export');entries.append(entry)
            del export, replica
    except Exception as exc:
        error = f'{host} rank{rank}: {type(exc).__name__}: {exc}'
    errors = [e for e in _all_objects(error) if e]
    if errors:
        raise RuntimeError('Checkpoint incomplete; no COMPLETE marker written: ' + '; '.join(errors))
    files = sum(_all_objects(entries), [])
    gain = None
    if 'vgain.theta' in state:
        theta = state['vgain.theta']
        gain = float(theta.full_tensor().reshape(-1)[0]) if hasattr(theta, 'full_tensor') else float(theta.reshape(-1)[0])
    if leader:
        manifest = {'format': 'pose-duplex-distill-resume-v1', 'storage_layout': 'node-local-shards',
                    'step': int(step), 'world': world, 'hosts': hosts, 'host': host,
                    'run_dir': str(Path(run_dir).resolve()),
                    'local_ranks': [r for r, h in enumerate(hosts) if h == host],
                    'cfg': cfg, 'runtime_state': cfg.get('_runtime_state', {}),
                    'files': files, 'created_unix': time.time()}
        _atomic_json(tmp / 'resume' / 'train_state.json', manifest)
        export_meta = {'format': 'ppm-local-bf16-shards-v1', 'storage_layout': 'node-local-shards',
                       'step': int(step), 'world': world, 'hosts': hosts, 'host': host,
                       'run_dir': str(Path(run_dir).resolve()),
                       'run_tag': cfg.get('run_tag'), 'cfg': cfg, 'files': [f for f in files if f['section'] == 'export'],
                       'pose': int(any(k.startswith('tower_p.') for k in state)),
                       'decp_bb': int(any(k.startswith('decp_bb.') for k in state)),
                       'decp_bb_alpha': cfg.get('decp_bb_alpha', 1.0),
                       **({'vgain_theta_fp32': gain} if gain is not None else {})}
        _atomic_json(tmp / 'export' / 'meta.json', export_meta)
        (tmp / 'resume' / 'COMPLETE').write_text(str(step) + '\n')
        _atomic_json(tmp / 'export' / 'NODE_COMPLETE', {'step': int(step), 'host': host, 'world': world,
                      'ranks': manifest['local_ranks'], 'note': 'Other ranks live on the hosts in meta.json. Gather them before changing world size.'})
    _barrier()
    if leader:
        os.rename(tmp, final)
    _barrier()
    retention = _commit_latest_resume(run_dir, final, rank, world, hosts, bool(cfg.get('keep_latest_optimizer_state', False)))
    return {'path': str(final), 'resume_path': str(final / 'resume'), 'export_path': str(final / 'export'),
            'step': int(step), 'world': world, 'hosts': hosts, 'files': files, 'retention': retention}


@torch.no_grad()
def resume_checkpoint(model, optimizers, path, dev, *, restore_rng=True):
    """Restore a save_checkpoint resume directory, including optimizer and RNG state."""
    path = Path(path)
    rank, world, hosts = _rank_world_hosts()
    if not (path / 'COMPLETE').is_file():
        raise ValueError(f'Incomplete resume checkpoint: {path}')
    meta = json.loads((path / 'train_state.json').read_text())
    if meta.get('format') != 'pose-duplex-distill-resume-v1' or meta['world'] != world or meta['hosts'] != hosts:
        raise ValueError('Resume requires the saved format, world size and exact rank-to-host mapping')
    for filename in (f'rank{rank}.pt', 'rep.pt'):
        matches = [f for f in meta['files'] if f['section'] == 'resume' and f['host'] == hosts[rank] and f['file'] == filename]
        if len(matches) != 1 or _sha256(path / filename) != matches[0]['sha256']:
            raise ValueError(f'Resume file hash mismatch: {filename}')
    shard = torch.load(path / f'rank{rank}.pt', map_location='cpu', mmap=True, weights_only=True)
    replica = torch.load(path / 'rep.pt', map_location='cpu', mmap=True, weights_only=True)
    saved_model = {**shard['model'], **replica['model']}
    state = _state(model)
    _check_keys(state, saved_model, False)
    for name, tensor in state.items():
        local = _local(tensor)
        saved = saved_model[name]
        if local.shape != saved.shape or local.dtype != saved.dtype:
            raise ValueError(f'Resume shape/dtype mismatch for {name}')
        local.copy_(saved.to(device=local.device))
    parameters = {_canonical(n): p for n, p in model.named_parameters()}
    if set(optimizers) != set(shard['optimizers']) or set(optimizers) != set(replica['optimizers']):
        raise ValueError('Resume optimizer names differ')
    for name, optimizer in optimizers.items():
        saved_groups = shard['optimizers'][name]['param_groups']
        if len(saved_groups) != len(optimizer.param_groups):
            raise ValueError(f'Resume optimizer group count differs: {name}')
        for live, saved in zip(optimizer.param_groups, saved_groups):
            if [parameters[n] is p for n, p in zip(saved['params'], live['params'])] != [True] * len(live['params']) or len(saved['params']) != len(live['params']):
                raise ValueError(f'Resume optimizer parameter ordering differs: {name}')
            live.update({k: v for k, v in saved.items() if k != 'params'})
        optimizer.state.clear()
        all_states = {**shard['optimizers'][name]['state'], **replica['optimizers'][name]['state']}
        for param_name, saved in all_states.items():
            p = parameters[param_name]
            optimizer.state[p] = _unpack_value(saved, p)
    if restore_rng:
        _restore_rng(shard['rng'])
    _finite_check(state, dev)
    return meta
