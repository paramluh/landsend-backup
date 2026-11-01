"""Run-scoped retention for completed joint-training checkpoints."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import uuid


WEIGHTS = {'student.safetensors', 'critic.safetensors', 'model_ema.safetensors'}


def checkpoint_compatibility(value):
    """Use the same run identity for joint resume and checkpoint retention.

    Old joint manifests include runtime policy fields that newer checkpoints
    omit. Normalize both sides without rewriting either saved identity. Other
    trainers retain their existing semantics: only retention count is ignored.
    """
    value = dict(value)
    runtime_keys = {'checkpoint_keep_last'}
    if value.get('schema') in ('joint-nine-ref-v2-paired-batches', 'full-gt-nine-ref-v1-single-target'):
        runtime_keys.update({'resume_from', 'output_dir', 'num_dataloader_workers', 'prefetch_factor',
                            'wandb', 'compile_cache_size_limit', 'compile_accumulated_cache_size_limit',
                            'save_every_n_steps'})
    if value.get('schema') == 'full-gt-nine-ref-v1-single-target':
        # This is an explicit resume permission, not an objective parameter.
        # teacher_guidance_scale itself remains part of the strict identity.
        runtime_keys.update({'resume_allow_teacher_guidance_change', 'eval_every_n_steps'})
    if 'training' in value:
        value['training'] = {k: v for k, v in value['training'].items()
                             if k not in runtime_keys}
    return value


def _completed(path):
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f'Checkpoint must be a real directory: {path}')
    marker = json.loads((path / 'checkpoint.complete.json').read_text())
    meta = json.loads((path / 'metadata.json').read_text())
    if marker['checkpoint_id'] != meta['checkpoint_id']:
        raise ValueError(f'Checkpoint generation differs: {path}')
    names = set()
    for item in marker['files']:
        name = item['path']
        file = path / name
        if (Path(name).name != name or name in names or file.is_symlink()
                or not file.is_file() or file.stat().st_size != item['size']):
            raise ValueError(f'Invalid completed checkpoint file: {file}')
        names.add(name)
    if not WEIGHTS <= names or 'metadata.json' not in names:
        raise ValueError(f'Completed checkpoint lacks raw/critic/EMA weights: {path}')
    # These checkpoints are flat. Never recursively delete unexpected data.
    if any(p.is_symlink() or not p.is_file() for p in path.iterdir()):
        raise ValueError(f'Unexpected nested or symlinked checkpoint content: {path}')
    return meta


def _finish_quarantine(root, receipt, record, write_json_atomic, sync_directory):
    token = record['retirement_id']
    if (not re.fullmatch('[0-9a-f]{32}', token) or receipt.name != f'{token}.json'
            or record['quarantine'] != f'.retired-checkpoint-{token}'
            or not record['checkpoint_name'].startswith('step-')
            or Path(record['checkpoint_name']).name != record['checkpoint_name']):
        raise ValueError('Retirement receipt identity differs')
    trash = root / record['quarantine']
    if trash.parent != root or not trash.name.startswith('.retired-checkpoint-'):
        raise ValueError('Invalid retention quarantine path')
    if trash.exists():
        if trash.is_symlink() or not trash.is_dir():
            raise ValueError('Retention quarantine must be a real directory')
        anchor = trash / '.retirement-identity.json'
        identity = {k: record[k] for k in ('retirement_id', 'checkpoint_id',
                    'metadata_sha256', 'manifest_sha256')}
        if anchor.exists():
            if anchor.is_symlink() or json.loads(anchor.read_text()) != identity:
                raise ValueError('Quarantine generation anchor differs')
        elif any(trash.iterdir()):
            # Before cleanup starts both original identity files must survive.
            for name, key in [('metadata.json', 'metadata_sha256'),
                              ('checkpoint.complete.json', 'manifest_sha256')]:
                file = trash / name
                if (file.is_symlink() or not file.is_file()
                        or hashlib.sha256(file.read_bytes()).hexdigest() != identity[key]):
                    raise ValueError('Quarantine checkpoint generation differs')
            write_json_atomic(anchor, identity)
            sync_directory(trash)
        # The anchor permits recovery after one identity file was deleted, but
        # must not hide replacement of an identity file that still survives.
        for name, key in [('metadata.json', 'metadata_sha256'),
                          ('checkpoint.complete.json', 'manifest_sha256')]:
            file = trash / name
            if file.exists() and (file.is_symlink() or not file.is_file()
                    or hashlib.sha256(file.read_bytes()).hexdigest() != identity[key]):
                raise ValueError('Quarantine checkpoint generation differs')
        allowed = record['files']
        for file in trash.iterdir():
            if file == anchor:
                continue
            if (file.name not in allowed or file.is_symlink() or not file.is_file()
                    or file.stat().st_size != allowed[file.name]):
                raise ValueError(f'Quarantined checkpoint changed: {file}')
        # The durable receipt lists every file and generation before rename.
        # A crash midway can safely retry the remaining subset of these files.
        for file in trash.iterdir():
            if file != anchor:
                file.unlink()
        sync_directory(trash)
        anchor.unlink(missing_ok=True)
        trash.rmdir()
        sync_directory(root)
    record['status'] = 'deleted'
    write_json_atomic(receipt, record)
    return record


def enforce_checkpoint_retention(directory, keep_last=8):
    """Keep N completed weight snapshots and only the latest optimizer state.

    The latest.json target must be the supplied, complete resumable checkpoint.
    Incomplete saves and foreign checkpoint identities never authorize deletion.
    A run-local policy persists across saves/resumes; deletion is recoverable
    through durable receipts and an atomic quarantine rename.
    """
    from distillation.checkpoints import prune_optimizer_state, read_checkpoint_metadata, write_json_atomic
    from distillation.joint_runtime import sync_directory
    from distillation.resume_transition import TRANSITIONS_FILE, accepted_run_signatures

    if type(keep_last) is not int or keep_last < 1:
        raise ValueError('checkpoint_keep_last must be a positive integer')
    original = Path(directory).absolute()
    directory = original.resolve()
    if original != directory or not directory.name.startswith('step-'):
        raise ValueError('Retention requires a real direct checkpoint path')
    root = directory.parent
    for name in ('.checkpoint-retention.lock', 'latest.json', 'epoch-plan.json',
                 'checkpoint-retention.json', 'checkpoint-retention-history', TRANSITIONS_FILE):
        if (root / name).is_symlink():
            raise ValueError(f'Retention control path must not be a symlink: {name}')
    with (root / '.checkpoint-retention.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        latest_bytes = (root / 'latest.json').read_bytes()
        latest = json.loads(latest_bytes)
        if Path(latest['checkpoint']).absolute() != directory:
            raise ValueError('Refusing retention from a checkpoint other than latest.json')
        newest = _completed(directory)
        read_checkpoint_metadata(directory)  # Requires every optimizer shard.
        if not newest['optimizer_state_retained'] or latest['counters'] != newest['counters']:
            raise ValueError('Latest complete resumable checkpoint differs from latest.json')
        signature = checkpoint_compatibility(newest['compatibility'])
        plan = root / 'epoch-plan.json'
        plan_bytes = plan.read_bytes() if plan.exists() else None
        plan_hash = hashlib.sha256(plan_bytes).hexdigest() if plan_bytes is not None else None
        transitions = root / TRANSITIONS_FILE
        transition_bytes = transitions.read_bytes() if transitions.exists() else None
        signatures = accepted_run_signatures(root, json.loads(plan_bytes)['signature']) if plan_bytes else [signature]
        if transition_bytes is not None and plan_bytes is None:
            raise ValueError('Resume transitions require the original epoch plan')
        if signature not in signatures:
            raise ValueError('Checkpoint identity does not belong to this run')
        policy_path = root / 'checkpoint-retention.json'
        policy = {'schema': 'joint-checkpoint-retention-v1', 'run_directory': str(root),
                  'epoch_plan_sha256': plan_hash, 'keep_last': keep_last,
                  'latest_optimizer_only': True, 'weight_files': sorted(WEIGHTS)}
        if policy_path.exists():
            previous = json.loads(policy_path.read_text())
            if any(previous.get(k) != policy[k] for k in policy if k != 'keep_last'):
                raise ValueError('Run retention policy identity differs')
            keep_last = previous['keep_last']
            if type(keep_last) is not int or keep_last < 1:
                raise ValueError('Invalid persisted checkpoint retention count')
            policy = previous

        candidates, skipped = [], []
        for path in sorted(root.glob('step-*')):
            if path.is_symlink() or not (path / 'checkpoint.complete.json').is_file():
                skipped.append({'path': str(path), 'reason': 'symlink_or_incomplete'})
                continue
            meta = _completed(path)
            if checkpoint_compatibility(meta.get('compatibility', {})) not in signatures:
                skipped.append({'path': str(path), 'reason': 'foreign_identity'})
                continue
            if meta['counters']['rounds'] > newest['counters']['rounds']:
                raise ValueError('Refusing to prune with a newer completed checkpoint present')
            candidates.append((path, meta))
        candidates.sort(key=lambda x: (x[0] == directory, x[1]['counters']['rounds'], x[0].name), reverse=True)
        kept = {p for p, _ in candidates[:keep_last]}
        assert directory in kept

        def unchanged():
            if (root / 'latest.json').read_bytes() != latest_bytes:
                raise ValueError('Latest checkpoint changed during retention')
            if ((plan.read_bytes() if plan.exists() else None) != plan_bytes
                    or (transitions.read_bytes() if transitions.exists() else None) != transition_bytes):
                raise ValueError('Run identity or resume transitions changed during retention')
            if _completed(directory)['checkpoint_id'] != newest['checkpoint_id']:
                raise ValueError('Protected checkpoint changed during retention')

        unchanged()
        write_json_atomic(policy_path, policy)
        history = root / 'checkpoint-retention-history'
        history.mkdir(exist_ok=True)
        optimizer_prunes, checkpoint_prunes = [], []
        for receipt in sorted(history.glob('*.json')):
            if receipt.is_symlink():
                raise ValueError('Retention receipt must not be a symlink')
            record = json.loads(receipt.read_text())
            if record.get('status') in ('deleted', 'cancelled_before_rename'):
                continue
            if (record.get('schema') != 'joint-checkpoint-retirement-v1'
                    or record.get('run_directory') != str(root)
                    or Path(record['checkpoint_name']).name != record['checkpoint_name']
                    or record['checkpoint_name'] == directory.name):
                raise ValueError('Foreign retention receipt')
            unchanged()
            trash = root / record['quarantine']
            if trash.exists():
                checkpoint_prunes.append(_finish_quarantine(root, receipt, record, write_json_atomic, sync_directory))
            elif not (root / record['checkpoint_name']).exists():
                checkpoint_prunes.append(_finish_quarantine(root, receipt, record, write_json_atomic, sync_directory))
            else:
                # The previous process stopped before renaming. Re-plan under
                # today's policy rather than replaying an outdated deletion.
                record['status'] = 'cancelled_before_rename'
                write_json_atomic(receipt, record)
        for path, meta in candidates:
            if path == directory:
                continue
            unchanged()
            if path in kept:
                if meta['optimizer_state_retained']:
                    optimizer_prunes.append(prune_optimizer_state(path, meta['checkpoint_id'], verify_weight_hashes=False))
                continue
            if _completed(path)['checkpoint_id'] != meta['checkpoint_id']:
                raise ValueError('Checkpoint changed before retirement')
            token = uuid.uuid4().hex
            record = {'schema': 'joint-checkpoint-retirement-v1', 'run_directory': str(root),
                      'retirement_id': token,
                      'checkpoint_name': path.name, 'checkpoint_id': meta['checkpoint_id'],
                      'rounds': meta['counters']['rounds'], 'protected_latest': str(directory),
                      'quarantine': f'.retired-checkpoint-{token}', 'status': 'planned',
                      'metadata_sha256': hashlib.sha256((path / 'metadata.json').read_bytes()).hexdigest(),
                      'manifest_sha256': hashlib.sha256((path / 'checkpoint.complete.json').read_bytes()).hexdigest(),
                      'files': {p.name: p.stat().st_size for p in path.iterdir()}}
            receipt = history / f'{token}.json'
            write_json_atomic(receipt, record)
            sync_directory(history)
            path.rename(root / record['quarantine'])
            sync_directory(root)
            checkpoint_prunes.append(_finish_quarantine(root, receipt, record, write_json_atomic, sync_directory))
        result = {'keep_last': keep_last, 'protected_latest': str(directory),
                  'kept': [str(p) for p, _ in candidates if p in kept], 'skipped': skipped,
                  'optimizer_prunes': optimizer_prunes, 'checkpoint_prunes': checkpoint_prunes}
        write_json_atomic(root / 'checkpoint-retention-last.json', result)
        sync_directory(root)
        return result
