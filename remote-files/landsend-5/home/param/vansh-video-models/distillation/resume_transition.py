"""Explicit, run-scoped authorization for a full-state guidance-only resume."""
import copy
import fcntl
import hashlib
import json
from pathlib import Path

from distillation.checkpoint_retention import checkpoint_compatibility


TRANSITIONS_FILE = 'resume-transitions.json'
FULL_GT_SCHEMA = 'full-gt-nine-ref-v1-single-target'
GUIDANCE_OPT_IN = 'resume_allow_teacher_guidance_change'


def validate_resume_compatibility(saved, requested, *, allow_teacher_guidance_change=False):
    """Keep exact resume checks except an explicitly enabled full-GT 1 -> 2.5.

    This validates training identity only. It never changes checkpoint data or
    selects a weights-only restore path; normal optimizer/EMA/RNG restoration
    and counter validation remain mandatory in the caller.
    """
    if type(allow_teacher_guidance_change) is not bool:
        raise ValueError('resume_allow_teacher_guidance_change must be a boolean')
    saved, requested = checkpoint_compatibility(saved), checkpoint_compatibility(requested)
    if saved == requested:
        return None
    if allow_teacher_guidance_change and saved.get('schema') == requested.get('schema') == FULL_GT_SCHEMA:
        source_training, target_training = saved.get('training', {}), requested.get('training', {})
        if (type(source_training.get('teacher_guidance_scale')) in (int, float)
                and source_training.get('teacher_guidance_scale') == 1.0
                and target_training.get('teacher_guidance_scale') == 2.5):
            changed = copy.deepcopy(saved)
            changed['training']['teacher_guidance_scale'] = 2.5
            if changed == requested:
                return {'field': 'training.teacher_guidance_scale', 'from': 1.0, 'to': 2.5}
    raise ValueError('Resume checkpoint data, model, objective, topology or schedule differs')


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _read_ledger(root, plan_signature):
    """Read and validate the complete enrollment chain without mutating it."""
    plan = root / 'epoch-plan.json'
    ledger_path = root / TRANSITIONS_FILE
    if plan.is_symlink() or ledger_path.is_symlink():
        raise ValueError('Resume transition control paths must not be symlinks')
    signatures = [checkpoint_compatibility(plan_signature)]
    if not ledger_path.exists():
        return signatures, None
    if not plan.is_file():
        raise ValueError('Resume transitions require the original epoch plan')
    ledger = json.loads(ledger_path.read_text())
    if (ledger.get('schema') != 'full-gt-resume-transitions-v1'
            or ledger.get('run_directory') != str(root)
            or ledger.get('epoch_plan_sha256') != hashlib.sha256(plan.read_bytes()).hexdigest()):
        raise ValueError('Resume transition run identity differs')
    previous = None
    for entry in ledger['transitions']:
        payload = {key: value for key, value in entry.items() if key != 'transition_id'}
        if (entry.get('transition_id') != _digest(payload)
                or entry.get('previous_transition_id') != previous):
            raise ValueError('Resume transition receipt hash or chain differs')
        source = checkpoint_compatibility(entry['source_signature'])
        target = checkpoint_compatibility(entry['target_signature'])
        checkpoint = Path(entry['source_checkpoint'])
        if (source != signatures[-1] or target in signatures
                or checkpoint.parent != root or not checkpoint.name.startswith('step-')
                or not entry.get('source_checkpoint_id')
                or type(entry.get('source_counters', {}).get('rounds')) is not int
                or entry['source_counters']['rounds'] < 0
                or entry.get('authorization') != GUIDANCE_OPT_IN):
            raise ValueError('Resume transition source identity differs')
        for name in ('source_metadata_sha256', 'source_manifest_sha256'):
            digest = entry.get(name, '')
            if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError('Resume transition checkpoint fingerprint differs')
        change = validate_resume_compatibility(source, target, allow_teacher_guidance_change=True)
        if change is None or entry.get('change') != change:
            raise ValueError('Resume transition must contain the authorized guidance change')
        signatures.append(target)
        previous = entry['transition_id']
    return signatures, ledger


def accepted_run_signatures(root, plan_signature):
    """Only the immutable plan and its explicit validated successors belong."""
    return _read_ledger(Path(root), plan_signature)[0]


def record_resume_transition(directory, requested_signature, *, expected_checkpoint_id,
                             allow_teacher_guidance_change=False):
    """Enroll a validated guidance change after the caller restores full state.

    The immutable epoch plan and old checkpoint are never rewritten. The
    receipt survives normal retention, including pruning its source checkpoint.
    Repeating the same resume is idempotent; changing its source is not.
    """
    from distillation.checkpoints import read_checkpoint_metadata, write_json_atomic
    from distillation.joint_runtime import sync_directory

    original = Path(directory).absolute()
    directory = original.resolve()
    if original != directory or not directory.name.startswith('step-'):
        raise ValueError('Resume transition requires a real direct checkpoint path')
    root = directory.parent
    for name in ('.checkpoint-retention.lock', 'latest.json', 'epoch-plan.json', TRANSITIONS_FILE):
        if (root / name).is_symlink():
            raise ValueError('Resume transition control paths must not be symlinks')
    with (root / '.checkpoint-retention.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        metadata = read_checkpoint_metadata(directory)
        if metadata['checkpoint_id'] != expected_checkpoint_id:
            raise ValueError('Resume checkpoint generation changed before transition enrollment')
        source = checkpoint_compatibility(metadata['compatibility'])
        target = checkpoint_compatibility(requested_signature)
        change = validate_resume_compatibility(source, target,
            allow_teacher_guidance_change=allow_teacher_guidance_change)
        if change is None:
            return None
        plan_path = root / 'epoch-plan.json'
        plan_bytes = plan_path.read_bytes()
        plan_signature = json.loads(plan_bytes)['signature']
        signatures, ledger = _read_ledger(root, plan_signature)
        latest_bytes = (root / 'latest.json').read_bytes()
        latest = json.loads(latest_bytes)
        if (Path(latest['checkpoint']).absolute() != directory
                or latest['counters'] != metadata['counters']):
            raise ValueError('Only the latest full-state checkpoint can authorize a guidance transition')
        if ledger is not None:
            for entry in ledger['transitions']:
                if (entry['source_signature'] == source and entry['target_signature'] == target
                        and entry['source_checkpoint'] == str(directory)
                        and entry['source_checkpoint_id'] == expected_checkpoint_id):
                    return entry
        if source != signatures[-1] or target in signatures:
            raise ValueError('Resume transition is not the next enrolled run identity')
        if ledger is None:
            ledger = {'schema': 'full-gt-resume-transitions-v1', 'run_directory': str(root),
                      'epoch_plan_sha256': hashlib.sha256(plan_bytes).hexdigest(), 'transitions': []}
        entry = {'previous_transition_id': ledger['transitions'][-1]['transition_id']
                 if ledger['transitions'] else None,
                 'authorization': GUIDANCE_OPT_IN, 'change': change,
                 'source_signature': source, 'target_signature': target,
                 'source_checkpoint': str(directory), 'source_checkpoint_id': expected_checkpoint_id,
                 'source_counters': metadata['counters'],
                 'source_metadata_sha256': hashlib.sha256((directory / 'metadata.json').read_bytes()).hexdigest(),
                 'source_manifest_sha256': hashlib.sha256((directory / 'checkpoint.complete.json').read_bytes()).hexdigest()}
        entry['transition_id'] = _digest(entry)
        ledger['transitions'].append(entry)
        if plan_path.read_bytes() != plan_bytes:
            raise ValueError('Original epoch plan changed during transition enrollment')
        if ((root / 'latest.json').read_bytes() != latest_bytes
                or read_checkpoint_metadata(directory)['checkpoint_id'] != expected_checkpoint_id):
            raise ValueError('Latest checkpoint changed during transition enrollment')
        write_json_atomic(root / TRANSITIONS_FILE, ledger)
        sync_directory(root)
        return entry
