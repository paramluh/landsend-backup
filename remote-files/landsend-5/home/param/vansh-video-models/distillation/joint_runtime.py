"""Small runtime controls shared by the joint trainer and its CPU checks."""
import json
import hashlib
import os
from pathlib import Path

from distillation.checkpoint_retention import checkpoint_compatibility, enforce_checkpoint_retention


def validate_reference_bank(directory):
    """Verify a finalized bank once on rank zero and return its resume identity.

    Source completeness is separate from artifact validity: the recipe permits
    quality-empty slots, while unavailable source candidates remain reported.
    All available work and every published file must nevertheless be verified.
    """
    directory = Path(directory).resolve()

    def digest(path):
        with path.open('rb') as handle:
            return hashlib.file_digest(handle, 'sha256').hexdigest()

    marker_path = directory / 'artifact-validation.json'
    marker_bytes = marker_path.read_bytes()
    marker = json.loads(marker_bytes)
    if (marker.get('schema_version') != 1 or marker.get('artifact_valid') is not True
            or marker.get('available_work_complete') is not True):
        raise ValueError('Reference bank has not completed artifact validation')
    for name, key in [('coverage.json', 'coverage_sha256'), ('SHA256SUMS', 'manifest_sha256')]:
        if digest(directory / name) != marker.get(key):
            raise ValueError(f'Reference bank validation no longer matches {name}')
    report = json.loads((directory / 'coverage.json').read_text())
    if (report.get('sources', {}).get('errors') != [] or report.get('latents', {}).get('errors') != []
            or any(report.get(key) is not marker[key] for key in
                   ('artifact_valid', 'available_work_complete', 'source_complete'))
            or report.get('selection_sha256') != marker.get('selection_sha256')
            or report.get('vae_sha256') != marker.get('vae_sha256')
            or report.get('encoder_fingerprint', {}).get('sha256') != marker.get('encoder_fingerprint_sha256')
            or report.get('complete') is not marker.get('source_complete')):
        raise ValueError('Reference bank coverage and final validation disagree')
    files = set()
    for line in (directory / 'SHA256SUMS').read_text().splitlines():
        expected, relative = line.split('  ', 1)
        path = directory / relative
        if (Path(relative).is_absolute() or '..' in Path(relative).parts
                or not path.resolve().is_relative_to(directory) or relative in files):
            raise ValueError(f'Invalid reference bank manifest path: {relative}')
        if digest(path) != expected:
            raise ValueError(f'Reference bank checksum differs: {relative}')
        files.add(relative)
    required = {'coverage.json', 'selection.parquet', 'bank/bank.csv', 'bank/SHA256SUMS'}
    if not required <= files or digest(directory / 'selection.parquet') != marker['selection_sha256']:
        raise ValueError('Reference bank manifest lacks its required selection/coverage files')
    actual_latents = {str(path.relative_to(directory)) for path in (directory / 'bank').rglob('*.safetensors')}
    recorded_latents = {name for name in files if name.startswith('bank/') and name.endswith('.safetensors')}
    if actual_latents != recorded_latents or not actual_latents:
        raise ValueError('Reference bank latent files differ from its finalized manifest')
    if marker_path.read_bytes() != marker_bytes:
        raise ValueError('Reference bank validation changed during verification')
    return {'validation_sha256': hashlib.sha256(marker_bytes).hexdigest(),
            'manifest_sha256': marker['manifest_sha256'], 'selection_sha256': marker['selection_sha256'],
            'encoder_fingerprint_sha256': marker['encoder_fingerprint_sha256'],
            'source_complete': marker['source_complete'], 'verified_files': len(files)}


def sync_checkpoint_files(directory):
    """Flush saved model and rank shards before publishing a completion marker."""
    for path in Path(directory).iterdir():
        if path.is_file():
            with path.open('rb') as handle:
                os.fsync(handle.fileno())


def sync_directory(directory):
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def claim_save_request(directory):
    """Claim touch requests without losing a new request arriving during a save.

    A failed save leaves the pending marker for an exact resume to retry.
    """
    directory = Path(directory)
    claimed = []
    quitting = False
    for name in ('save', 'save_q'):
        source, pending = directory / name, directory / f'.{name}.pending'
        if source.exists() and not pending.exists():
            source.rename(pending)
        if pending.exists():
            claimed.append(pending.name)
            quitting |= name == 'save_q'
    return {'save': bool(claimed), 'quit': quitting, 'markers': claimed}


def acknowledge_save_request(directory, request):
    for name in request['markers']:
        if name not in ('.save.pending', '.save_q.pending'):
            raise ValueError(f'Invalid save request marker: {name}')
        (Path(directory) / name).unlink(missing_ok=True)


def prune_previous_checkpoints(directory):
    """Keep optimizer shards only in the newly completed checkpoint.

    Call after promoting the durable checkpoint, before optional evaluation.
    Partial saves are never pruned and never authorize pruning another save.
    """
    from distillation.checkpoints import prune_optimizer_state, read_checkpoint_metadata
    directory = Path(directory).resolve()
    newest = read_checkpoint_metadata(directory)
    if not newest.get('optimizer_state_retained'):
        raise ValueError('The newest checkpoint must retain its optimizer state')
    candidates = []
    for older in sorted(directory.parent.glob('step-*')):
        if older == directory or not (older / 'checkpoint.complete.json').exists():
            continue
        metadata = json.loads((older / 'metadata.json').read_text())
        if metadata.get('counters', {}).get('rounds', -1) > newest['counters']['rounds']:
            raise ValueError('Refusing to prune optimizer state from a newer checkpoint')
        if metadata.get('optimizer_state_retained'):
            candidates.append((older, metadata['checkpoint_id']))
    return [prune_optimizer_state(older.resolve(), checkpoint_id, verify_weight_hashes=False)
            for older, checkpoint_id in candidates]


def planned_student_steps(epoch_steps, max_student_steps=None):
    """Cap work within the existing epoch without altering its data identity."""
    if epoch_steps < 1:
        raise ValueError('The epoch must contain at least one optimizer round')
    if max_student_steps is not None and (type(max_student_steps) is not int or max_student_steps < 1):
        raise ValueError('max_student_steps must be a positive integer when supplied')
    return min(epoch_steps, max_student_steps) if max_student_steps is not None else epoch_steps


def full_gt_checkpoint_due(counters, interval, student_steps, *, warming):
    """Periodic full-GT saves count student updates, excluding critic headstart."""
    step = counters['student_updates']
    return not warming and step > 0 and (step % interval == 0 or step == student_steps)


def full_gt_evaluation_due(counters, interval, *, warming):
    """Independent probe cadence; checkpoint/manual saves take priority."""
    step = counters['student_updates']
    return not warming and step > 0 and step % interval == 0


def training_indices(epoch_steps, counters, warmup, benchmark_steps=0, *, max_student_steps=None):
    student_steps = planned_student_steps(epoch_steps, max_student_steps)
    if benchmark_steps:
        # Every benchmark iteration exercises the complete joint objective.
        return [], [index % epoch_steps for index in range(benchmark_steps)]
    if not 0 <= counters['epoch_step'] <= student_steps:
        raise ValueError('Resume student cursor lies outside the planned horizon')
    return ([index % epoch_steps for index in range(counters['warmup_step'], warmup)],
            range(counters['epoch_step'], student_steps))
