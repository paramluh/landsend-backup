"""Compact per-site sensitivity views and verified, JSON-only run-file uploads."""
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
import time


GROUPS = ('sensitivity_history', 'sensitivity_sink', 'sensitivity_reference', 'sensitivity_overview')
MAX_REPORT_BYTES = 8 * 1024 * 1024


def _json(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_REPORT_BYTES:
        raise ValueError('Sensitivity upload requires a bounded, real JSON file')
    data = path.read_bytes()
    return json.loads(data), data


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def sensitivity_metrics(report, weights):
    """Two scientific scalars per configured site; never mix distinct sites.

    Max is max(per-seed maximum amplification); last-frame amplification is
    mean(per-seed last-frame amplification), excluding explicitly skipped seeds.
    One invalid measured seed suppresses BOTH scalars for that site. The raw
    report retains every seed/frame/row and each original weighted measurement.
    """
    if weights not in ('raw', 'ema') or report.get('format') != 'joint-latent-sensitivity-v1':
        raise ValueError('Unsupported sensitivity report')
    status = report.get('status')
    if status not in ('ok', 'nonfinite_output', 'skipped_no_valid_rows'):
        raise ValueError('Unknown sensitivity report status')
    base = f'sensitivity_overview/{weights}'
    rollouts = report.get('rollouts', [])
    result = {f'{base}_status_ok': int(status == 'ok'), f'{base}_cases_failed': 0,
              f'{base}_cases_skipped': 0, f'{base}_seed_count': len(rollouts)}
    if status == 'skipped_no_valid_rows':
        if rollouts:
            raise ValueError('Skipped sensitivity report cannot contain rollouts')
        return result
    frames = report['measured_target_frames']
    magnitude = report['requested_input_rms']
    if type(frames) is not int or frames < 1 or not _finite(magnitude) or magnitude <= 0:
        raise ValueError('Invalid sensitivity horizon or perturbation magnitude')
    result.update({f'{base}_target_frames': frames, f'{base}_input_rms': magnitude})
    sites = {'sink': []}
    for kind, key, limit in (('history', 'history_indices_oldest_first', 3),
                             ('reference', 'reference_indices', 9)):
        indices = report[key]
        if len(indices) != len(set(indices)) or any(type(i) is not int or not 0 <= i < limit for i in indices):
            raise ValueError('Invalid sensitivity site indices')
        sites.update({f'{kind}_{index}': [] for index in indices})
    seen = set()
    for rollout in rollouts:
        seed = rollout['seed']
        if type(seed) is not int or seed in seen:
            raise ValueError('Sensitivity seeds must be distinct integers')
        seen.add(seed)
        if set(rollout['cases']) != set(sites):
            raise ValueError('Unknown or missing sensitivity case names')
        for name, case in rollout['cases'].items():
            case_status = case['status']
            if case_status == 'skipped_no_active_rows':
                if case.get('frames'):
                    raise ValueError('Skipped sensitivity case cannot contain frames')
                result[f'{base}_cases_skipped'] += 1
                continue
            if case_status not in ('ok', 'nonfinite_output'):
                raise ValueError('Unknown sensitivity case status')
            measured = case['frames']
            if [frame['target_index'] for frame in measured] != list(range(1, frames + 1)):
                raise ValueError('Sensitivity frame identities differ from configured horizon')
            maximum, last = case.get('maximum_amplification'), case.get('last_frame_amplification')
            valid = (case_status == 'ok' and _finite(maximum) and _finite(last)
                     and maximum >= 0 and last >= 0
                     and all(_finite(frame.get('amplification')) and frame['amplification'] >= 0
                             and frame.get('nonfinite_rows', 0) == 0 for frame in measured))
            if valid and (not math.isclose(maximum, max(frame['amplification'] for frame in measured),
                                          rel_tol=1e-6, abs_tol=1e-10)
                          or not math.isclose(last, measured[-1]['amplification'], rel_tol=1e-6, abs_tol=1e-10)):
                raise ValueError('Sensitivity case aggregate differs from its per-frame measurements')
            sites[name].append((maximum, last) if valid else None)
            if not valid:
                result[f'{base}_cases_failed'] += 1
                result[f'{base}_status_ok'] = 0
    if not rollouts or report.get('seeds') != [row['seed'] for row in rollouts]:
        raise ValueError('Sensitivity rollouts differ from configured seeds')
    for name, measurements in sites.items():
        if not measurements or any(value is None for value in measurements):
            continue
        kind = name.split('_')[0]
        prefix = f'sensitivity_{kind}/{weights}_{name}'
        result[f'{prefix}_max_amplification'] = max(value[0] for value in measurements)
        result[f'{prefix}_last_frame_amplification'] = sum(value[1] for value in measurements) / len(measurements)
    return result


def _checkpoint_reports(directory):
    directory = Path(directory).absolute()
    metadata, _ = _json(directory / 'metadata.json')
    evaluation, _ = _json(directory / 'evaluation.complete.json')
    current, original = metadata['checkpoint_id'], evaluation['checkpoint_id']
    for identity in (current, original):
        if not isinstance(identity, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', identity):
            raise ValueError('Invalid sensitivity checkpoint identity')
    if current != original:
        # Pruning rewrites the checkpoint UUID but preserves the original
        # evaluation. Accept only the explicit committed prune lineage.
        try:
            receipt, _ = _json(directory.with_name(directory.name + '.optimizer-prune.json'))
            complete, _ = _json(directory / 'checkpoint.complete.json')
        except (ValueError, OSError) as error:
            raise ValueError('Sensitivity completion does not identify this checkpoint or valid prune lineage') from error
        weights = metadata.get('optimizer_pruned_weights')
        if not (metadata.get('optimizer_state_retained') is False
                and metadata.get('optimizer_pruned_from_checkpoint_id') == original
                and receipt.get('previous_checkpoint_id') == original
                and receipt.get('checkpoint_id') == current
                and Path(receipt.get('path', '')).absolute() == directory
                and receipt.get('weights_unchanged') is True
                and receipt.get('optimizer_state_retained') is False
                and weights and receipt.get('weights') == weights
                and complete.get('checkpoint_id') == current):
            raise ValueError('Sensitivity completion does not identify this checkpoint or valid prune lineage')
        manifest = {entry['path']: entry['size'] for entry in complete['files']}
        for name, proof in weights.items():
            if Path(name).name != name or not name.endswith('.safetensors'):
                raise ValueError('Invalid pruned checkpoint weight identity')
            if manifest.get(name) != proof['size'] or (directory / name).stat().st_size != proof['size']:
                raise ValueError('Pruned checkpoint weight size differs from committed lineage')
    entries, seen = [], set()
    for entry in evaluation.get('reports', []):
        label = entry['weights']
        if label not in ('raw', 'ema') or label in seen or entry['file'] != f'sensitivity-{label}.json':
            raise ValueError('Unexpected sensitivity report path or weight label')
        seen.add(label)
        report, data = _json(directory / entry['file'])
        compact = sensitivity_metrics(report, label)
        entries.append({'weights': label, 'filename': entry['file'], 'data': data, 'report': report,
                        'metrics': compact})
    if not entries:
        raise ValueError('Sensitivity completion does not identify any reports')
    return directory, metadata, evaluation, entries


def checkpoint_metrics(directory):
    _, metadata, _, entries = _checkpoint_reports(directory)
    result = {'sensitivity_overview/checkpoint_round': metadata['counters']['rounds'],
              'sensitivity_overview/checkpoint_student_updates': metadata['counters']['student_updates']}
    for entry in entries:
        result.update(entry['metrics'])
    return result


def stage_checkpoint_reports(directory, staging_root):
    """Copy only allowlisted small JSON, pinned to original evaluation UUID.

    Returns explicit file records usable with SDK run.save(policy='now',
    glob=False) or public Run.upload_file(path, root=staging_root). No glob,
    checkpoint directory, model weight, teacher, or optimizer is uploaded.
    """
    directory, metadata, evaluation, entries = _checkpoint_reports(directory)
    root = Path(staging_root).absolute()
    identity = evaluation['checkpoint_id']
    destination = root / 'sensitivity_reports' / identity
    destination.mkdir(parents=True, exist_ok=True)
    mapping = {'format': 'joint-sensitivity-overview-v1', 'checkpoint_id': identity,
               'checkpoint_round': metadata['counters']['rounds'],
               'scientific_units': 'stabilizer latent endpoint amplification; not pixels or resolver outputs',
               'max_amplification': 'maximum of per-seed maximum_amplification for this exact site',
               'last_frame_amplification': 'arithmetic mean of per-seed last_frame_amplification for this exact site',
               'seed_support': 'Explicit skipped cases excluded; any invalid measured seed suppresses both site scalars. Each per-frame source value already uses the original valid-row sample weights.',
               'health': 'cases_failed and cases_skipped count seed/site cases, not independent output frames; status_ok reflects report and case health.',
               'causal_distinction': 'History perturbations begin at generated-frame history commitment. Sink/reference perturbations begin at startup. Sites are never averaged together.',
               'reports': []}
    files = []
    for entry in entries:
        data = entry['data']
        mapping['reports'].append({'weights': entry['weights'], 'file': entry['filename'],
            'sha256': hashlib.sha256(data).hexdigest(), 'seeds': entry['report'].get('seeds', []),
            'history_indices_oldest_first': entry['report'].get('history_indices_oldest_first', []),
            'reference_indices': entry['report'].get('reference_indices', []),
            'compact_metrics': entry['metrics']})
        files.append((entry['filename'], data, entry['weights']))
    files.append(('sensitivity-overview-definition.json',
                  (json.dumps(mapping, indent=2, allow_nan=False) + '\n').encode(), 'definition'))
    records = []
    for name, data, label in files:
        target = destination / name
        if target.exists():
            if target.is_symlink() or target.read_bytes() != data:
                raise ValueError('Immutable sensitivity staging file differs from its source')
        else:
            with target.open('xb') as handle:
                handle.write(data)
        records.append({'path': str(target), 'name': str(target.relative_to(root)),
            'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(), 'weights': label,
            'checkpoint_id': identity, 'checkpoint_round': metadata['counters']['rounds']})
    return records


def verify_report_files(remote, records, *, api=None):
    """Verify remote bytes by SHA256; metadata visibility alone is insufficient.

    Returns None while uploads are not visible/complete. A visible file whose
    bytes differ fails closed. Downloads are bounded by expected small sizes.
    """
    files = {file.name: file for file in remote.files(names=[record['name'] for record in records])}
    if any(record['name'] not in files or files[record['name']].size != record['bytes'] for record in records):
        return None
    verified = []
    with tempfile.TemporaryDirectory(prefix='joint-sensitivity-verify-') as temporary:
        for record in records:
            file = files[record['name']]
            with file.download(root=temporary, replace=True, api=api) as handle:
                digest = hashlib.sha256(Path(handle.name).read_bytes()).hexdigest()
            if digest != record['sha256']:
                raise RuntimeError('Remote sensitivity JSON differs from the staged SHA256')
            verified.append({**record, 'remote_sha256_verified': True})
    return verified


def upload_checkpoint_reports(remote, directory, staging_root, *, api=None, verify_timeout=60):
    """Public-API historical backfill; never starts or changes an SDK run."""
    records = stage_checkpoint_reports(directory, staging_root)
    for record in records:
        remote.upload_file(record['path'], root=str(Path(staging_root).absolute()))
    deadline = time.monotonic() + verify_timeout
    while True:
        verified = verify_report_files(remote, records, api=api)
        if verified is not None:
            return verified
        if time.monotonic() >= deadline:
            raise RuntimeError('Sensitivity JSON uploads were not remotely verifiable before timeout')
        if api is not None:
            api.flush()
        time.sleep(2)
