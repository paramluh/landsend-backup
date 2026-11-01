"""Read-only post-resume verification of joint training scalars and report files.

Writes a proof only after at least five new updates are visible remotely and
every observed online row matches its actual local attempt log. Exit 2 means
the run has not yet reached the requested observation boundary; retry later.
This tool never initializes an SDK run, uploads data, or controls a job.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from distillation.joint_metrics import (TRAINING_DERIVED_FIELDS, TRAINING_METRIC_MAP,
    project_training_metrics)
from distillation.joint_sensitivity_tracking import MAX_REPORT_BYTES, checkpoint_metrics


class NotReady(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def parsed_log(path):
    records = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            records.append((line_number, value))
    return records


def compare_scalars(expected, actual, label):
    missing = sorted(set(expected) - set(actual))
    differences = []
    for key, value in expected.items():
        observed = actual.get(key)
        if key not in actual:
            continue
        if not (isinstance(observed, (int, float)) and math.isfinite(observed)
                and math.isclose(observed, value, rel_tol=1e-6, abs_tol=1e-10)):
            differences.append(key)
    if missing or differences:
        raise ValueError(f'{label}: missing={missing}, different={differences}')


def verified_startup_gate(parsed, checkpoint, counters, checkpoint_id, receipt):
    """Require the actual final online gate before any resumed training row.

    A durable checkpoint can lack its evaluation after a post-save failure.
    Its initial tracking gate then verifies training counters only; the later
    recovery event must bind the completed evaluation to this exact checkpoint.
    """
    initial = [(line, row) for line, row in parsed if row.get('event') == 'wandb_online_verified']
    recovered = [(line, row) for line, row in parsed
                 if row.get('event') == 'checkpoint_evaluation_recovered']
    metrics = [(line, row) for line, row in parsed if 'rounds' in row]
    if not initial:
        if metrics:
            raise ValueError('A training update preceded the remote startup gate')
        raise NotReady('The online startup verification has not yet appeared')
    if len(initial) != 1 or len(recovered) > 1:
        raise ValueError('Expected one initial online gate and at most one recovered evaluation')
    initial_line, initial_gate = initial[0]
    if (not initial_gate.get('online_metrics_verified')
            or initial_gate.get('verified_round') != counters['rounds']
            or initial_gate.get('verified_student_updates') != counters['student_updates']):
        raise ValueError('Initial online verification does not match restored training state')
    final_line, gate = initial_line, initial_gate
    if recovered:
        final_line, recovery = recovered[0]
        if (final_line <= initial_line or recovery.get('path') != str(checkpoint)
                or recovery.get('counters') != counters or recovery.get('checkpoint_id') != checkpoint_id):
            raise ValueError('Recovered evaluation does not identify this exact restored checkpoint')
        gate = recovery.get('tracking', {})
    if any(line <= final_line for line, _ in metrics):
        raise ValueError('A training update preceded the final remote startup gate')
    if (gate.get('verified_sensitivity_metrics') != 22
            or not gate.get('sensitivity_report_uploads_verified')
            or not gate.get('sensitivity_report_files')):
        if metrics or recovered:
            raise ValueError('Training/recovery event lacks verified sensitivity scalars and report uploads')
        raise NotReady('Checkpoint evaluation recovery has not yet completed online verification')
    required = ('publish_token', 'online_metrics_verified', 'verified_round', 'verified_student_updates',
                'verified_sensitivity_metrics', 'sensitivity_report_uploads_verified', 'sensitivity_report_files')
    if (not gate.get('publish_token') or not gate.get('online_metrics_verified')
            or gate.get('verified_round') != counters['rounds']
            or gate.get('verified_student_updates') != counters['student_updates']
            or any(gate.get(key) != receipt.get(key) for key in required)):
        raise ValueError('Final startup verification differs from the current online receipt')
    return initial_line, final_line, gate, bool(recovered)


def validate_training_rows(rows, first_round, minimum_updates):
    if len(rows) < minimum_updates:
        raise NotReady(f'Only {len(rows)} local resumed updates; need {minimum_updates}')
    rounds = sorted(rows)
    if rounds != list(range(first_round, rounds[-1] + 1)):
        raise ValueError('Actual attempt log has missing or unexpected resumed rounds')
    expected_keys = set(TRAINING_METRIC_MAP) | set(TRAINING_DERIVED_FIELDS)
    projected = {}
    for number, raw in rows.items():
        if raw.get('benchmark') is not False or raw.get('phase') != 'joint':
            raise ValueError('Expected actual joint production updates')
        values = project_training_metrics(raw)
        if set(values) != expected_keys:
            raise ValueError(f'Round{number} differs from full98-field training contract: '
                             f'missing={sorted(expected_keys - set(values))}')
        if values['phase/joint'] != 1 or values['phase/critic_warmup'] != 0:
            raise ValueError('Incorrect resumed phase indicators')
        if not 0 <= values['dropout/resolver_history_slots_fraction'] <= 1:
            raise ValueError('Resolver dropped-slot fraction is outside[0,1]')
        for key, value in values.items():
            if '/velocity_cosine' in key and not -1.000001 <= value <= 1.000001:
                raise ValueError(f'Invalid measured cosine: {key}={value}')
            if (key.startswith(('timing/', 'timing_stabilizer/', 'timing_resolver/'))
                    or key.endswith(('_rms', '/normalizer', '/difference'))) and value < 0:
                raise ValueError(f'Invalid measured diagnostic: {key}={value}')
        if any(values[f'learning_rate/{role}_next'] <= 0 for role in ('student', 'critic')):
            raise ValueError('Post-scheduler learning rates must be positive')
        projected[number] = values
    return projected


def verify_full_reports(remote, api, records, checkpoint, run_dir):
    """Independently compare downloaded report bytes with actual checkpoint JSON."""
    metadata, evaluation = read_json(checkpoint / 'metadata.json'), read_json(checkpoint / 'evaluation.complete.json')
    identity = evaluation['checkpoint_id']
    expected = {f'sensitivity_reports/{identity}/{entry["file"]}': checkpoint / entry['file']
                for entry in evaluation['reports']}
    definition_name = f'sensitivity_reports/{identity}/sensitivity-overview-definition.json'
    expected[definition_name] = run_dir / definition_name
    by_name = {record['name']: record for record in records}
    if set(by_name) != set(expected) or len(by_name) != len(records):
        raise ValueError('Startup receipt does not contain exactly the expected bounded report files')
    remote_files = {file.name: file for file in remote.files(names=list(expected))}
    if set(remote_files) != set(expected):
        raise NotReady('Not all sensitivity report files are visible remotely')
    definition = read_json(expected[definition_name])
    if definition['checkpoint_id'] != identity or definition['checkpoint_round'] != metadata['counters']['rounds']:
        raise ValueError('Remote report definition source identifies a different checkpoint')
    for entry in definition['reports']:
        if entry['sha256'] != sha(checkpoint / entry['file']):
            raise ValueError('Report definition digest differs from actual checkpoint source')
    verified = []
    with tempfile.TemporaryDirectory(prefix='joint-online-audit-') as temporary:
        for name, local in expected.items():
            if local.is_symlink() or not local.is_file() or local.stat().st_size > MAX_REPORT_BYTES:
                raise ValueError('Report verification requires a bounded real JSON file')
            digest, size = sha(local), local.stat().st_size
            record, file = by_name[name], remote_files[name]
            if (record['sha256'] != digest or record['bytes'] != size
                    or record['checkpoint_id'] != identity or file.size != size):
                raise ValueError('Report receipt or remote file size differs from actual source')
            with file.download(root=temporary, replace=True, api=api) as handle:
                downloaded_sha = sha(handle.name)
            if downloaded_sha != digest:
                raise ValueError('Downloaded remote report differs from actual checkpoint JSON')
            verified.append({'name': name, 'bytes': size, 'sha256': digest,
                             'remote_sha256_verified': True})
    return verified


def audit(job_dir, *, minimum_updates=5, maximum_rows=2000, api=None):
    job_dir = Path(job_dir).resolve()
    job = read_json(job_dir / 'job.json')
    if job['mode'] != 'train':
        raise ValueError('Expected a production job')
    resumes = [value.split('=', 1)[1] for value in job['overrides'] if value.startswith('resume_from=')]
    if len(resumes) != 1:
        raise ValueError('Expected exactly one explicit resume checkpoint')
    checkpoint, source = Path(resumes[0]), Path(job['source'])
    run_dir = checkpoint.parent
    pointer = read_json(run_dir / 'latest-resume.json')
    config, runtime = read_json(pointer['config']), read_json(pointer['runtime'])
    if runtime['source_directory'] != str(source) or runtime['resume_from'] != str(checkpoint):
        raise NotReady('Actual resume receipt does not yet identify this job')
    if sha(pointer['config']) != pointer['config_sha256'] or sha(pointer['runtime']) != pointer['runtime_sha256']:
        raise ValueError('Actual effective resume receipts differ from their committed digests')
    for name in ('distillation/joint_metrics.py', 'distillation/joint_sensitivity_tracking.py'):
        if sha(REPO / name) != sha(source / name):
            raise ValueError(f'Local audit definition differs from deployed immutable source: {name}')
    metadata = read_json(checkpoint / 'metadata.json')
    counters = metadata['counters']
    if runtime['restored_counters'] != counters or runtime['checkpoint_id'] != metadata['checkpoint_id']:
        raise ValueError('Actual restored counters/checkpoint ID differ from source')
    if runtime['benchmark'] or config['save_every_n_steps'] != 1000 or config['checkpoint_keep_last'] != 8:
        raise ValueError('Unexpected production mode, save cadence or retention')
    parsed = parsed_log(job_dir / 'node-0.log')
    receipt = read_json(run_dir / 'wandb.json')
    initial_gate_line, gate_line, gate, recovered = verified_startup_gate(
        parsed, checkpoint, counters, metadata['checkpoint_id'], receipt)
    metric_records = [(line, row) for line, row in parsed if 'rounds' in row]
    rows = {row['rounds']: row for _, row in metric_records}
    if len(rows) != len(metric_records) or len(rows) > maximum_rows:
        raise ValueError('Duplicate rounds or observation exceeds bounded maximum_rows')
    first = counters['rounds'] + 1
    expected = validate_training_rows(rows, first, minimum_updates)
    seen = counters['student_rows_seen']
    for number in sorted(rows):
        row, delta = rows[number], number - counters['rounds']
        seen += row['global_valid_rows']
        required_counters = {
            'warmup_step': counters['warmup_step'],
            'epoch_step': counters['epoch_step'] + delta,
            'student_updates': counters['student_updates'] + delta,
            'critic_updates': counters['critic_updates'] + delta * config['critic_updates'],
            'student_rows_seen': seen,
        }
        if any(row[key] != value for key, value in required_counters.items()):
            raise ValueError(f'Round{number} counters do not advance exactly from restored state')
    if api is None:
        import wandb
        api = wandb.Api(timeout=15)
    identity = f'{receipt["entity"]}/{receipt["project"]}/{receipt["id"]}'
    remote = api.run(identity)
    history = list(remote.scan_history(min_step=first, max_step=max(rows) + 1,
                                      page_size=100, use_cache=False))
    if len(history) < minimum_updates:
        raise NotReady(f'Only {len(history)} new history rows visible remotely; need {minimum_updates}')
    actual = {int(row['_step']): row for row in history}
    if sorted(actual) != list(range(first, max(actual) + 1)) or len(actual) != len(history):
        raise ValueError('Online resumed history has a duplicate or interior gap')
    for number, row in actual.items():
        compare_scalars(expected[number], row, f'history round{number}')
    api.flush()
    remote = api.run(identity)
    summary = dict(remote.summary)
    summary_round = summary.get('train/round')
    if summary_round not in rows:
        newer = {row['rounds']: row for _, row in parsed_log(job_dir / 'node-0.log') if 'rounds' in row}
        if summary_round not in newer:
            raise NotReady('Summary round has not yet appeared in this local attempt log')
        summary_expected = project_training_metrics(newer[summary_round])
    else:
        summary_expected = expected[summary_round]
    compare_scalars(summary_expected, summary, f'summary round{summary_round}')
    sensitivity = checkpoint_metrics(checkpoint)
    if len(sensitivity) != 22 or gate['verified_sensitivity_metrics'] != 22:
        raise ValueError('Expected the configured22-field compact sensitivity contract')
    compare_scalars(sensitivity, summary, 'compact checkpoint sensitivity')
    if summary.get('tracking/publish_token') != gate['publish_token']:
        raise ValueError('Remote startup token no longer matches this observation checkpoint')
    files = verify_full_reports(remote, api, receipt.get('sensitivity_report_files', []), checkpoint, run_dir)
    artifact_names = [artifact.name for artifact in remote.logged_artifacts()]
    if artifact_names:
        raise ValueError(f'Unexpected logged model/artifact objects: {artifact_names}')
    keys = sorted(next(iter(expected.values())))
    return {
        'schema_version': 1, 'passed': True, 'checked_at': datetime.now(timezone.utc).isoformat(),
        'job': str(job_dir / 'job.json'), 'job_id': job['id'], 'source_commit': runtime['source_commit'],
        'run_directory': str(run_dir), 'wandb_url': remote.url, 'checkpoint': str(checkpoint),
        'restored_counters': counters, 'actual_config': pointer['config'],
        'actual_config_sha256': pointer['config_sha256'], 'actual_runtime_sha256': pointer['runtime_sha256'],
        'save_every_n_steps': config['save_every_n_steps'],
        'next_periodic_round': (counters['rounds'] // config['save_every_n_steps'] + 1) * config['save_every_n_steps'],
        'startup_gate': {'line': gate_line, 'initial_line': initial_gate_line,
                         'recovered_evaluation': recovered, 'fresh_token': gate['publish_token'],
                         'verified_before_first_update': True, 'compact_sensitivity_fields': len(sensitivity)},
        'local_snapshot_range': [first, max(rows)], 'remote_verified_range': [first, max(actual)],
        'remote_rows_verified': len(actual), 'scalar_fields_per_row': len(keys),
        'history_scalar_values_checked': len(actual) * len(keys),
        'local_rows_not_yet_visible': sorted(set(rows) - set(actual)),
        'summary_round': summary_round, 'summary_fields_checked': len(summary_expected),
        'missing_fields': [], 'value_mismatches': [],
        'namespace_counts': dict(sorted(Counter(key.split('/')[0] for key in keys).items())),
        'all_training_keys': keys, 'compact_sensitivity_keys': sorted(sensitivity),
        'full_report_files': files, 'remote_logged_artifact_count': 0,
        'first_update': expected[first], 'last_verified_update': expected[max(actual)],
        'comparison_tolerance': {'relative': 1e-6, 'absolute': 1e-10},
        'read_only': True, 'scope': 'Every resumed history row visible at the bounded snapshot, '
            'at least five updates, current summary, compact checkpoint sensitivity and actual '
            'downloaded full JSON bytes verified. No job, SDK run or remote data mutations.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-dir', type=Path, required=True)
    parser.add_argument('--minimum-updates', type=int, default=5)
    parser.add_argument('--maximum-rows', type=int, default=2000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.minimum_updates < 5 or args.maximum_rows < args.minimum_updates:
        parser.error('Require at least five updates and maximum_rows >= minimum_updates')
    try:
        proof = audit(args.job_dir, minimum_updates=args.minimum_updates, maximum_rows=args.maximum_rows)
    except NotReady as error:
        print(json.dumps({'ready': False, 'reason': str(error)}))
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    temporary.write_text(json.dumps(proof, indent=2, allow_nan=False) + '\n')
    temporary.replace(args.output)
    print(json.dumps({'passed': True, 'artifact': str(args.output), 'sha256': sha(args.output),
                      'verified_rounds': proof['remote_verified_range'],
                      'fields_per_row': proof['scalar_fields_per_row']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
