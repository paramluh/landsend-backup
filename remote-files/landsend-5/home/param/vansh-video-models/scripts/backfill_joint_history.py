"""Prepare a canonical historical scalar archive without rewriting a live run.

Default is a local dry run. --apply creates a separate, labeled W&B archive
with resume='never' and verifies all uploaded history. --through-round must
identify a completed checkpoint. Each retained/rejected row is bound to its
source JSONL line and an exact matching production attempt log record.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import time

DEFAULT_RUN = Path('/mnt/paramfs/param/outputs/joint-nine-ref-20261018/training/run-20261018-101840-016723')
PROTECTED_RUN_ID = '0db61f9a18bf'
REPO = Path(__file__).resolve().parents[1]
RECOVERY_ROUND = 1314
RECOVERY_CHECKPOINT_ID = '23be3408cd624f94889e244e9ad1565b'
# This is the reviewed, immutable producer-03 failure receipt, not a blanket
# permission to accept missing completion events from other checkpoints.
DURABLE_RECOVERY_SHA256 = 'e5466eda7d877ed438452528c72bf77fc3cf779d59f4447ba30caf4f5fd2c76a'


def bytes_sha(value):
    return hashlib.sha256(value).hexdigest()


def sha(path):
    return bytes_sha(Path(path).read_bytes())


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def flat_numeric(value, prefix=''):
    """Baseline scalar inventory, preserving original per-row availability."""
    result = {}
    if isinstance(value, dict):
        for key, item in value.items():
            child = flat_numeric(item, f'{prefix}/{key}' if prefix else str(key))
            if result.keys() & child.keys():
                raise ValueError('Flattened metric-name collision')
            result.update(child)
    elif isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ValueError(f'Nonfinite historical scalar: {prefix}')
        result[prefix] = int(value) if isinstance(value, bool) else value
    elif isinstance(value, str) and prefix == 'phase':
        pass
    else:
        raise ValueError(f'Unexpected historical metric type: {prefix}/{type(value).__name__}')
    return result


def validate_counters(row):
    number = row.get('rounds')
    if type(number) is not int or number < 1:
        raise ValueError('Invalid original round number')
    warmup, student = min(number, 150), max(0, number - 150)
    expected = {'warmup_step': warmup, 'epoch_step': student, 'student_updates': student,
                'critic_updates': 4 * number, 'student_rows_seen': student * 40,
                'phase': 'critic_warmup' if number <= 150 else 'joint'}
    if any(row.get(k) != v for k, v in expected.items()) or row.get('benchmark') is not False:
        raise ValueError(f'Historical counters/phase mismatch at round{number}')
    flat_numeric(row)


def attempt_inventory(run_dir):
    """Find immutable launch evidence and exact metric bodies in attempt logs."""
    identities, by_hash = {}, {}
    decoder = json.JSONDecoder()
    for job in sorted((run_dir.parent.parent / 'jobs').glob('production-*')):
        jp, lp = job / 'job.json', job / 'node-0.log'
        if not(jp.is_file() and lp.is_file()):
            continue
        data = json.loads(jp.read_text())
        if data.get('mode') != 'train':
            continue
        resume_values = [v.split('=', 1)[1] for v in data.get('overrides', []) if v.startswith('resume_from=')]
        if len(resume_values) > 1:
            raise ValueError(f'Ambiguous resume source: {jp}')
        resume = Path(resume_values[0]).resolve() if resume_values else None
        if resume is not None and resume.parent != run_dir:
            continue
        if resume is None and job.name != 'production-01':
            continue
        source = Path(data['source'])
        found = []
        with lp.open(errors='replace') as handle:
            for line_number, line in enumerate(handle, 1):
                start = line.find('{"rounds":')
                if start < 0:
                    continue
                try:
                    row, _ = decoder.raw_decode(line[start:])
                except json.JSONDecodeError:
                    continue
                if 'phase' not in row or 'student_updates' not in row:
                    continue
                identity = bytes_sha(canonical(row).encode())
                found.append((row['rounds'], identity))
                by_hash.setdefault(identity, []).append({'attempt': job.name, 'attempt_log_line': line_number})
        if not found:
            continue
        resume_meta = json.loads((resume / 'metadata.json').read_text()) if resume else None
        first_round = resume_meta['counters']['rounds'] + 1 if resume_meta else 1
        if [n for n, _ in found] != list(range(first_round, first_round + len(found))):
            raise ValueError(f'Attempt metric log is not contiguous from its resume checkpoint: {job}')
        identities[job.name] = {
            'job_id': data['id'], 'created_at': data['created_at'],
            'job_manifest': str(jp), 'job_manifest_sha256': sha(jp),
            'code_directory': str(source), 'code_commit': source.name,
            'code_manifest_sha256': sha(source / 'source-manifest.json'),
            'config': data['config'], 'config_sha256': sha(data['config']),
            'overrides': data.get('overrides', []),
            'resume_checkpoint': str(resume) if resume else None,
            'resume_round': resume_meta['counters']['rounds'] if resume_meta else 0,
            'resume_metadata_sha256': sha(resume / 'metadata.json') if resume else None,
            'node_log': str(lp), 'observed_logged_rounds': [found[0][0], found[-1][0]],
            'observed_log_metric_rows': len(found),
            'observed_log_metric_bodies_sha256': bytes_sha(canonical(found).encode()),
        }
    return identities, by_hash


def select_history(run_dir, through):
    data = (run_dir / 'metrics.jsonl').read_bytes()
    lines = data.splitlines(keepends=True)
    # An active append may expose an unfinished trailing line. Never use it.
    if lines and not lines[-1].endswith(b'\n'):
        lines.pop()
    attempts, matching = attempt_inventory(run_dir)
    selected, rejected, counts = {}, [], Counter()
    for line_number, raw in enumerate(lines, 1):
        row = json.loads(raw)
        number = row.get('rounds')
        if not isinstance(number, int) or number > through:
            continue
        validate_counters(row)
        body_hash = bytes_sha(canonical(row).encode())
        matches = matching.get(body_hash, [])
        if len(matches) != 1:
            raise ValueError(f'Source row{line_number}/round{number} has{len(matches)} exact attempt matches')
        record = {'round': number, 'source_line': line_number, 'source_line_sha256': bytes_sha(raw),
                  'metric_body_sha256': body_hash, **matches[0]}
        counts[record['attempt']] += 1
        if number in selected:
            previous = selected[number][2]
            old_attempt, new_attempt = attempts[previous['attempt']], attempts[record['attempt']]
            if new_attempt['created_at'] <= old_attempt['created_at'] or new_attempt['resume_round'] >= number:
                raise ValueError(f'Unproven or reversed replay order at round{number}')
            rejected.append({**previous, 'reason': 'superseded_by_later_attempt',
                             'replacement_source_line': line_number, 'replacement_attempt': record['attempt']})
        selected[number] = (row, raw, record)
    if sorted(selected) != list(range(1, through + 1)):
        raise ValueError(f'Historical coverage is not complete and contiguous1..{through}')
    # Original601..699 must never masquerade as the later replay.
    stale = [n for n, (_, _, record) in selected.items() if 601 <= n <= 699 and record['attempt'] == 'production-01']
    if stale:
        raise ValueError(f'Original abandoned rounds still selected: {stale[:3]}')
    chosen = [selected[n] for n in range(1, through + 1)]
    evidence = {'attempts': attempts, 'input_rows_through_cutoff': sum(counts.values()),
                'input_rows_by_attempt': dict(counts),
                'selected_rows_by_attempt': dict(Counter(r[2]['attempt'] for r in chosen)),
                'superseded_rows': len(rejected), 'superseded_rows_by_attempt': dict(Counter(r['attempt'] for r in rejected)),
                'source_snapshot_complete_lines': len(lines),
                'source_snapshot_sha256': bytes_sha(b''.join(lines)),
                'source_snapshot_bytes': sum(map(len, lines))}
    return chosen, rejected, evidence


def verified_json(record):
    path = Path(record['path'])
    if path.is_symlink() or not path.is_file():
        raise ValueError('Recovery evidence file differs from its pinned hash')
    data = path.read_bytes()
    if bytes_sha(data) != record['sha256']:
        raise ValueError('Recovery evidence file differs from its pinned hash')
    return json.loads(data)


def verified_log_line(record):
    lines = Path(record['log']).read_bytes().splitlines(keepends=True)
    number = record['line']
    if type(number) is not int or not 1 <= number <= len(lines) or bytes_sha(lines[number-1]) != record['sha256']:
        raise ValueError('Recovery log evidence differs from its pinned line/hash')
    return lines[number-1], lines


def log_training_row(line):
    start = line.find(b'{"rounds":')
    if start < 0:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(line[start:].decode())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return value if 'phase' in value and 'student_updates' in value else None


def recovered_completion(receipt_path, run_dir, directory, metadata, completion,
                         training_row, row_provenance, source_evidence):
    """Accept only the explicitly reviewed round1314 durable-save recovery."""
    final_bytes = Path(receipt_path).read_bytes()
    final = json.loads(final_bytes)
    if final.get('schema') != 'joint-durable-checkpoint-recovery-completion-v1' or final.get('status') != 'completed':
        raise ValueError('Archive requires a completed recovery receipt, never a pending receipt')
    durable_ref = final['durable_save_receipt']
    if durable_ref['sha256'] != DURABLE_RECOVERY_SHA256:
        raise ValueError('Unreviewed durable-save recovery receipt')
    durable = verified_json(durable_ref)
    counters = metadata['counters']
    if not (durable.get('schema') == 'joint-durable-checkpoint-recovery-v1'
            and durable.get('status') == 'pending_startup_completion'
            and Path(durable['run_directory']).resolve() == run_dir.resolve()
            and Path(durable['checkpoint']).resolve() == directory.resolve()
            and durable['checkpoint_id'] == metadata['checkpoint_id'] == RECOVERY_CHECKPOINT_ID
            and counters == durable['counters'] and counters['rounds'] == RECOVERY_ROUND
            and row_provenance['attempt'] == 'production-03'
            and completion['checkpoint_id'] == metadata['checkpoint_id']
            and sha(directory/'metadata.json') == durable['checkpoint_metadata_sha256']
            and sha(directory/'checkpoint.complete.json') == durable['checkpoint_complete_sha256']
            and durable['latest_pointer'] == {'checkpoint': str(directory), 'counters': counters}):
        raise ValueError('Recovery receipt does not identify the reviewed durable checkpoint')
    entries = completion['files']
    if (len(entries) != durable['manifest_file_count'] or sum(e['size'] for e in entries) != durable['manifest_total_bytes']
            or durable['all_manifest_file_sizes_verified'] is not True
            or len([e for e in entries if e['path'].startswith('rank-') and e['path'].endswith('.pt')]) != durable['rank_shards']):
        raise ValueError('Recovered checkpoint inventory differs from the durable receipt')
    for entry in entries:
        path = directory/entry['path']
        if Path(entry['path']).name != entry['path'] or path.is_symlink() or not path.is_file() or path.stat().st_size != entry['size']:
            raise ValueError('Recovered checkpoint committed file is absent or changed')
    producer = source_evidence['attempts']['production-03']
    producer_job = verified_json({'path': durable['producer_job'], 'sha256': durable['producer_job_sha256']})
    if (producer['job_manifest'] != durable['producer_job'] or producer['job_manifest_sha256'] != durable['producer_job_sha256']
            or producer_job['source'] != durable['producer_source']):
        raise ValueError('Recovery producer differs from retained training provenance')
    last_line, producer_lines = verified_log_line(durable['producer_last_metric'])
    if log_training_row(last_line) != training_row:
        raise ValueError('Recovery producer metric differs from retained training row')
    if any(log_training_row(line) is not None for line in producer_lines[durable['producer_last_metric']['line']:]):
        raise ValueError('Recovery producer advanced beyond the reviewed durable save')
    failures = durable['producer_failure_lines']
    if not failures:
        raise ValueError('Recovery producer failure evidence is absent')
    for failure in failures:
        line, _ = verified_log_line({'log': durable['producer_last_metric']['log'], **failure})
        if failure['line'] <= durable['producer_last_metric']['line'] or b'Checkpoint identity does not belong to this run' not in line:
            raise ValueError('Recovery producer failure does not follow the retained metric')
    exits = durable['worker_exits']
    if len(exits) != 5 or any(e['job_id'] != producer_job['id'] or e['returncode'] == 0 for e in exits):
        raise ValueError('Recovery producer exit evidence is incomplete')
    request = durable['save_request']
    verified_json({'path': request['request_receipt'], 'sha256': request['request_receipt_sha256']})
    if sha(request['archived_marker']) != request['marker_sha256']:
        raise ValueError('Recovery save request marker differs')
    recovery_job = verified_json(final['recovery_job'])
    recovery_job_path = Path(final['recovery_job']['path']).resolve()
    if (recovery_job_path.parent.parent != Path(durable['producer_job']).resolve().parent.parent
            or recovery_job_path.parent.name != 'production-04' or recovery_job.get('mode') != 'train'
            or recovery_job['created_at'] <= producer_job['created_at']
            or [v.split('=',1)[1] for v in recovery_job.get('overrides',[]) if v.startswith('resume_from=')] != [str(directory)]):
        raise ValueError('Recovery launch is not production-04 resuming the exact durable checkpoint')
    resume = verified_json(final['resume_record'])
    if Path(resume['resume_from']).resolve() != directory.resolve():
        raise ValueError('Recovery resume record identifies another checkpoint')
    for name in ('config', 'runtime'):
        verified_json({'path': resume[name], 'sha256': resume[name+'_sha256']})
    event_ref = final['recovery_event']
    if Path(event_ref['log']).resolve() != recovery_job_path.parent/'node-0.log':
        raise ValueError('Recovery event is not from the recovery launch log')
    event_line, recovery_lines = verified_log_line(event_ref)
    event = json.loads(event_line)
    if (event.get('event') != 'checkpoint_evaluation_recovered' or event.get('checkpoint_id') != metadata['checkpoint_id']
            or Path(event.get('path','')).resolve() != directory.resolve() or event.get('counters') != counters
            or any(log_training_row(line) is not None for line in recovery_lines[:event_ref['line']-1])):
        raise ValueError('Recovery event does not prove evaluation before new training')
    evaluation = verified_json(final['evaluation'])
    if Path(final['evaluation']['path']).resolve() != directory/'evaluation.complete.json' or evaluation['checkpoint_id'] != metadata['checkpoint_id']:
        raise ValueError('Recovery evaluation identifies another checkpoint')
    expected_reports = {str(directory/r['file']) for r in evaluation['reports']}
    if {r['path'] for r in final['reports']} != expected_reports or len(final['reports']) != len(expected_reports):
        raise ValueError('Recovery report inventory differs from evaluation')
    for record in final['reports']:
        verified_json(record)
    tracker = event['tracking']
    if not (tracker.get('id') == PROTECTED_RUN_ID and tracker.get('online_metrics_verified') is True
            and tracker.get('verified_round') == RECOVERY_ROUND and tracker.get('verified_student_updates') == counters['student_updates']
            and tracker.get('verified_sensitivity_metrics') == 22 and tracker.get('sensitivity_report_uploads_verified') is True):
        raise ValueError('Recovery event lacks verified compact metrics and raw JSON publication')
    uploaded = tracker.get('sensitivity_report_files', [])
    for record in final['reports']:
        matching = [r for r in uploaded if r.get('name') == f"sensitivity_reports/{metadata['checkpoint_id']}/{Path(record['path']).name}"]
        if len(matching) != 1 or matching[0].get('sha256') != record['sha256'] or matching[0].get('remote_sha256_verified') is not True:
            raise ValueError('Recovered raw report lacks matching remote byte verification')
    if not final.get('gate_proofs'):
        raise ValueError('Recovery gate proof references are absent')
    for record in final['gate_proofs']:
        verified_json(record)
    return {'kind': 'recovered_evaluation_after_durable_save', 'receipt': str(receipt_path),
            'receipt_sha256': bytes_sha(final_bytes), 'durable_save_receipt': durable_ref,
            'producer_attempt': 'production-03', 'recovery_attempt': 'production-04',
            'recovery_event': event_ref, 'counters': counters}


def merge_checkpoint_sensitivity(run_dir, chosen, payload, source_evidence, recovery_receipt=None):
    """Project only existing evaluations belonging to retained training attempts.

    The reviewed helper validates raw report structure and any optimizer-prune
    UUID lineage. Completion events additionally bind each source checkpoint
    to the same attempt as its retained training row. No probe is rerun and no
    sensitivity value is forward-filled into ordinary training rounds.
    """
    sys.path.insert(0, str(REPO))
    helper = importlib.import_module('distillation.joint_sensitivity_tracking')
    through = len(payload)
    events = {}
    decoder = json.JSONDecoder()
    for attempt, evidence in source_evidence['attempts'].items():
        with Path(evidence['node_log']).open('rb') as handle:
            for line_number, raw in enumerate(handle, 1):
                line = raw.decode(errors='replace')
                start = line.find('{"event":')
                if start < 0:
                    continue
                try:
                    event, _ = decoder.raw_decode(line[start:])
                except json.JSONDecodeError:
                    continue
                if event.get('event') == 'checkpoint_complete':
                    events.setdefault((attempt, str(Path(event['path']).resolve())), []).append({
                        'attempt': attempt, 'node_log': evidence['node_log'],
                        'node_log_line': line_number, 'node_log_line_sha256': bytes_sha(raw),
                        'event_body_sha256': bytes_sha(canonical(event).encode()),
                        'counters': event['counters']})
    records, seen, recovery_used = [], set(), False
    for directory in sorted(run_dir.glob('step-*')):
        if not directory.is_dir():
            continue
        # Ignore later checkpoints and unfinished directories. Every completed
        # checkpoint inside this archive range must also have an evaluation.
        parts = directory.name.split('-')
        if len(parts) < 3 or not parts[1].isdigit() or int(parts[1]) > through:
            continue
        completion_path = directory / 'checkpoint.complete.json'
        if not completion_path.is_file():
            continue
        metadata_path, evaluation_path = directory / 'metadata.json', directory / 'evaluation.complete.json'
        metadata_bytes = metadata_path.read_bytes()
        completion_bytes = completion_path.read_bytes()
        evaluation_bytes = evaluation_path.read_bytes()
        metadata, completion, evaluation = map(json.loads, (metadata_bytes, completion_bytes, evaluation_bytes))
        number = metadata['counters']['rounds']
        if type(number) is not int or number != int(parts[1]) or not 1 <= number <= through or number in seen:
            raise ValueError('Duplicate or inconsistent sensitivity checkpoint round')
        if completion.get('checkpoint_id') != metadata['checkpoint_id']:
            raise ValueError('Sensitivity checkpoint is not committed under its current identity')
        training_row, _, row_provenance = chosen[number - 1]
        if any(training_row.get(k) != v for k, v in metadata['counters'].items()):
            raise ValueError('Sensitivity checkpoint counters differ from retained training row')
        matching = events.get((row_provenance['attempt'], str(directory.resolve())), [])
        if len(matching) == 1 and matching[0]['counters'] == metadata['counters']:
            completion_evidence = matching[0]
        elif not matching and recovery_receipt is not None and number == RECOVERY_ROUND:
            completion_evidence = recovered_completion(recovery_receipt, run_dir, directory, metadata, completion,
                                                       training_row, row_provenance, source_evidence)
            recovery_used = True
        else:
            raise ValueError('Sensitivity checkpoint lacks exact completion evidence in the retained attempt')
        before = {str(path): bytes_sha(data) for path, data in (
            (metadata_path, metadata_bytes), (completion_path, completion_bytes), (evaluation_path, evaluation_bytes))}
        reports = []
        for report in evaluation.get('reports', []):
            label = report.get('weights')
            if label not in ('raw', 'ema') or report.get('file') != f'sensitivity-{label}.json':
                raise ValueError('Unreviewed sensitivity source path')
            path = directory / report['file']
            data = path.read_bytes()
            before[str(path)] = bytes_sha(data)
            reports.append({'weights': label, 'source_path': str(path), 'sha256': bytes_sha(data),
                            'bytes': len(data),
                            'original_run_file': f"sensitivity_reports/{evaluation['checkpoint_id']}/{path.name}"})
        prune_path = directory.with_name(directory.name + '.optimizer-prune.json')
        if prune_path.exists():
            before[str(prune_path)] = sha(prune_path)
        compact = helper.checkpoint_metrics(directory)
        if before != {path: sha(path) for path in before}:
            raise ValueError('Sensitivity source changed while preparing archive')
        if compact.get('sensitivity_overview/checkpoint_round') != number or any(
                type(value) not in (int, float) or not math.isfinite(value) for value in compact.values()):
            raise ValueError('Invalid compact sensitivity projection')
        if any(not key.startswith(tuple(group + '/' for group in helper.GROUPS)) for key in compact):
            raise ValueError('Unexpected compact sensitivity namespace')
        if payload[number - 1].keys() & compact.keys():
            raise ValueError('Sensitivity projection overlaps a training scalar')
        payload[number - 1].update(compact)
        seen.add(number)
        records.append({'round': number, 'source_checkpoint': str(directory),
                        'checkpoint_id': metadata['checkpoint_id'],
                        'original_evaluation_checkpoint_id': evaluation['checkpoint_id'],
                        'source_sha256': before, 'reports': reports,
                        'retained_training_row': row_provenance, 'completion_event': completion_evidence,
                        'compact_scalar_count': len(compact),
                        'compact_metrics_sha256': bytes_sha(canonical(compact).encode())})
    if recovery_receipt is not None and not recovery_used:
        raise ValueError('Supplied recovery receipt did not identify a missing reviewed completion event')
    return {'projection_module': helper.__name__, 'projection_sha256': sha(helper.__file__),
            'axis': 'sensitivity_overview/checkpoint_round', 'groups': list(helper.GROUPS),
            'checkpoint_rounds': sorted(seen), 'checkpoints': records,
            'raw_reports_policy': 'Existing raw JSON remains unchanged. Source paths, byte hashes and original-run file names are preserved; this archive uploads scalar projections only.',
            'absence_policy': 'Only actual completed/evaluated checkpoint rows receive sensitivity values. No new probes, interpolation or forward fill.'}


def prepare(run_dir, output, through, recovery_receipt=None):
    if type(through) is not int or through < 600:
        raise ValueError('Canonical archive cutoff must be an explicit completed round >=600')
    config = json.loads((run_dir / 'config.json').read_text())
    epoch = json.loads((run_dir / 'epoch-plan.json').read_text())
    runtime = json.loads((run_dir / 'runtime.json').read_text())
    online = json.loads((run_dir / 'wandb.json').read_text())
    if runtime['benchmark'] is not False or config['wandb']['enabled'] is not False:
        raise ValueError('Expected the original production directory with local-only initial metrics')
    if online['id'] != PROTECTED_RUN_ID or online['entity'] != 'canopy-labs' or online['project'] != 'param-distillation':
        raise ValueError('Current online identity differs from the reviewed target')
    completed = []
    for cp in run_dir.glob(f'step-{through:06d}-*'):
        if not(cp / 'checkpoint.complete.json').is_file():
            continue
        meta = json.loads((cp / 'metadata.json').read_text())
        marker = json.loads((cp / 'checkpoint.complete.json').read_text())
        if marker['checkpoint_id'] == meta['checkpoint_id'] and meta['counters']['rounds'] == through:
            completed.append(cp)
    if len(completed) != 1:
        raise ValueError('Cutoff must identify exactly one durable completed checkpoint')
    chosen, rejected, source_evidence = select_history(run_dir, through)
    sys.path.insert(0, str(REPO))
    projection = importlib.import_module('distillation.joint_metrics')
    payload = [projection.project_training_metrics(row) for row, _, _ in chosen]
    training_fields = Counter(key for row in payload for key in row)
    sensitivity = merge_checkpoint_sensitivity(run_dir, chosen, payload, source_evidence, recovery_receipt)
    if through not in sensitivity['checkpoint_rounds']:
        raise ValueError('Archive cutoff has no validated completed sensitivity evaluation')
    for index, row in enumerate(payload, 1):
        if row.get('train/round') != index or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in row.values()):
            raise ValueError(f'Invalid canonical scalar payload at round{index}')
    source_bytes = b''.join(raw for _, raw, _ in chosen)
    payload_bytes = ''.join(canonical(row) + '\n' for row in payload).encode()
    source_hash, payload_hash = bytes_sha(source_bytes), bytes_sha(payload_bytes)
    archive_id = f'archive1to{through}-' + payload_hash[:12]
    assert archive_id != PROTECTED_RUN_ID
    output.mkdir(parents=True, exist_ok=True)
    (output / 'selected-original-metrics.jsonl').write_bytes(source_bytes)
    (output / 'canonical-payload.jsonl').write_bytes(payload_bytes)
    (output / 'selected-row-provenance.jsonl').write_text(''.join(canonical(r) + '\n' for _, _, r in chosen))
    (output / 'superseded-row-provenance.jsonl').write_text(''.join(canonical(r) + '\n' for r in rejected))
    fields = Counter(k for row in payload for k in row)
    raw_fields = Counter(k for row, _, _ in chosen for k in flat_numeric(row))
    manifest = {
        'schema': 'joint-canonical-history-archive-v3',
        'prepared_at_utc': datetime.now(timezone.utc).isoformat(), 'remote_mutation_performed': False,
        'archive_run_id': archive_id,
        'archive_run_name': f'ARCHIVE canonical production rounds1-{through} (recorded history)',
        'entity': online['entity'], 'project': online['project'],
        'protected_online_run_id': PROTECTED_RUN_ID, 'protected_online_run_url': online['url'],
        'source_run_directory': str(run_dir), 'source_metrics_path': str(run_dir / 'metrics.jsonl'),
        'selected_source_sha256': source_hash, 'payload_sha256': payload_hash,
        'source_config_sha256': sha(run_dir / 'config.json'), 'source_epoch_plan_sha256': sha(run_dir / 'epoch-plan.json'),
        'source_runtime_sha256': sha(run_dir / 'runtime.json'), 'script_sha256': sha(__file__),
        'projection_module': projection.__name__, 'projection_sha256': sha(projection.__file__),
        'first_round': 1, 'last_round': through, 'rows': len(chosen),
        'phase_counts': dict(Counter(row['phase'] for row, _, _ in chosen)),
        'canonical_scalar_field_counts': dict(sorted(fields.items())), 'original_scalar_field_counts': dict(sorted(raw_fields.items())),
        'training_scalar_field_counts': dict(sorted(training_fields.items())),
        'sensitivity_provenance': sensitivity,
        'source_teacher_sha256': epoch['signature']['teacher_sha256'], 'source_data_signature': epoch['signature']['data'],
        'last_counters': {k:chosen[-1][0][k] for k in ('rounds','warmup_step','epoch_step','student_updates','critic_updates','student_rows_seen')},
        'cutoff_checkpoint': str(completed[0]), 'cutoff_metadata_sha256': sha(completed[0]/'metadata.json'),
        'cutoff_completion_sha256': sha(completed[0]/'checkpoint.complete.json'),
        'attempt_provenance': source_evidence,
        'row_provenance_sha256': sha(output/'selected-row-provenance.jsonl'),
        'superseded_provenance_sha256': sha(output/'superseded-row-provenance.jsonl'),
        'history_policy': 'Latest recorded occurrence per round, validated against a later production launch/resume and exact node-0.log metric body. Original abandoned601..699 are explicitly rejected. Current online history is never rewritten.',
        'missing_fields_policy': 'Use the same pure canonical projection as current production. Omit absent old diagnostics, absent student loss in critic warmup, and unrecorded lr_next. Sensitivity uses only the reviewed compact projection of existing, provenance-validated checkpoint evaluations.',
        'timestamp_policy': 'W&B ingestion timestamps describe archive upload time; train/round is the original training x-axis.',
        'remote_policy': 'Distinct archive; resume=never; fail if ID exists; no writes to current run or workspace; no weights, code or artifacts.',
        'official_logging_reference': 'https://docs.wandb.ai/ref/python/experiments/run/',
    }
    (output/'dry-run-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest, payload, config


def verify_history(api, manifest, payload, timeout):
    path=f"{manifest['entity']}/{manifest['project']}/{manifest['archive_run_id']}"
    deadline=time.monotonic()+timeout
    while True:
        api.flush(); remote=api.run(path); actual=list(remote.scan_history(page_size=1000))
        by_round={row.get('train/round'):row for row in actual}
        if len(actual)==len(payload) and len(by_round)==len(payload):
            mismatch=[]
            for expected in payload:
                got=by_round.get(expected['train/round'],{})
                for key,value in expected.items():
                    if key not in got or not isinstance(got[key],(int,float)) or not math.isclose(got[key],value,rel_tol=1e-10,abs_tol=1e-12):
                        mismatch.append([expected['train/round'],key])
                extras=set(got)-set(expected)-{'_step','_timestamp','_runtime'}
                if any(got[k] is not None for k in extras) or got.get('_step')!=expected['train/round']:
                    mismatch.append([expected['train/round'],'unexpected_fields_or_step'])
            if not mismatch:
                return {'remote_rows':len(actual),'verified_scalar_values':sum(map(len,payload)),'history_mismatches':[],'url':remote.url}
        if time.monotonic()>=deadline:
            raise RuntimeError('Archive history did not match every prepared scalar before timeout')
        time.sleep(2)


def apply(manifest,payload,config,output,timeout):
    import wandb
    api=wandb.Api(timeout=30)
    existing=list(api.runs(f"{manifest['entity']}/{manifest['project']}",filters={'name':manifest['archive_run_id']},per_page=1))
    if existing or manifest['archive_run_id']==PROTECTED_RUN_ID:
        raise ValueError('Archive exists or targets current run; refusing to overwrite/resume')
    notes=(f"Canonical historical scalar archive, rounds1..{manifest['last_round']}. Not a new training experiment. "
           'Exact local source rows are attributed to their production attempts; abandoned original601..699 are excluded. '
           'Current online history is unchanged; unavailable older diagnostics remain absent.')
    run=wandb.init(entity=manifest['entity'],project=manifest['project'],id=manifest['archive_run_id'],
                   name=manifest['archive_run_name'],job_type='historical-metrics-archive',
                   tags=['historical-archive','canonical-metrics-backfill'],notes=notes,
                   mode='online',resume='never',save_code=False,
                   config={'archive_provenance':manifest,'original_training_config':config},dir=str(output),
                   settings={'console':'off','disable_code':True,'disable_git':True,'x_disable_stats':True,
                             'x_disable_meta':True,'init_timeout':90})
    try:
        if run.id!=manifest['archive_run_id'] or run.id==PROTECTED_RUN_ID or run.offline:
            raise RuntimeError('Unexpected archive identity/mode')
        run.define_metric('train/round');run.define_metric('*',step_metric='train/round')
        run.define_metric('sensitivity_overview/checkpoint_round')
        for group in manifest['sensitivity_provenance']['groups']:
            run.define_metric(group+'/*',step_metric='sensitivity_overview/checkpoint_round')
        for row in payload:run.log(row,step=row['train/round'],commit=True)
        run.finish()
    except BaseException:
        run.finish(exit_code=1);raise
    verified=verify_history(api,manifest,payload,timeout)
    remote=api.run(f"{manifest['entity']}/{manifest['project']}/{manifest['archive_run_id']}")
    files=[f.name for f in remote.files()]
    forbidden=[n for n in files if n.endswith(('.safetensors','.pt','.pth','.py','.patch')) or n.startswith('code/')]
    if forbidden or list(remote.logged_artifacts()):raise RuntimeError('Unexpected model/code/artifact upload')
    receipt={**verified,'archive_run_id':manifest['archive_run_id'],'protected_online_run_id':PROTECTED_RUN_ID,
             'remote_mutation_performed':True,'payload_sha256':manifest['payload_sha256'],
             'selected_source_sha256':manifest['selected_source_sha256'],'remote_files':files,
             'model_code_artifacts_uploaded':False,'completed_at_utc':datetime.now(timezone.utc).isoformat()}
    (output/'applied-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    parser.add_argument('--through-round',type=int,required=True)
    parser.add_argument('--output-dir',type=Path,default=Path('/tmp/joint-history-archive-20261018'))
    parser.add_argument('--recovery-receipt',type=Path,help='Explicit completed receipt for the reviewed round1314 recovery only')
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--verify-timeout',type=float,default=90)
    args=parser.parse_args()
    manifest,payload,config=prepare(args.run_dir.resolve(),args.output_dir.resolve(),args.through_round,args.recovery_receipt)
    if args.apply:apply(manifest,payload,config,args.output_dir.resolve(),args.verify_timeout)
    else:
        print(json.dumps({k:manifest[k] for k in ['archive_run_id','rows','phase_counts','selected_source_sha256','payload_sha256','remote_mutation_performed']},indent=2))
        print('Dry-run manifest:',args.output_dir/'dry-run-manifest.json')


if __name__=='__main__':main()
