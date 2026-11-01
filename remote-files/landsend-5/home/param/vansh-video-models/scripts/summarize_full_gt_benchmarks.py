"""Summarize completed full-GT trials and their per-device telemetry.

The trainer records distributed MAX values, so round times, peaks and compiler
deltas are worst-rank observations. GPU CSV timestamps are interpreted as UTC
(the benchmark hosts use UTC). The measured telemetry window is approximate:
completion time minus the sum of measured round wall times. Summary writing,
collectives and logging outside the timed rounds can shift this window slightly.
"""

import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
from statistics import median


DEFAULT_ROOT = Path('/mnt/paramfs/param/outputs/fullgt-performance-20261020-0505')
COMPILE_FIELDS = ('compile/new_graphs', 'compile/new_dynamo_seconds', 'compile/new_backward_seconds')


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def read_metrics(path):
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    result = []
    for index, line in enumerate(lines):
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            # A live trainer may be midway through writing its final line.
            if index != len(lines) - 1:
                raise
    return result


def stats(values):
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return {'count': len(values), 'median': median(values), 'min': min(values), 'max': max(values)} if values else None


def timestamp(text):
    parsed = datetime.fromisoformat(text.replace('Z', '+00:00'))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def telemetry_rows(trial):
    records, rejected = [], 0
    for path in sorted(trial.glob('node-*/gpu.csv')):
        with path.open(newline='') as handle:
            for row in csv.reader(handle, skipinitialspace=True):
                try:
                    if len(row) != 8:
                        raise ValueError('Incomplete telemetry row')
                    when = datetime.strptime(row[0], '%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=timezone.utc)
                    record = {'at': when, 'node': path.parent.name, 'index': int(row[1]), 'uuid': row[2],
                              'util_gpu_percent': float(row[3]), 'util_memory_percent': float(row[4]),
                              'used_gib': float(row[5]) / 1024, 'total_gib': float(row[6]) / 1024,
                              'power_watts': float(row[7])}
                    if not all(math.isfinite(record[key]) for key in ('util_gpu_percent', 'util_memory_percent',
                                                                       'used_gib', 'total_gib', 'power_watts')):
                        raise ValueError('Nonfinite telemetry value')
                    records.append(record)
                except (ValueError, IndexError):
                    rejected += 1
    return records, rejected


def trial_state(trial, completion):
    """Inspect exit receipts and at most 1 MiB from node zero's log.

    An absent terminal receipt is called incomplete, not running: process
    liveness requires a separate check against the live supervisor/process.
    """
    exits = [read_json(path) for path in sorted(trial.glob('node-*/exit.json'))]
    failed = any(item.get('returncode', 0) != 0 for item in exits)
    failed = failed or bool(completion and not completion.get('all_successful', False))
    status = 'failed' if failed else 'complete' if completion else 'incomplete'
    log = trial / 'node-0/train.log'
    text = ''
    if log.exists():
        with log.open('rb') as handle:
            handle.seek(max(0, log.stat().st_size - 1024 * 1024))
            text = handle.read(1024 * 1024).decode(errors='replace')
    phases = []
    for line in text.splitlines():
        if '"event": "benchmark_phase_started"' not in line:
            continue
        try:
            event = json.loads(line[line.index('{'):])
            if event.get('event') == 'benchmark_phase_started':
                phases.append(event)
        except (ValueError, json.JSONDecodeError):
            pass
    last = phases[-1] if phases else None
    error_lines = [line.strip() for line in text.splitlines()
                   if 'OutOfMemoryError' in line or 'CUDA out of memory' in line]
    failure_kind = 'oom' if error_lines else 'worker_error' if failed else None
    phase = last.get('phase') if last else None
    phase_evidence = 'last_benchmark_phase_event' if phase else None
    if failed:
        # The final trace is more specific than a broad "critic" phase event.
        role = 'critic_pool' if 'in generate_pooled_targets' in text else 'critic' if 'in critic_update' in text else None
        objective = 'resolver' if 'in score_resolver' in text else 'stabilizer' if 'in score_pass' in text else None
        if role or objective:
            phase = '/'.join(value for value in (role, objective) if value)
            phase_evidence = 'inferred_from_node0_traceback'
    teardown = read_json(trial / 'teardown-stop.json')
    if teardown and teardown.get('all_40_rank_completion_verified') is True:
        # Retain raw nonzero exit receipts, but distinguish an explicitly
        # stopped cleanup phase from failure of the measured training work.
        status = 'benchmark_complete_teardown_stopped'
        failure_kind, phase = 'teardown', 'shutdown'
        phase_evidence = 'recorded_stop_after_all_rank_completion'
    return {'status': status, 'failure_kind': failure_kind,
            'failure_phase': phase if failed else None,
            'cold_last_phase': last if last else ({'phase': phase} if phase else None),
            'phase_evidence': phase_evidence,
            'error_excerpt': error_lines[-1][:1200] if error_lines else None,
            'node_returncodes': [item.get('returncode') for item in exits],
            'teardown_stop': teardown,
            'log_tail_bytes_limit': 1024 * 1024}


def summarize_telemetry(records, expected_devices=40):
    if not records:
        return {'samples': 0, 'device_count': 0, 'expected_devices': expected_devices, 'complete_device_coverage': False}
    devices = {}
    for row in records:
        devices.setdefault((row['node'], row['uuid']), []).append(row)
    per_device = []
    for (node, uuid), values in sorted(devices.items()):
        per_device.append({'node': node, 'index': values[0]['index'], 'uuid': uuid,
                           'samples': len(values),
                           'gpu_util_percent': stats(row['util_gpu_percent'] for row in values),
                           'memory_util_percent': stats(row['util_memory_percent'] for row in values),
                           'device_used_gib': stats(row['used_gib'] for row in values),
                           'power_watts': stats(row['power_watts'] for row in values)})
    return {'samples': len(records), 'device_count': len(per_device), 'expected_devices': expected_devices,
            'complete_device_coverage': len(per_device) == expected_devices,
            'first_at_utc': min(row['at'] for row in records).isoformat(),
            'last_at_utc': max(row['at'] for row in records).isoformat(),
            'gpu_util_device_medians_percent': stats(item['gpu_util_percent']['median'] for item in per_device),
            'gpu_util_all_samples_percent': stats(row['util_gpu_percent'] for row in records),
            'device_peak_used_gib': stats(item['device_used_gib']['max'] for item in per_device),
            'power_device_medians_watts': stats(item['power_watts']['median'] for item in per_device),
            'per_device': per_device}


def summarize_run(run, telemetry, skipped_default=2):
    completion = read_json(run / 'benchmark.complete.json')
    rows = read_metrics(run / 'metrics.jsonl')
    skipped = int(completion.get('skipped_updates', skipped_default)) if completion else skipped_default
    expected = int(completion['measured_updates']) if completion else None
    measured = rows[skipped:]
    missing = sorted({key for row in measured for key in COMPILE_FIELDS if key not in row})
    finite_compilation = all(math.isfinite(float(row[key])) for row in measured for key in COMPILE_FIELDS if key in row)
    deltas = {key: sum(float(row[key]) for row in measured if key in row) for key in COMPILE_FIELDS}
    compiler_stable = bool(measured) and not missing and finite_compilation and all(
        float(row[key]) == 0 for row in measured for key in COMPILE_FIELDS)
    graph_counts = [item['unique_graphs'] for item in completion.get('ranks', [])] if completion else []
    expected_devices = int(completion.get('world_size', 40)) if completion else 40
    rank_ids = {item['rank'] for item in completion.get('ranks', [])} if completion else set()
    compiled = (len(graph_counts) == expected_devices and rank_ids == set(range(expected_devices))
                and all(value > 0 for value in graph_counts))
    complete = bool(completion and completion.get('complete') and len(measured) == expected)
    end = timestamp(completion['completed_at_utc']) if completion else None
    duration = sum(float(row['seconds/wall']) for row in measured)
    start = end - timedelta(seconds=duration) if end else None
    steady_rows = [row for row in telemetry if start <= row['at'] <= end] if end else []
    cold_rows = [row for row in telemetry if row['at'] < start] if start else telemetry
    timing_keys = sorted({key for row in measured for key in row if key.startswith('seconds/')})
    objective_names = sorted({key for row in measured for key in row.get('global_objective_targets', {})})
    throughput = {kind: stats(row['global_objective_targets'][kind] / row['seconds/wall']
                              for row in measured if row['seconds/wall'] > 0)
                  for kind in objective_names}
    return {'run': str(run), 'status': 'complete' if complete else 'incomplete',
            'skipped_rounds': skipped, 'logged_rounds': len(rows), 'measured_rounds': len(measured),
            'expected_measured_rounds': expected,
            'median_seconds_per_global_round': median(row['seconds/wall'] for row in measured) if measured else None,
            'round_seconds': stats(row['seconds/wall'] for row in measured),
            'target_frames_per_second_per_objective': throughput,
            'phase_seconds': {key.removeprefix('seconds/'): stats(row[key] for row in measured if key in row)
                              for key in timing_keys},
            'allocated_peak_gib': max((float(row['memory/peak_allocated_gib']) for row in measured), default=None),
            'reserved_peak_gib': max((float(row['memory/peak_reserved_gib']) for row in measured), default=None),
            'compiler': {'stable_measured_rounds': compiler_stable, 'compiled_on_all_ranks': compiled,
                         'stable_completed_benchmark': complete and compiler_stable and compiled,
                         'missing_measurement_fields': missing, 'final_graphs_per_rank': stats(graph_counts),
                         'final_graphs_by_rank': [{'rank': item['rank'], 'graphs': item['unique_graphs']}
                                                   for item in completion.get('ranks', [])] if completion else [],
                         'measured_delta_sum_of_round_rank_maxima': deltas,
                         'cold_rounds': [{key: row.get(key) for key in ('seconds/wall', *COMPILE_FIELDS)}
                                         for row in rows[:skipped]],
                         'final_cumulative_timers_rank_max_seconds': {
                             key: rows[-1].get(key) for key in ('compile/dynamo_seconds_total', 'compile/backward_seconds_total')
                         } if rows else {}},
            'steady_telemetry_window': {'approximate': True, 'method': 'completion timestamp minus sum of measured round wall times',
                                       'limitations': 'Host timestamps assumed UTC; summary/logging/collective overhead outside round timers may shift the window.',
                                       'start_utc': start.isoformat() if start else None,
                                       'end_utc': end.isoformat() if end else None, 'duration_seconds': duration if end else None},
            'steady_telemetry': summarize_telemetry(steady_rows, expected_devices),
            'cold_telemetry': summarize_telemetry(cold_rows, expected_devices)}


def summarize(root, skipped_default=2):
    trials = []
    for trial in sorted(root.iterdir()):
        if not trial.is_dir() or not (trial / 'plan.json').exists():
            continue
        plan = read_json(trial / 'plan.json')
        if not plan.get('benchmark', True):
            continue
        telemetry, rejected = telemetry_rows(trial)
        runs = sorted(path for path in (trial / 'runs').glob('*') if path.is_dir())
        completion = read_json(trial / 'cluster.complete.json')
        total_telemetry = summarize_telemetry(telemetry)
        peak = total_telemetry.get('device_peak_used_gib')
        trials.append({'trial': trial.name, 'directory': str(trial), 'overrides': plan.get('overrides', []),
                       **trial_state(trial, completion),
                       'peak_device_used_gib_total': peak['max'] if peak else None,
                       'cluster_completion': completion, 'watchdog': read_json(trial / 'watchdog.json'),
                       'telemetry_rejected_rows': rejected, 'total_telemetry': total_telemetry,
                       'runs': [summarize_run(run, telemetry, skipped_default) for run in runs
                                if (run / 'metrics.jsonl').exists() or (run / 'benchmark.complete.json').exists()]})
    return {'schema_version': 1, 'root': str(root), 'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'definitions': {'global_round': 'Four sequential critic optimizer updates and one student optimizer update.',
                            'timing_reduction': 'Each round metric is the maximum across distributed ranks.',
                            'phase_times': 'Nested phases overlap; do not sum phase timings.',
                            'gpu_util_summary': 'Median, minimum and maximum of individual device time medians; raw sample range also provided.',
                            'memory': 'Allocated/reserved are PyTorch round peaks; device-used includes runtime, contexts and allocator reservations.',
                            'stable': 'Completed compiled benchmark with zero new Dynamo graphs and zero Dynamo/lazy-backward compiler seconds in every measured round.'},
            'trials': trials}


def table(summary):
    lines = ['Trial | Status / last cold phase | Round s (median) | Target/s/objective | GPU util % (median; device range) | Alloc/reserved/device GiB | Graphs/rank | Stable',
             '--- | --- | ---: | ---: | --- | --- | --- | ---']
    for trial in summary['trials']:
        state = trial['status'] + ('/' + trial['failure_kind'] if trial['failure_kind'] else '')
        phase = trial['failure_phase'] or (trial['cold_last_phase'] or {}).get('phase')
        state += f'; {phase}' if phase else ''
        if not trial['runs']:
            peak = trial['peak_device_used_gib_total']
            memory = f'—/—/{peak:.2f} (total)' if peak is not None else '—'
            lines.append(f"{trial['trial']} | {state} | — | — | — | {memory} | — | no")
        for run in trial['runs']:
            def number(value):
                return '—' if value is None else f'{value:.2f}'
            seconds = run['median_seconds_per_global_round']
            rate = run['target_frames_per_second_per_objective']
            rate = ', '.join(f'{key}: {value["median"]:.2f}' for key, value in rate.items()) or '—'
            util = run['steady_telemetry'].get('gpu_util_device_medians_percent')
            util_text = f'{util["median"]:.1f}; {util["min"]:.1f}–{util["max"]:.1f}' if util else '—'
            used = run['steady_telemetry'].get('device_peak_used_gib')
            peak = used['max'] if used else trial['peak_device_used_gib_total']
            memory = '/'.join(number(value) for value in (run['allocated_peak_gib'], run['reserved_peak_gib'], peak))
            if used is None and peak is not None:
                memory += ' (device total)'
            graphs = run['compiler']['final_graphs_per_rank']
            graphs_text = f'{graphs["min"]:g}–{graphs["max"]:g}' if graphs else '—'
            stable = 'yes' if run['compiler']['stable_completed_benchmark'] else 'no'
            lines.append(f'{trial["trial"]} | {state} | {number(seconds)} | {rate} | {util_text} | {memory} | {graphs_text} | {stable}')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, nargs='?', default=DEFAULT_ROOT)
    parser.add_argument('--output', type=Path, help='Save complete JSON in addition to stdout.')
    parser.add_argument('--format', choices=('json', 'table'), default='table')
    parser.add_argument('--skip', type=int, default=2, help='Fallback skipped rounds when no completion receipt exists.')
    args = parser.parse_args()
    summary = summarize(args.root, args.skip)
    encoded = json.dumps(summary, indent=2, allow_nan=False) + '\n'
    if args.output:
        temporary = args.output.with_suffix(args.output.suffix + '.tmp')
        temporary.write_text(encoded)
        temporary.replace(args.output)
    print(table(summary) if args.format == 'table' else encoded, end='\n' if args.format == 'table' else '')


if __name__ == '__main__':
    main()
