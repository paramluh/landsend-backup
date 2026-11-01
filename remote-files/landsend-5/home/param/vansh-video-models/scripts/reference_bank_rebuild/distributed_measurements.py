#!/usr/bin/env python3
"""Supervise incremental upstream measurements across all allocated nodes.

Run this launcher from an immutable snapshot on shared paramfs. This preserves
the same measurement_config paths on every node. Workers write disjoint track
checkpoints and shard-local reports; this supervisor publishes the global status expected
by run_rebuild.py. Stop the old standalone watcher before starting this helper.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from distributed_scan import DEFAULT_NODES, remote_command
from hf_sources import write_json
from measure_upstream import checkpoint_valid, measurement_config
from measure_watcher import shard_report_dir
from process_output import CPU_ENV, start_logged


def utc():
    return datetime.now(timezone.utc).isoformat()


def provision(node, args, source, config, local_nodes):
    code = (f'import sys; sys.path.insert(0, {str(source)!r}); '
            'import json, socket, mediapipe; from measure_upstream import measurement_config; '
            "print(json.dumps({'hostname': socket.gethostname(), 'mediapipe': mediapipe.__version__, "
            "'config': measurement_config()}))")
    argv = ['env', *CPU_ENV, args.python, '-c', code]
    command = argv if node in local_nodes else remote_command(node, argv)
    result = subprocess.run(command, stdout=subprocess.PIPE, text=True, timeout=60)
    print(result.stdout, end='', flush=True)
    if result.returncode:
        raise RuntimeError(f'Node {node} measurement preflight failed ({result.returncode})')
    result = json.loads(result.stdout.strip().splitlines()[-1])
    if result.pop('config') != config:
        raise ValueError(f'Node {node} measurement config differs; existing checkpoints cannot be reused')
    return {'node': node, **result}


def read_current(path, run_id):
    if not path.is_file():
        return None
    value = json.loads(path.read_text())
    return value if value.get('run_id') == run_id else None


def run(args):
    root, source = args.output_dir, Path(__file__).resolve().parent
    if not source.is_relative_to('/mnt/paramfs'):
        raise ValueError('Run distributed_measurements.py from an immutable shared paramfs code snapshot')
    hashes = json.loads((source / 'source-hashes.json').read_text())
    if any(hashlib.sha256((source / name).read_bytes()).hexdigest() != digest for name, digest in hashes.items()):
        raise ValueError('Shared measurement code snapshot changed')
    existing_runtime = root / 'watch-runtime.json'
    if existing_runtime.is_file():
        pid = json.loads(existing_runtime.read_text()).get('pid')
        if pid and pid != os.getpid():
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError(f'Existing measurement watcher PID {pid} is alive; stop it before distributing')
    tracks = [line for line in args.tracks.read_text().splitlines() if line]
    if len(set(tracks)) != len(tracks):
        raise ValueError('Duplicate tracks in measurement plan')
    track_hash = hashlib.sha256(args.tracks.read_bytes()).hexdigest()
    config = measurement_config()
    run_id = uuid.uuid4().hex
    runtime = {'pid': os.getpid(), 'run_id': run_id, 'started_at': utc(), 'num_shards': len(args.nodes),
               'tracks_sha256': track_hash, 'tracks': len(tracks), 'source_directory': str(source),
               'source_hashes': hashes}
    write_json(existing_runtime, runtime)
    write_json(root / 'watch-failures.json', [])
    (root / 'watch-complete.json').unlink(missing_ok=True)
    children, failures = [], []
    lease = root / f'lease-{run_id}.json'
    write_json(lease, {'run_id': run_id, 'time': time.time()})
    stopped, handlers = [], {}
    for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        handlers[number] = signal.signal(number, lambda number, frame: stopped.append(number))
    try:
        local_nodes = set(subprocess.check_output(['hostname', '-I'], text=True).split()) | {'localhost', '127.0.0.1'}
        with ThreadPoolExecutor(max_workers=len(args.nodes)) as pool:
            nodes = list(pool.map(lambda node: provision(node, args, source, config, local_nodes), args.nodes))
        logs = root / 'distributed-logs'
        logs.mkdir(exist_ok=True)
        write_json(root / 'watch-code-inputs.json', config)
        for shard, node in enumerate(args.nodes):
            report_dir = shard_report_dir(root, shard, len(args.nodes))
            argv = [args.python, '-u', source / 'measure_watcher.py', '--tracks', args.tracks,
                    '--data-root', args.data_root, '--output-dir', root, '--report-dir', report_dir,
                    '--workers', args.workers, '--shard', shard, '--num-shards', len(args.nodes), '--run-id', run_id]
            for receipt in args.receipts:
                argv.extend(['--receipts', receipt])
            argv = list(map(str, argv))
            wrapped = ['env', *CPU_ENV, args.python, '-u', source / 'owned_subprocess.py',
                       '--lease', lease, '--run-id', run_id, '--', *argv]
            wrapped = list(map(str, wrapped))
            command = wrapped if node in local_nodes else remote_command(node, wrapped)
            log_path = logs / f'measure-{shard:05d}-{node}.log'
            process, output = start_logged(command, log_path, stdin=subprocess.PIPE)
            children.append((node, process, output, report_dir))
            nodes[shard].update(shard=shard, process_pid=process.pid, log=str(log_path))
        runtime.update(nodes=nodes)
        write_json(existing_runtime, runtime)
        print(json.dumps({'stage': 'distributed-measurements', **runtime}), flush=True)
        started, last_print = time.monotonic(), 0
        while True:
            if stopped:
                raise RuntimeError('Owned distributed measurement run was stopped')
            write_json(lease, {'run_id': run_id, 'time': time.time()})
            totals = dict(completed=0, active=0, waiting_for_download=0, failed=0, verified_source_files=0)
            failures, returns = [], {}
            for shard, (node, process, _, report_dir) in enumerate(children):
                identity = read_current(report_dir / 'watch-runtime.json', run_id)
                status = read_current(report_dir / 'watch-status.json', run_id)
                if identity and (identity['tracks_sha256'] != track_hash or identity['shard'] != shard or
                                 identity['num_shards'] != len(args.nodes)):
                    raise ValueError(f'Incorrect watcher assignment on {node}')
                if status:
                    for key in ('completed', 'active', 'waiting_for_download', 'failed'):
                        totals[key] += status[key]
                    totals['verified_source_files'] = max(totals['verified_source_files'], status['verified_source_files'])
                else:
                    totals['waiting_for_download'] += len(tracks[shard::len(args.nodes)])
                if identity:
                    path = report_dir / 'watch-failures.json'
                    if path.is_file():
                        failures.extend({**failure, 'node': node, 'shard': shard} for failure in json.loads(path.read_text()))
                returns[node] = process.poll()
                if returns[node] not in (None, 0):
                    failures.append({'node': node, 'shard': shard, 'error': f'Watcher exited {returns[node]}',
                                     'log': nodes[shard]['log']})
            status = {**totals, 'failed': max(totals['failed'], len(failures)), 'run_id': run_id,
                      'num_shards': len(args.nodes), 'tracks': len(tracks), 'tracks_sha256': track_hash,
                      'seconds': round(time.monotonic() - started), 'returncodes': returns, 'updated_at': utc()}
            write_json(root / 'watch-status.json', status)
            write_json(root / 'watch-failures.json', failures)
            if time.monotonic() - last_print >= 30:
                print(json.dumps(status), flush=True)
                last_print = time.monotonic()
            if all(code is not None for code in returns.values()):
                break
            time.sleep(5)
        if failures:
            raise RuntimeError('Distributed measurement failures; inspect watch-failures.json and node logs')
        if config != measurement_config() or track_hash != hashlib.sha256(args.tracks.read_bytes()).hexdigest():
            raise ValueError('Measurement implementation or track plan changed during this run')
        for shard, (node, _, _, report_dir) in enumerate(children):
            report = read_current(report_dir / 'watch-complete.json', run_id)
            if (not report or report.get('complete') is not True or report.get('tracks_sha256') != track_hash or
                    report.get('tracks') != len(tracks[shard::len(args.nodes)]) or report.get('shard') != shard or
                    report.get('num_shards') != len(args.nodes)):
                raise ValueError(f'Incomplete or mismatched completion report on {node}')
            if json.loads((report_dir / 'watch-code-inputs.json').read_text()) != config:
                raise ValueError(f'Measurement code changed on {node}')
        invalid = [track for track in tracks if not checkpoint_valid(track, args.data_root, root, config)]
        if invalid:
            raise ValueError(f'{len(invalid)} measurement checkpoints failed final verification: {invalid[:5]}')
        complete = {'complete': True, 'tracks': len(tracks), 'num_shards': len(args.nodes),
                    'run_id': run_id, 'tracks_sha256': track_hash, 'finished_at': utc()}
        write_json(root / 'watch-complete.json', complete)
        print(json.dumps(complete), flush=True)
        return 0
    except BaseException as exc:
        failures.append({'error': f'{type(exc).__name__}: {exc}', 'run_id': run_id})
        write_json(root / 'watch-failures.json', failures)
        raise
    finally:
        write_json(lease, {'run_id': run_id, 'time': 0})
        for _, process, _, _ in children:
            if process.stdin:
                process.stdin.close()
        deadline = time.monotonic() + 15
        while any(process.poll() is None for _, process, _, _ in children) and time.monotonic() < deadline:
            time.sleep(.1)
        for _, process, output, _ in children:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
            process.wait()
            output.join()
        for number, handler in handlers.items():
            signal.signal(number, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tracks', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--receipts', type=Path, action='append', required=True)
    parser.add_argument('--nodes', nargs='+', default=list(DEFAULT_NODES))
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--workers', type=int, default=32)
    args = parser.parse_args()
    if args.workers < 1 or len(set(args.nodes)) != len(args.nodes) or any(node.startswith('-') for node in args.nodes):
        parser.error('workers must be positive and nodes must be unique valid SSH hosts')
    args.output_dir, args.data_root, args.tracks = args.output_dir.resolve(), args.data_root.resolve(), args.tracks.resolve()
    args.receipts = [path.resolve() for path in args.receipts]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'distributed-measurements.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
