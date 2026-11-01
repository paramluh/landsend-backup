#!/usr/bin/env python3
"""Foreground supervision of disjoint reference encoder shards on allocated GPUs.

Each SSH wrapper owns its encoder process groups. Disconnect, shared abort,
expired controller lease, or one failed shard stops only this run's children.
Existing verified per-group encoder receipts provide restart/resume support.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import select
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid

import pandas as pd

from distributed_scan import DEFAULT_NODES, snapshot_code
from encode_bank import (ENCODER_ROOT, VAE, VAE_SHA256, encoder_fingerprint,
                         encoding_checkpoint_matches, selection_groups, sha256)
from hf_sources import write_json
from process_output import CPU_ENV, start_logged


def now():
    return datetime.now(timezone.utc).isoformat()


def assignments(nodes, gpus):
    if not nodes or len(set(nodes)) != len(nodes) or gpus < 1:
        raise ValueError('Unique nodes and a positive GPU count are required')
    return [{'node': node, 'node_rank': rank, 'gpu': gpu, 'shard': rank * gpus + gpu,
             'shards': len(nodes) * gpus} for rank, node in enumerate(nodes) for gpu in range(gpus)]


def ssh_command(node, argv):
    if not node or node.startswith('-') or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-' for c in node):
        raise ValueError(f'Invalid SSH node: {node}')
    return ['ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=10',
            '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=2', node, shlex.join(list(map(str, argv)))]


def ssh_environment():
    environment = dict(os.environ)
    environment.pop('SSH_AUTH_SOCK', None)
    return environment


def gpu_memory(gpus):
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                                      '--format=csv,noheader,nounits'], text=True, timeout=20)
    values = {int(index): int(memory) for index, memory in (line.split(',') for line in output.strip().splitlines())}
    if any(gpu not in values for gpu in range(gpus)):
        raise ValueError('Requested encoder GPUs are not present')
    return {gpu: values[gpu] for gpu in range(gpus)}


def verify_snapshot(snapshot):
    hashes = json.loads((Path(snapshot) / 'source-hashes.json').read_text())
    if any(sha256(Path(snapshot) / name) != digest for name, digest in hashes.items()):
        raise ValueError('Immutable encoder code snapshot changed')


def preflight(args):
    verify_snapshot(Path(__file__).resolve().parent)
    vae_hash = sha256(args.vae) if args.vae.is_file() and os.access(args.vae, os.R_OK) else None
    return {'hostname': socket.gethostname(), 'selection_sha256': sha256(args.selection),
            'encoder_fingerprint': encoder_fingerprint(args.encoder_root),
            'vae_path': str(args.vae), 'vae_sha256': vae_hash,
            'gpu_memory_mib': gpu_memory(args.gpus_per_node)}


def check_preflights(reports, selection_hash, fingerprint, *, require_vae=True):
    for report in reports:
        if report['selection_sha256'] != selection_hash or report['encoder_fingerprint'] != fingerprint:
            raise ValueError(f'Node {report["node"]} has different selection or encoder implementation/runtime')
        if require_vae and report['vae_sha256'] != VAE_SHA256:
            raise ValueError(f'Node {report["node"]} cannot read the pinned VAE')


def remote_preflight(node, args, snapshot, vae):
    argv = ['env', *CPU_ENV, args.python, '-u', snapshot / 'distributed_encode.py', 'preflight', '--selection', args.selection,
            '--vae', vae, '--encoder-root', args.encoder_root, '--gpus-per-node', args.gpus_per_node]
    result = subprocess.run(ssh_command(node, argv), env=ssh_environment(), stdout=subprocess.PIPE,
                            text=True, timeout=180)
    print(result.stdout, end='', flush=True)
    if result.returncode:
        raise RuntimeError(f'{node} encoder preflight failed ({result.returncode})')
    return {'node': node, **json.loads(result.stdout.strip().splitlines()[-1])}


def shared_vae(source, assets):
    destination = Path(assets) / 'Wan2.2_VAE.pth'
    if destination.is_file() and sha256(destination) == VAE_SHA256:
        return destination
    if sha256(source) != VAE_SHA256:
        raise ValueError('Local VAE does not match the documented checkpoint')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        shutil.copyfile(source, temporary)
        if sha256(temporary) != VAE_SHA256:
            raise ValueError('Shared VAE copy failed checksum verification')
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def stop_children(children):
    # PIDs/PGIDs come only from Popen(start_new_session=True) in this wrapper.
    for process, _ in children:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 10
    while any(process.poll() is None for process, _ in children) and time.monotonic() < deadline:
        time.sleep(.1)
    for process, output in children:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait()
        output.join()


def node_worker(args):
    manifest = json.loads(args.manifest.read_text())
    run_dir = args.manifest.parent
    verify_snapshot(manifest['snapshot'])
    fingerprint = encoder_fingerprint(manifest['encoder_root'])
    if fingerprint != manifest['encoder_fingerprint'] or sha256(manifest['selection']) != manifest['selection_sha256']:
        raise ValueError('Node input identity changed after preflight')
    report = manifest['nodes'][args.node_rank]
    if sha256(report['vae_path']) != VAE_SHA256:
        raise ValueError('Pinned VAE changed after preflight')
    jobs = [job for job in manifest['assignments'] if job['node_rank'] == args.node_rank]
    if any(value >= 2048 for value in gpu_memory(len(jobs)).values()):
        raise RuntimeError('GPU became busy after idle preflight; restart when idle')
    stopped = []
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda number, frame: stopped.append(number))
    children = []
    receipt = {'run_id': manifest['run_id'], 'node_rank': args.node_rank, 'node': report['node'],
               'wrapper_pid': os.getpid(), 'selection_sha256': manifest['selection_sha256'],
               'encoder_fingerprint': fingerprint, 'shards': [], 'complete': False}
    receipt_path = run_dir / f'node-{args.node_rank:02d}.json'
    try:
        for job in jobs:
            argv = [manifest['python'], '-u', str(Path(manifest['snapshot']) / 'encode_bank.py'),
                    '--selection', manifest['selection'], '--source-root', manifest['source_root'],
                    '--output-dir', manifest['output_dir'], '--vae', report['vae_path'],
                    '--encoder-root', manifest['encoder_root'], '--gpu', str(job['gpu']),
                    '--shards', str(job['shards']), '--shard', str(job['shard']),
                    '--batch-size', str(manifest['batch_size'])]
            log_path = run_dir / f'node-{args.node_rank:02d}-gpu-{job["gpu"]:02d}.log'
            process, output = start_logged(argv, log_path, start_new_session=True)
            children.append((process, output))
            receipt['shards'].append({**job, 'pid': process.pid, 'log': str(log_path), 'returncode': None})
            write_json(receipt_path, receipt)
        while True:
            if stopped or (run_dir / 'abort.json').exists():
                raise RuntimeError('Owned encoder run was stopped')
            lease = json.loads((run_dir / 'lease.json').read_text())
            if lease['run_id'] != manifest['run_id'] or time.time() - lease['time'] > 90:
                raise RuntimeError('Encoder controller lease expired')
            if select.select([sys.stdin], [], [], 0)[0] and not os.read(sys.stdin.fileno(), 1):
                raise RuntimeError('Encoder controller disconnected')
            returns = [process.poll() for process, _ in children]
            for item, code in zip(receipt['shards'], returns):
                item['returncode'] = code
            write_json(receipt_path, receipt)
            if any(code not in (None, 0) for code in returns):
                raise RuntimeError(f'Encoder shard failed: {returns}')
            if all(code == 0 for code in returns):
                break
            time.sleep(1)
        receipt.update(complete=True, finished_at=now())
    except BaseException as error:
        receipt['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        stop_children(children)
        write_json(receipt_path, receipt)


def aggregate(manifest, run_dir):
    expected = {job['shard']: job for job in manifest['assignments']}
    seen = set()
    for rank in range(len(manifest['nodes'])):
        report = json.loads((run_dir / f'node-{rank:02d}.json').read_text())
        if (report.get('run_id') != manifest['run_id'] or report.get('node_rank') != rank
                or report.get('node') != manifest['nodes'][rank]['node'] or report.get('complete') is not True
                or report.get('selection_sha256') != manifest['selection_sha256']
                or report.get('encoder_fingerprint') != manifest['encoder_fingerprint']):
            raise ValueError(f'Invalid encoder node receipt: {rank}')
        for shard in report['shards']:
            number = shard['shard']
            if (number in seen or number not in expected or shard.get('returncode') != 0
                    or expected[number]['node_rank'] != rank
                    or any(shard.get(key) != value for key, value in expected[number].items())):
                raise ValueError(f'Invalid encoder shard receipt: {number}')
            seen.add(number)
    if seen != expected.keys():
        raise ValueError('Encoder shard coverage is incomplete')
    if sha256(manifest['selection']) != manifest['selection_sha256']:
        raise ValueError('Selection changed during distributed encoding')
    groups = selection_groups(pd.read_parquet(manifest['selection']))
    root = Path(manifest['output_dir'])
    for bank, actor, prop, rows in groups:
        receipt = root / 'encoding-checkpoints' / f'{bank}__{actor}__{prop}.json'
        destination = root / bank / actor / f'property_{prop}/latents.safetensors'
        if not encoding_checkpoint_matches(receipt, destination, manifest['selection_sha256'],
                                           manifest['encoder_fingerprint'], len(rows)):
            raise ValueError(f'Invalid encoded group checkpoint: {bank}/{actor}/{prop}')
    return {'complete': True, 'shards': len(seen), 'groups': len(groups),
            'selection_sha256': manifest['selection_sha256'], 'encoder_fingerprint': manifest['encoder_fingerprint']}


def run(args):
    owner_pid = os.getppid()
    plan = assignments(args.nodes, args.gpus_per_node)
    snapshot, hashes = snapshot_code(Path(__file__).resolve().parent, args.output_dir)
    fingerprint, selection_hash = encoder_fingerprint(args.encoder_root), sha256(args.selection)
    state_path = args.output_dir / 'distributed-encode.json'
    write_json(state_path, {'complete': False, 'stage': 'preflight', 'supervisor_pid': os.getpid(),
                            'selection_sha256': selection_hash, 'encoder_fingerprint': fingerprint})
    paths = {node: args.vae for node in args.nodes}
    with ThreadPoolExecutor(max_workers=len(args.nodes)) as pool:
        reports = list(pool.map(lambda node: remote_preflight(node, args, snapshot, paths[node]), args.nodes))
    check_preflights(reports, selection_hash, fingerprint, require_vae=False)
    if any(report['vae_sha256'] != VAE_SHA256 for report in reports):
        fallback = shared_vae(args.vae, args.output_dir.parent / 'assets')
        for report in reports:
            if report['vae_sha256'] != VAE_SHA256:
                paths[report['node']] = fallback
        with ThreadPoolExecutor(max_workers=len(args.nodes)) as pool:
            reports = list(pool.map(lambda node: remote_preflight(node, args, snapshot, paths[node]), args.nodes))
    check_preflights(reports, selection_hash, fingerprint)
    while any(value >= 2048 for report in reports for value in report['gpu_memory_mib'].values()):
        if os.getppid() != owner_pid:
            raise RuntimeError('Parent rebuild controller exited')
        print(json.dumps({'stage': 'waiting-for-idle-encoder-gpus', 'nodes': [
            {'node': report['node'], 'gpu_memory_mib': report['gpu_memory_mib']} for report in reports]}), flush=True)
        time.sleep(30)
        with ThreadPoolExecutor(max_workers=len(args.nodes)) as pool:
            reports = list(pool.map(lambda node: remote_preflight(node, args, snapshot, paths[node]), args.nodes))
        check_preflights(reports, selection_hash, fingerprint)
    if os.getppid() != owner_pid:
        raise RuntimeError('Parent rebuild controller exited')
    run_id = uuid.uuid4().hex
    run_dir = args.output_dir / 'distributed-encode' / run_id
    run_dir.mkdir(parents=True)
    manifest = {'run_id': run_id, 'started_at': now(), 'supervisor_pid': os.getpid(), 'snapshot': str(snapshot),
                'source_hashes': hashes, 'selection': str(args.selection), 'selection_sha256': selection_hash,
                'source_root': str(args.source_root), 'output_dir': str(args.output_dir),
                'encoder_root': str(args.encoder_root), 'encoder_fingerprint': fingerprint,
                'python': args.python, 'batch_size': args.batch_size, 'assignments': plan, 'nodes': reports}
    manifest_path = run_dir / 'manifest.json'
    write_json(manifest_path, manifest)
    state = {'run_id': run_id, 'run_dir': str(run_dir), 'complete': False}
    children = []
    def interrupted(number, frame):
        raise KeyboardInterrupt(f'Signal {number}')

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupted)
    try:
        write_json(run_dir / 'lease.json', {'run_id': run_id, 'time': time.time()})
        for rank, node in enumerate(args.nodes):
            argv = ['env', *CPU_ENV, args.python, '-u', snapshot / 'distributed_encode.py', 'node', '--manifest', manifest_path,
                    '--node-rank', rank]
            log_path = run_dir / f'node-{rank:02d}.log'
            process, output = start_logged(ssh_command(node, argv), log_path, env=ssh_environment(),
                                           stdin=subprocess.PIPE)
            children.append((process, output))
        print(json.dumps({'stage': 'distributed-encoding', 'run_dir': str(run_dir), 'shards': len(plan)}), flush=True)
        while True:
            if os.getppid() != owner_pid:
                raise RuntimeError('Parent rebuild controller exited')
            returns = [process.poll() for process, _ in children]
            state.update(returncodes=returns, updated_at=now())
            write_json(state_path, state)
            write_json(run_dir / 'lease.json', {'run_id': run_id, 'time': time.time()})
            if any(code not in (None, 0) for code in returns):
                raise RuntimeError(f'Distributed encoder failed; inspect {run_dir}: {returns}')
            if all(code == 0 for code in returns):
                break
            time.sleep(2)
        verify_snapshot(snapshot)
        state.update(aggregate(manifest, run_dir), finished_at=now())
        write_json(state_path, state)
        print(json.dumps(state), flush=True)
    except BaseException as error:
        state.update(error=f'{type(error).__name__}: {error}', complete=False)
        write_json(run_dir / 'abort.json', {'run_id': run_id, 'error': state['error']})
        write_json(state_path, state)
        raise
    finally:
        for process, _ in children:
            if process.stdin:
                process.stdin.close()
        deadline = time.monotonic() + 20
        while any(process.poll() is None for process, _ in children) and time.monotonic() < deadline:
            time.sleep(.2)
        for process, output in children:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
            process.wait()
            output.join()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    launch = commands.add_parser('run')
    probe = commands.add_parser('preflight')
    worker = commands.add_parser('node')
    for command in (launch, probe):
        command.add_argument('--selection', type=Path, required=True)
        command.add_argument('--vae', type=Path, default=VAE)
        command.add_argument('--encoder-root', type=Path, default=ENCODER_ROOT)
        command.add_argument('--gpus-per-node', type=int, default=8)
    launch.add_argument('--source-root', type=Path, required=True)
    launch.add_argument('--output-dir', type=Path, required=True)
    launch.add_argument('--nodes', nargs='+', default=list(DEFAULT_NODES))
    launch.add_argument('--python', default=sys.executable)
    launch.add_argument('--batch-size', type=int, default=64)
    worker.add_argument('--manifest', type=Path, required=True)
    worker.add_argument('--node-rank', type=int, required=True)
    args = parser.parse_args()
    if args.command == 'preflight':
        print(json.dumps(preflight(args)), flush=True)
    elif args.command == 'node':
        node_worker(args)
    else:
        if args.batch_size < 1:
            parser.error('--batch-size must be positive')
        for name in ('selection', 'source_root', 'output_dir', 'vae', 'encoder_root'):
            setattr(args, name, getattr(args, name).resolve())
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_dir / 'distributed-encode.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run(args)


if __name__ == '__main__':
    main()
