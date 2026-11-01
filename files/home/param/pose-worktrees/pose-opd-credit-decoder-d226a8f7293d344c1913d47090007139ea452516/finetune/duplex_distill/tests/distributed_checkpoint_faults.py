"""Bounded two-rank CPU fault injection against the actual checkpoint writer.

Each fault runs in a fresh Gloo group with a four-second timeout. Workers write
results independently and remain alive until their peer reports, so an early
rank exit cannot hide a mismatched-collective timeout. No GPUs are used.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest import mock

import torch
import torch.distributed as dist
from torch import nn


def load_module(path):
    spec = importlib.util.spec_from_file_location('checkpoint_fault_target', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def worker(args):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', timeout=datetime.timedelta(seconds=4))
    rank = dist.get_rank()
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor
    mesh = init_device_mesh('cpu', (2,))
    module = load_module(args.module)
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(distribute_tensor(torch.ones(7, 3), mesh, [Shard(0)]))
            self.replica = nn.Parameter(torch.ones(2))
    model = Model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, foreach=False)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    root = args.output / args.case
    root.mkdir(parents=True, exist_ok=True)
    cfg = {'checkpoint_reserve_gb': 0, 'keep_latest_optimizer_state': True}
    old = module.save_checkpoint(model, {'main': optimizer}, root / 'run', 1, cfg)
    injected = {'hit': False}
    def fail():
        injected['hit'] = True
        raise OSError(f'injected {args.case} on rank{rank}')
    def selected(path):
        return 'step-000002' in str(path)
    with ExitStack() as patches:
        if args.case in ('state', 'size') and rank == 1:
            attribute = '_state' if args.case == 'state' else '_tensor_bytes'
            patches.enter_context(mock.patch.object(module, attribute, side_effect=lambda *_a, **_kw: fail()))
        if args.case == 'mkdir' and rank == 0:
            original = Path.mkdir
            def mkdir(path, *a, **kw):
                if selected(path) and path.name == 'resume':
                    fail()
                return original(path, *a, **kw)
            patches.enter_context(mock.patch.object(Path, 'mkdir', mkdir))
        if args.case == 'write' and rank == 1:
            original = module._save_file
            def write(obj, path):
                if selected(path):
                    fail()
                return original(obj, path)
            patches.enter_context(mock.patch.object(module, '_save_file', write))
        if args.case in ('write_hash', 'verify_hash') and rank == 1:
            original = module._sha256
            def sha(path):
                path = Path(path)
                if (selected(path) and path.name == 'rank1.pt'
                        and (('.tmp' in str(path)) == (args.case == 'write_hash'))):
                    fail()
                return original(path)
            patches.enter_context(mock.patch.object(module, '_sha256', sha))
        if args.case == 'metadata' and rank == 0:
            original = module._atomic_json
            def metadata(path, value):
                if selected(path):
                    fail()
                return original(path, value)
            patches.enter_context(mock.patch.object(module, '_atomic_json', metadata))
        if args.case == 'rename' and rank == 0:
            original = os.rename
            def rename(src, dst):
                if selected(src):
                    fail()
                return original(src, dst)
            patches.enter_context(mock.patch.object(os, 'rename', rename))
        if args.case == 'latest' and rank == 0:
            patches.enter_context(mock.patch.object(module, '_write_latest_marker', side_effect=lambda *_a: fail()))
        if args.case == 'prune' and rank == 0:
            patches.enter_context(mock.patch.object(module, '_remove_flat_resume', side_effect=lambda *_a: fail()))
        if args.case == 'prune_enumeration' and rank == 0:
            original = Path.iterdir
            def iterdir(path):
                if path == root / 'run/checkpoints':
                    fail()
                return original(path)
            patches.enter_context(mock.patch.object(Path, 'iterdir', iterdir))
        started = time.monotonic()
        try:
            saved = module.save_checkpoint(model, {'main': optimizer}, root / 'run', 2, cfg)
            outcome = {'returned': True, 'retention': saved['retention']}
        except Exception as exc:
            outcome = {'returned': False, 'exception_type': type(exc).__name__, 'message': str(exc)}
        outcome.update(rank=rank, seconds=time.monotonic() - started, fault_hit=injected['hit'],
                       old_resume_preserved=Path(old['resume_path']).is_dir())
    if hasattr(module, 'CheckpointSaveError'):
        # A synchronized error is only recoverable if the group still supports
        # the next training collective and a fresh same-step checkpoint attempt.
        check = torch.tensor(1)
        dist.all_reduce(check)
        assert int(check) == 2
        outcome['next_collective_passed'] = True
        if outcome.get('exception_type') == 'CheckpointSaveError':
            retry = module.save_checkpoint(model, {'main': optimizer}, root / 'run', 2, cfg, attempt=1)
            module.resume_checkpoint(model, {'main': optimizer}, retry['resume_path'], 'cpu')
            outcome['retry_resumed'] = True
    (root / f'rank{rank}.json').write_text(json.dumps(outcome, indent=2) + '\n')
    until = time.monotonic() + 6
    while time.monotonic() < until and not (root / f'rank{1-rank}.json').exists():
        time.sleep(.02)
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--module', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', default=['state', 'size', 'mkdir', 'write', 'write_hash',
                        'verify_hash', 'metadata', 'rename', 'latest', 'prune', 'prune_enumeration'])
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--case')
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'module': str(args.module), 'source_sha256': hashlib.sha256(args.module.read_bytes()).hexdigest(),
              'transport': 'CPU/Gloo, two ranks, four-second collective timeout', 'cases': []}
    for case in args.cases:
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
                   '--nproc-per-node=2', str(Path(__file__).resolve()), '--worker', '--case', case,
                   '--module', str(args.module), '--output', str(args.output)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        try:
            run = subprocess.run(command, env=env, text=True, capture_output=True, timeout=22)
            (args.output / f'{case}.log').write_text(run.stdout + run.stderr)
            records = [json.loads((args.output / case / f'rank{rank}.json').read_text()) for rank in range(2)]
            transport_error = any(any(word in record.get('message', '').lower()
                                      for word in ('timed out', 'timeout', 'connection closed', 'connection reset')) for record in records)
            consistent = (len({record['returned'] for record in records}) == 1
                          and (records[0]['returned'] or len({record.get('message') for record in records}) == 1))
            item = {'case': case, 'exit_code': run.returncode, 'records': records,
                    'consistent': consistent, 'transport_error': transport_error,
                    'pass': run.returncode == 0 and consistent and not transport_error
                            and any(record['fault_hit'] for record in records)
                            and all(record['old_resume_preserved'] for record in records)}
        except Exception as exc:
            item = {'case': case, 'pass': False, 'harness_error': repr(exc)}
        report['cases'].append(item)
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'case': case, 'pass': item['pass'], 'transport_error': item.get('transport_error')}), flush=True)


if __name__ == '__main__':
    main()
