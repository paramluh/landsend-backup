"""Five-node resolver DMD benchmarks, with explicit clip-row/target units.

Each dataset row contains four resolver targets. One student row per GPU is
therefore four targets per GPU, or 160 targets over forty GPUs. Compilation,
startup and checkpoint writing are not included in steady-step projections.
"""

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import time
import tomllib

from distillation.benchmark import load_config, override, quantile, step_record


ROOT = Path(__file__).resolve().parents[1]
TARGETS_PER_ROW = 4
SOURCE_FILES = ('learn_dmd_resolver.py', 'run_dmd_resolver.sh',
                'learn_dmd_bimodal.py', 'distillation/resolver_dmd.py',
                'distillation/resolver_data.py', 'distillation/bimodal_net.py',
                'distillation/bimodal_dmd.py', 'distillation/checkpoints.py',
                'utils/harness.py')
OBJECTIVE_KEYS = ('resolver_method', 'dataset_history', 'history_t', 't_start',
                  'res_t_min', 'res_t_max', 'normalizer_epsilon',
                  'history_dropout', 'history_dropout_source',
                  'stabilizer_checkpoint_path', 'model')


def validate_config(config):
    if config['resolver_method'] not in ('gt', 'onpolicy'):
        raise ValueError('resolver_method must be gt or onpolicy')
    if config.get('resume_from') or config.get('critic_resume_from') or config.get('lora'):
        raise ValueError('Resolver benchmarks require fresh dense student/critic models')
    for name in ('student_micro_rows', 'critic_updates', 'pool_micro_batch'):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(f'{name} must be a positive integer')
    if config['global_batch_size'] != 160:
        raise ValueError('Every student and critic optimizer update must retain global batch160')
    if config['dataset_history'] != 6:
        raise ValueError('Both methods use six real GT history frames per dataset row')
    if not config.get('shard_teacher', True):
        raise ValueError('The dense memory plan requires a sharded frozen teacher')
    if config['resolver_method'] == 'onpolicy' and not config.get('shard_stabilizer', True):
        raise ValueError('The on-policy dense memory plan requires a sharded frozen stabilizer')


def summarize(records, skip, config, world_size, project_steps):
    measured = [row for row in records[skip:] if not row.get('profiled', False)]
    if not measured:
        raise ValueError('No unprofiled steps remain after compilation/warmup exclusion')
    for row in measured:
        for key, value in row.items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(f'Nonfinite metric {key} at outer step {row.get("outer_step")}')
        if row['seconds/wall'] <= 0:
            raise ValueError('Measured step time must be positive')
    walls = [row['seconds/wall'] for row in measured]
    median = statistics.median(walls)
    if world_size % 5 or not 1 <= world_size // 5 <= 8:
        raise ValueError('Timing summary requires five homogeneous nodes, each with 1..8 ranks')
    slots = world_size // 5
    rows_per_node = config['global_batch_size'] // (TARGETS_PER_ROW * 5)
    per_gpu_targets = [TARGETS_PER_ROW * (rows_per_node // slots + (rank < rows_per_node % slots))
                       for rank in range(slots)]
    student_rows = config['global_batch_size'] // TARGETS_PER_ROW
    critic_rows = student_rows
    pool_rows = critic_rows * config['critic_updates']
    phases = sorted({key for row in measured for key in row if key.startswith('seconds/')})
    result = {
        'method': config['resolver_method'],
        'measured_steps': len(measured), 'skipped_steps': skip,
        'profiled_steps_excluded': sum(bool(row.get('profiled', False)) for row in records[skip:]),
        'seconds_per_step': {'median': median, 'mean': statistics.mean(walls),
                             'p10': quantile(walls, .1), 'p90': quantile(walls, .9),
                             'min': min(walls), 'max': max(walls)},
        'phase_seconds_median': {key: statistics.median(row[key] for row in measured if key in row)
                                 for key in phases},
        'peak_allocated_gib': max(row['memory/peak_allocated_gib'] for row in measured),
        'first_step_seconds': records[0]['seconds/wall'],
        'student_clip_rows_per_step': student_rows,
        'student_resolver_targets_per_step': TARGETS_PER_ROW * student_rows,
        'student_resolver_targets_by_local_rank': per_gpu_targets,
        'critic_clip_rows_per_update': critic_rows,
        'critic_resolver_targets_per_update': TARGETS_PER_ROW * critic_rows,
        'critic_updates_per_step': config['critic_updates'],
        'pool_clip_rows_per_step': pool_rows,
        'pool_resolver_targets_per_step': TARGETS_PER_ROW * pool_rows,
        'pool_micro_targets_per_gpu': TARGETS_PER_ROW * config['pool_micro_batch'],
        'student_resolver_targets_per_second': TARGETS_PER_ROW * student_rows / median,
        'projected_student_steps': project_steps,
        'projected_steady_training_seconds': project_steps * median,
        'projection_excludes': ['startup', 'compilation', 'critic warmup', 'checkpoint writes'],
        'step_seconds': walls,
    }
    cycles = [row['seconds/observed_cycle'] for row in measured if 'seconds/observed_cycle' in row]
    if cycles:
        result['observed_cycle_seconds'] = {'median': statistics.median(cycles),
                                            'mean': statistics.mean(cycles)}
    return result


def comparison(result, reference):
    mismatches = [key for key in OBJECTIVE_KEYS if result['config'].get(key) != reference['config'].get(key)]
    if result['world_size'] != reference['world_size']:
        mismatches.append('world_size')
    if mismatches:
        raise ValueError(f'Performance reference changes objective or hardware: {mismatches}')
    ratio = reference['seconds_per_step']['median'] / result['seconds_per_step']['median']
    return {'reference': reference['name'], 'step_speedup': ratio,
            'student_target_batch_ratio': (result['student_resolver_targets_per_step'] /
                                           reference['student_resolver_targets_per_step']),
            'critic_target_batch_ratio': (result['critic_resolver_targets_per_update'] /
                                          reference['critic_resolver_targets_per_update'])}


def show(result):
    timing = result['seconds_per_step']
    print(f"{result['name']}: {timing['median']:.3f} s/step; "
          f"{result['peak_allocated_gib']:.2f} GiB peak; "
          f"{result['student_resolver_targets_per_step']} global student targets; "
          f"{result['projected_student_steps']} steady student steps "
          f"~{result['projected_steady_training_seconds']/60:.1f} min (projection)", flush=True)


def write_json(path, result):
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(path)


def run(args):
    nodes = args.nodes.split()
    if len(set(nodes)) != len(nodes):
        raise ValueError('Repeated node names are not a distributed benchmark')
    if len(nodes) != 5:
        raise ValueError('Resolver timing requires five nodes')
    slots = int(os.environ.get('SLOTS', '8'))
    if not nodes or not 1 <= slots <= 8:
        raise ValueError('Require at least one node and 1..8 GPU slots per node')
    if not 0 <= args.skip < args.steps:
        raise ValueError('Require 0 <= --skip < --steps')
    config = load_config(args.config, args.set)
    cases = tomllib.loads(Path(args.matrix).read_text())['cases'] if args.matrix else [{'name': args.name, 'set': {}}]
    if args.case:
        chosen = set(args.case)
        if chosen - {case['name'] for case in cases}:
            raise ValueError('Unknown --case')
        cases = [case for case in cases if case['name'] in chosen]
    names = [case['name'] for case in cases]
    if len(names) != len(set(names)) or any(not name or Path(name).name != name for name in names):
        raise ValueError('Case names must be distinct filenames')
    reference = json.loads(Path(args.reference).read_text()) if args.reference else None
    output = Path(args.output).resolve()
    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=True)
    for case in cases:
        effective, changes = copy.deepcopy(config), list(args.set)
        for key, value in case.get('set', {}).items():
            override(effective, key, value)
            changes.append(f'{key}={json.dumps(value)}')
        validate_config(effective)
        command = ['bash', str(ROOT/'run_dmd_resolver.sh'), args.nodes, str(Path(args.config).resolve()),
                   '--benchmark', str(args.steps), '--benchmark-skip', str(args.skip)]
        for change in changes:
            command.extend(['--set', change])
        print(shlex.join(command), flush=True)
        if args.dry_run:
            print(json.dumps({'case': case['name'], 'nodes': nodes, 'world_size': len(nodes)*slots,
                              'global_student_targets': effective['global_batch_size'],
                              'global_critic_targets_per_update': effective['global_batch_size'],
                              'pool_micro_targets_per_gpu': TARGETS_PER_ROW*effective['pool_micro_batch']}), flush=True)
            continue
        log, destination = output/f"{case['name']}.log", output/f"{case['name']}.json"
        if log.exists() or destination.exists():
            raise FileExistsError(f'Existing artifacts for {case["name"]}; use a fresh output directory')
        result = {'name': case['name'], 'config': effective, 'world_size': len(nodes)*slots,
                  'nodes': nodes, 'slots_per_node': slots,
                  'official_five_node_timing': True,
                  'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                  'command': command, 'source_root': str(ROOT),
                  'config_sha256': hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
                  'source_sha256_at_launch': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                             for name in SOURCE_FILES}}
        records, previous = [], None
        started = time.monotonic()
        with log.open('x') as handle:
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                print(line, end='', flush=True)
                handle.write(line)
                handle.flush()
                row = step_record(line)
                if row is not None:
                    now = time.monotonic()
                    if previous is not None:
                        row['seconds/observed_cycle'] = now - previous
                    previous = now
                    records.append(row)
            status = process.wait()
        result.update(returncode=status, launch_elapsed_seconds=time.monotonic()-started)
        if status:
            result['out_of_memory'] = any(value in log.read_text(errors='replace')
                                         for value in ('CUDA out of memory', 'torch.OutOfMemoryError'))
            write_json(destination, result)
            if args.memory_frontier and result['out_of_memory']:
                print('Confirmed CUDA OOM; proceeding to the next lower-memory case', flush=True)
                continue
            raise SystemExit(status)
        if len(records) != args.steps:
            result['validation_error'] = f'Expected {args.steps} completed outer steps; found {len(records)}'
            write_json(destination, result)
            raise ValueError(result['validation_error'])
        result.update(summarize(records, args.skip, effective, len(nodes)*slots, args.project_steps))
        if reference is not None:
            result['comparison'] = comparison(result, reference)
        write_json(destination, result)
        show(result)
        if args.memory_frontier:
            print(f'First fitting candidate: {case["name"]}; no upward search', flush=True)
            return
    if args.memory_frontier and not args.dry_run:
        raise SystemExit('No memory-frontier candidate fit')


def report(args):
    config = load_config(args.config, args.set)
    validate_config(config)
    records = [row for line in Path(args.log).read_text().splitlines() if (row := step_record(line)) is not None]
    result = {'name': args.name, 'config': config, 'world_size': args.world_size,
              'source_log': str(Path(args.log).resolve()),
              'source_log_sha256': hashlib.sha256(Path(args.log).read_bytes()).hexdigest(),
              **summarize(records, args.skip, config, args.world_size, args.project_steps)}
    if args.reference:
        result['comparison'] = comparison(result, json.loads(Path(args.reference).read_text()))
    if args.output:
        write_json(Path(args.output), result)
    show(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--config', required=True)
    common.add_argument('--set', action='append', default=[])
    common.add_argument('--name', default='resolver-nocp')
    common.add_argument('--skip', type=int, default=2)
    common.add_argument('--project-steps', type=int, default=1850,
                        help='Student steps after the 150 critic-only steps; total training is 2000 outer steps')
    common.add_argument('--reference')
    launch = subparsers.add_parser('run', parents=[common])
    launch.add_argument('--nodes', required=True)
    launch.add_argument('--output', required=True)
    launch.add_argument('--steps', type=int, default=6)
    launch.add_argument('--matrix')
    launch.add_argument('--case', action='append')
    launch.add_argument('--memory-frontier', action='store_true', help='Continue only after explicit CUDA OOM; stop at first fit')
    launch.add_argument('--dry-run', action='store_true', help='Validate and print launch commands without SSH, GPUs or output creation')
    launch.set_defaults(func=run)
    summary = subparsers.add_parser('report', parents=[common])
    summary.add_argument('--log', required=True)
    summary.add_argument('--world-size', type=int, required=True)
    summary.add_argument('--output')
    summary.set_defaults(func=report)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
