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


ROOT = Path(__file__).resolve().parents[1]
SCHEDULE_KEYS = ('band_steps', 'resolve_steps', 'history_t', 't_start', 'stab_t_min', 'stab_t_max',
                 'res_t_min', 'res_t_max', 'fm_weight_stab', 'fm_weight_res')
SOURCE_FILES = ('learn_dmd_bimodal.py', 'distillation/bimodal_net.py', 'distillation/bimodal_dmd.py',
                'distillation/lora.py', 'distillation/checkpoints.py')


def parse_value(value, previous=None):
    try:
        return tomllib.loads(f'value = {value}')['value']
    except tomllib.TOMLDecodeError:
        return value.split(',') if ',' in value or isinstance(previous, list) else value


def override(config, key, value):
    *parents, leaf = key.split('.')
    for parent in parents:
        config = config.setdefault(parent, {})
    config[leaf] = value


def quantile(values, q):
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lo, hi = int(index), min(int(index) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def step_record(line):
    start = line.find('{')
    if start < 0:
        return None
    try:
        row = json.loads(line[start:])
    except json.JSONDecodeError:
        return None
    return row if 'seconds/wall' in row and row.get('outer_step', 0) > 0 else None


def read_steps(log):
    return [row for line in Path(log).read_text().splitlines() if (row := step_record(line)) is not None]


def summarize(records, skip, config, world_size, project_steps):
    measured = [row for row in records[skip:] if not row.get('profiled', False)]
    if not measured:
        raise ValueError(f'Only {len(records)} training steps found; --skip is {skip}')
    walls = [row['seconds/wall'] for row in measured]
    median = statistics.median(walls)
    phases = sorted({key for row in measured for key in row if key.startswith('seconds/')})
    result = {
        'measured_steps': len(measured), 'skipped_steps': skip,
        'profiled_steps_excluded': sum(bool(row.get('profiled', False)) for row in records[skip:]),
        'seconds_per_step': {'median': median, 'mean': statistics.mean(walls), 'p10': quantile(walls, 0.1),
                             'p90': quantile(walls, 0.9), 'min': min(walls), 'max': max(walls)},
        'phase_seconds_median': {key: statistics.median([row[key] for row in measured if key in row]) for key in phases},
        'peak_allocated_gib': max(row['memory/peak_allocated_gib'] for row in measured),
        'student_rows_per_step': world_size * len(config['student_kinds']),
        'critic_rows_per_update': world_size * config['critic_rows'],
        'critic_updates_per_step': config['critic_updates'],
        'student_rows_per_second': world_size * len(config['student_kinds']) / median,
        'projected_student_steps': project_steps,
        'projected_training_seconds': median * project_steps,
        'step_seconds': walls,
    }
    cycles = [row['seconds/observed_cycle'] for row in measured if 'seconds/observed_cycle' in row]
    if cycles:
        result['observed_cycle_seconds'] = {'median': statistics.median(cycles), 'mean': statistics.mean(cycles),
                                           'p10': quantile(cycles, 0.1), 'p90': quantile(cycles, 0.9)}
        result['projected_training_with_reporting_seconds'] = statistics.median(cycles) * project_steps
    return result


def compare(result, reference, target):
    a, b = result['config'], reference['config']
    mismatches = [key for key in SCHEDULE_KEYS if a[key] != b[key]]
    if result['world_size'] != reference['world_size']:
        mismatches.append('world_size')
    if a['model'] != b['model']:
        mismatches.append('model')
    if mismatches:
        raise ValueError(f'Cannot compare different rollout/objective/hardware settings: {mismatches}')
    ratio = reference['seconds_per_step']['median'] / result['seconds_per_step']['median']
    result['comparison'] = {'reference': reference['name'], 'speedup': ratio, 'target_speedup': target,
                            'meets_target': ratio >= target,
                            'student_batch_ratio': result['student_rows_per_step'] / reference['student_rows_per_step'],
                            'critic_batch_ratio': result['critic_rows_per_update'] / reference['critic_rows_per_update']}
    if 'observed_cycle_seconds' in result and 'observed_cycle_seconds' in reference:
        result['comparison']['observed_cycle_speedup'] = (reference['observed_cycle_seconds']['median']
                                                         / result['observed_cycle_seconds']['median'])


def show(result):
    timing = result['seconds_per_step']
    comparison = result.get('comparison', {})
    speedup = f"; {comparison['speedup']:.2f}x baseline" if comparison else ''
    print(f"{result['name']}: {timing['median']:.3f} s/step "
          f"(p10-p90 {timing['p10']:.3f}-{timing['p90']:.3f}), "
          f"{result['peak_allocated_gib']:.1f} GiB, "
          f"{result['projected_student_steps']} student steps "
          f"{result['projected_training_seconds'] / 60:.1f} min{speedup}", flush=True)


def load_config(path, changes):
    config = tomllib.loads(Path(path).read_text())
    for item in changes:
        key, value = item.split('=', 1)
        node = config
        for parent in key.split('.')[:-1]:
            node = node.get(parent, {})
        override(config, key, parse_value(value, node.get(key.split('.')[-1])))
    return config


def run(args):
    config = load_config(args.config, args.set)
    cases = tomllib.loads(Path(args.matrix).read_text())['cases'] if args.matrix else [{'name': args.name, 'set': {}}]
    if args.case:
        selected = set(args.case)
        if selected - {case['name'] for case in cases}:
            raise ValueError('Unknown --case name')
        cases = [case for case in cases if case['name'] in selected]
    if not 0 <= args.skip < args.steps:
        raise ValueError('Require 0 <= --skip < --steps')
    names = [case['name'] for case in cases]
    if len(set(names)) != len(names) or any(not name or Path(name).name != name for name in names):
        raise ValueError('Case names must be distinct filenames')
    world = len(args.nodes.split()) * int(os.environ.get('SLOTS', 8))
    reference = json.loads(Path(args.reference).read_text()) if args.reference else None
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for case in cases:
        effective = copy.deepcopy(config)
        changes = list(args.set)
        for key, value in case.get('set', {}).items():
            override(effective, key, value)
            changes.append(f'{key}={json.dumps(value)}')
        if effective.get('resume_from') or effective.get('critic_resume_from'):
            raise ValueError('Benchmarks start fresh; explicitly override resume_from and critic_resume_from to empty strings')
        if effective['student_kinds'] != ['stab'] * len(effective['student_kinds']) or not effective['student_kinds']:
            raise ValueError('This sweep requires stabilizer-only student updates')
        if effective['fm_weight_res'] != 0 or effective['fm_weight_stab'] != 1:
            raise ValueError('This sweep requires stabilizer-only critic updates')
        if len(effective['student_kinds']) % effective['student_micro_rows']:
            raise ValueError('student_micro_rows must divide the student row count')
        command = ['bash', str(ROOT / 'run_dmd_bimodal.sh'), args.nodes, str(Path(args.config).resolve()),
                   '--benchmark', str(args.steps), '--benchmark-skip', str(args.skip)]
        for value in changes:
            command.extend(['--set', value])
        log = output / f"{case['name']}.log"
        destination = output / f"{case['name']}.json"
        if log.exists() or destination.exists():
            raise FileExistsError(f'Existing output for {case["name"]}; choose a new output directory')
        result = {'name': case['name'], 'config': effective, 'world_size': world, 'nodes': args.nodes.split(),
                  'command': command, 'config_sha256': hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
                  'source_root': str(ROOT),
                  'source_sha256_at_launch': {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                                              for path in SOURCE_FILES}}
        print(shlex.join(command), flush=True)
        started = time.monotonic()
        records, previous = [], None
        with log.open('w') as handle:
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
        result['launch_elapsed_seconds'] = time.monotonic() - started
        result['returncode'] = status
        if status:
            out_of_memory = any(message in log.read_text(errors='replace')
                                for message in ('CUDA out of memory', 'torch.OutOfMemoryError'))
            result['out_of_memory'] = out_of_memory
            destination.write_text(json.dumps(result, indent=2) + '\n')
            if args.memory_frontier and out_of_memory:
                print(f'{case["name"]}: out of memory; trying the next lower-memory case', flush=True)
                continue
            raise SystemExit(status)
        if len(records) != args.steps:
            raise ValueError(f'Expected {args.steps} training records, found {len(records)} in {log}')
        for row in records:
            for key in ('student/loss', 'critic/loss', 'student/grad_norm', 'critic/grad_norm'):
                if key in row and not math.isfinite(row[key]):
                    raise ValueError(f'Non-finite {key} at step {row["step"]}; rejecting benchmark')
        result.update(summarize(records, args.skip, effective, world, args.project_steps))
        if reference is None:
            reference = result
        compare(result, reference, args.target_speedup)
        destination.write_text(json.dumps(result, indent=2) + '\n')
        show(result)
        if args.memory_frontier:
            print(f'First fitting case: {case["name"]}; stopping the memory frontier', flush=True)
            return
    if args.memory_frontier:
        raise SystemExit('No memory-frontier candidate fit')


def report(args):
    config = load_config(args.config, args.set)
    result = {'name': args.name, 'config': config, 'world_size': args.world_size,
              **summarize(read_steps(args.log), args.skip, config, args.world_size, args.project_steps)}
    if args.reference:
        compare(result, json.loads(Path(args.reference).read_text()), args.target_speedup)
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    show(result)


def main():
    parser = argparse.ArgumentParser(description='Fresh-process stabilizer DMD timing sweeps; all launcher output stays attached.')
    commands = parser.add_subparsers(dest='command', required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--config', required=True)
    common.add_argument('--set', action='append', default=[])
    common.add_argument('--skip', type=int, default=4)
    common.add_argument('--name', default='baseline')
    common.add_argument('--reference')
    common.add_argument('--target-speedup', type=float, default=3.0)
    common.add_argument('--project-steps', type=int, default=1050, help='student steps after the 150-round warmup')
    launch = commands.add_parser('run', parents=[common])
    launch.add_argument('--nodes', required=True)
    launch.add_argument('--output', required=True)
    launch.add_argument('--steps', type=int, default=12)
    launch.add_argument('--matrix')
    launch.add_argument('--case', action='append')
    launch.add_argument('--memory-frontier', action='store_true',
                        help='try cases in matrix order, continue only after CUDA OOM, stop at the first fitting case')
    launch.set_defaults(func=run)
    summarize_log = commands.add_parser('report', parents=[common])
    summarize_log.add_argument('--log', required=True)
    summarize_log.add_argument('--world-size', type=int, required=True)
    summarize_log.add_argument('--output')
    summarize_log.set_defaults(func=report)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
