"""Collect, decode and verify one authorized second cohort without rendering."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np

TOOLS = Path('/home/param/pose-repo/tools/pose_jump_audit')
PYTHON = '/home/param/pose-repo/.venv/bin/python'
CHECKER = '771b696b6b3cdb3509b245bc5909d5552321ca22d79e49633d668b5b60efdd2e'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def cross_cohort(first, second):
    draws, poses = [], []
    for root, seed_base in ((first, 8018011), (second, 8018027)):
        cohort_draws, cohort_poses = set(), set()
        for index in range(1000):
            run = root/'runs'/f'rollout-{index:06d}'
            summary = read(run/'summary.json')
            execution = summary['execution']
            draw = (execution['batch_seed'], execution['engine_row'])
            assert draw == (seed_base+index//64, index%64), (run, draw)
            assert draw not in cohort_draws
            cohort_draws.add(draw)
            with np.load(run/'tokens.npz', allow_pickle=False) as saved:
                cohort_poses.add(hashlib.sha256(saved['pose'][66:].tobytes()).hexdigest())
        assert len(cohort_draws) == len(cohort_poses) == 1000
        draws.append(cohort_draws);poses.append(cohort_poses)
    assert not draws[0]&draws[1] and not poses[0]&poses[1]
    value = {'state': 'passed', 'at': time.time(), 'first_root': str(first), 'second_root': str(second),
             'samples_per_cohort': 1000, 'combined_unique_random_draws': len(draws[0]|draws[1]),
             'combined_unique_generated_pose_sequences': len(poses[0]|poses[1]),
             'overlapping_draws': 0, 'overlapping_generated_pose_sequences': 0,
             'first_batch_seed_range': [8018011, 8018026], 'second_batch_seed_range': [8018027, 8018042]}
    write(second/'cross-cohort-verification.json', value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--first-root', type=Path, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--collect-host', default='')
    parser.add_argument('--controls', type=Path, required=True)
    args = parser.parse_args()
    state = {'started': time.time(), 'pid': os.getpid(), 'root': str(args.root), 'step': args.step}

    def stage(name):
        state.update(stage=name, updated=time.time())
        write(args.root/'cohort2-pipeline-status.json', state)
        print(json.dumps(state), flush=True)

    def run(arguments):
        subprocess.run([PYTHON, *map(str, arguments)], check=True)

    try:
        assert hashlib.sha256((TOOLS/'strict_detection_r3.py').read_bytes()).hexdigest() == CHECKER
        stage('collecting_and_decoding_cpu')
        command = [TOOLS/'process_rollouts.py', '--root', args.root, '--workers', '8', '--watch', '--collect-hosts']
        if args.collect_host:
            command.append(args.collect_host)
        run(command)
        stage('verifying_generation_arrays')
        run([TOOLS/'verify_generation.py', '--root', args.root, '--expected-step', args.step, '--seed-base', '8018027'])
        cross_cohort(args.first_root, args.root)
        stage('strict_r3_full_scan')
        run([TOOLS/'strict_detection_r3.py', '--root', args.root,
             '--output', args.root/'strict-review-v5-r3.json', '--controls', args.controls])
        result = read(args.root/'strict-review-v5-r3.json')
        assert result['analyzed_rollouts'] == 1000 and result['checker_sha256'] == CHECKER
        state.update(candidate_events=result['candidate_events'], candidate_rollouts=result['candidate_rollouts'],
                     analyzed_rollouts=1000, finished=time.time())
        stage('complete')
    except Exception as error:
        state.update(error=str(error), traceback=traceback.format_exc(), finished=time.time())
        stage('failed')
        raise


if __name__ == '__main__':
    main()
