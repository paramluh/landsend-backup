"""Read-only full-corpus replay of the archived studio v2 video selector.

Run after historical-video-policy.json is published. Only the requested audit
receipt is written; its destination must be outside the pipeline workspace.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from historical_video_check import ARCHIVE, oracle
from video_pipeline import task_path
from video_scan import fit_selector_proxy

FIELDS = ['track', 'frame_index', 'time_s', 'uuid', 'bank', 'cohort', 'recording_id',
          'speaker', 'session_id', 'source', 'property', 'index', 'cx0', 'cy0',
          'cw', 'ch', 'scale', 'rank_metric', 'sel_score']


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def replay(work_root, output, expected_frames=None):
    root, output = Path(work_root).resolve(), Path(output).resolve()
    if output.is_relative_to(root):
        raise ValueError('Audit output must be outside the pipeline/bank workspace')
    video = root / 'video-scan'
    policy_path = video / 'historical-video-policy.json'
    if not policy_path.is_file():
        raise ValueError('Historical selection publication receipt is required before replay')
    started = time.monotonic()
    plan = video / 'tracks.json'
    selected_path = video / 'video-selected.parquet'
    upstream_path = root / 'selection/upstream-selection.parquet'
    fit = ARCHIVE.parent.parent.parent / 'fit_score.parquet'
    inputs = {str(path): digest(path) for path in (plan, selected_path, upstream_path, fit, ARCHIVE, policy_path)}
    if inputs[str(ARCHIVE)] != 'dff8511c0d748da675c25e076a5050def6c3ec5d5b00032ab37766dcae9b2d61':
        raise ValueError('Archived oracle changed')
    tracks = json.loads(plan.read_text())
    actual = pd.read_parquet(selected_path)
    upstream = pd.read_parquet(upstream_path)
    weights = fit_selector_proxy(fit)
    actors = sorted({track['uuid'] for track in tracks})
    frames_total = chunks_total = expected_rows = 0
    differences = []
    counts, dedup = ({prop: 0 for prop in range(1, 10)} for _ in range(2))
    for number, actor in enumerate(actors, 1):
        chunks = []
        for track in (row for row in tracks if row['uuid'] == actor):
            for chunk in track['usable_chunks']:
                path = task_path(video, track['track'], chunk)
                receipt = json.loads(path.with_suffix('.complete.json').read_text())
                if digest(path) != receipt['sha256']:
                    raise ValueError(f'Candidate checksum differs: {path}')
                frame = pd.read_parquet(path)
                chunks_total += 1
                if len(frame):
                    chunks.append(frame)
        if not chunks:
            continue
        candidates = pd.concat(chunks, ignore_index=True)
        frames_total += len(candidates)
        state = oracle(candidates, upstream[upstream.uuid.eq(actor)], weights)
        expected = []
        for addition in state['new']:
            index, prop = addition['iid_frame'], addition['property']
            source = state['F'].iloc[index]
            row = {key: source[key] for key in FIELDS[:10]}
            box = state['boxes'][index] if prop <= 4 else (0, 0, 512, 960)
            row.update(property=prop, index=addition['index'], cx0=box[0], cy0=box[1],
                       cw=box[2], ch=box[3], scale=256 / box[2] if prop <= 4 else .5,
                       rank_metric=state['rank'][prop].iloc[index],
                       sel_score=source[f'score{prop}'] if prop in (2, 3) else np.nan)
            expected.append(row)
        expected_rows += len(expected)
        observed = actual[actual.uuid.eq(actor)].sort_values(['property', 'index'])[FIELDS].reset_index(drop=True)
        desired = pd.DataFrame(expected, columns=FIELDS).sort_values(['property', 'index']).reset_index(drop=True)
        try:
            pd.testing.assert_frame_equal(observed, desired, check_dtype=False, check_exact=True)
        except AssertionError as error:
            differences.append({'uuid': actor, 'expected_rows': len(desired), 'actual_rows': len(observed),
                                'difference': str(error)[:3000]})
        for prop in range(1, 10):
            counts[prop] += int(state['pass_'][prop].sum())
            dedup[prop] += sum(len(rows) for (_, slot), rows in state['lists'].items() if slot == prop)
        if number % 25 == 0 or number == len(actors):
            print(json.dumps({'actors_checked': number, 'total': len(actors), 'frames': frames_total,
                              'differences': len(differences)}), flush=True)
    if set(actual.uuid) - set(actors):
        differences.append({'unplanned_actual_actors': sorted(set(actual.uuid) - set(actors))})
    if any(digest(path) != value for path, value in inputs.items()):
        raise ValueError('Replay inputs changed during audit')
    summary = json.loads((video / 'selection-summary.json').read_text())
    policy = json.loads(policy_path.read_text())
    if (policy['upstream_sha256'] != inputs[str(upstream_path)]
            or policy['selected_sha256'] != inputs[str(selected_path)]):
        differences.append({'policy_input_mismatch': True})
    if expected_frames is not None and frames_total != expected_frames:
        differences.append({'expected_candidate_frames': expected_frames, 'actual_candidate_frames': frames_total})
    if summary.get('candidate_rows') != frames_total:
        differences.append({'published_candidate_row_count_differs': True})
    result = {
        'passed': not differences, 'actors': len(actors), 'tracks': len(tracks), 'chunks': chunks_total,
        'candidate_frames': frames_total, 'expected_selected': expected_rows, 'actual_selected': len(actual),
        'exact_compared_fields': FIELDS, 'differences': differences,
        'archived_passing_frames_per_property': counts, 'archived_dedup_kept_per_property': dedup,
        'published_summary': summary, 'inputs': inputs,
        'oracle': 'Independent historical_video_check.oracle compiles the original archived select_v2.py filters/crop/dedup/append blocks, without the new wrapper',
        'candidate_set_caveat': 'Current cache excludes 18 invalid-pose sampled frames; replay compares all 580412 retained currently valid frames, not the unrestricted old candidate set.',
        'seconds': time.monotonic() - started, 'audit_command': [sys.executable, *sys.argv],
        'audit_script_sha256': digest(__file__),
        'oracle_script_sha256': digest(Path(__file__).with_name('historical_video_check.py')),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(output)
    print(json.dumps({key: result[key] for key in ('passed', 'actors', 'tracks', 'chunks', 'candidate_frames',
                                                 'expected_selected', 'actual_selected', 'seconds')}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-frames', type=int)
    args = parser.parse_args()
    result = replay(args.work_root, args.output, args.expected_frames)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
