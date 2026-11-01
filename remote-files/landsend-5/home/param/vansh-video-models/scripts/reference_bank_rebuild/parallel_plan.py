"""Run the frozen upstream planner over ordered track chunks in CPU processes."""

import argparse
import ctypes
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import signal
import time

import upstream
from upstream import BLACKBANDS_CSV, CSV_ROOT, build_plan, load_corpus, write_inventory


_WORKER = None


def initialize_worker(corpus, data_root, known_missing, parent_pid):
    global _WORKER
    # Linux workers must stop if the owned planner process is terminated.
    if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'Unable to bind planner worker to its parent lifetime')
    if os.getppid() != parent_pid:
        raise RuntimeError('Planner parent exited during worker startup')
    # Parallelism is across independent tracks; avoid per-process Arrow pools.
    import pyarrow
    pyarrow.set_cpu_count(1)
    pyarrow.set_io_thread_count(1)
    _WORKER = corpus, data_root, known_missing


def plan_chunk(tracks):
    import pandas as pd
    corpus, data_root, known_missing = _WORKER
    frame, diagnostics = build_plan(replace(corpus, tracks=tracks), data_root, known_missing)
    # Reconstruct the global DataFrame once. Concatenating inferred chunk dtypes
    # changes all-null/mixed columns compared with the original serial planner.
    records_frame = frame.astype(object)
    for column in frame.columns:
        # An explicitly NaN numeric score must still infer float globally;
        # all-None columns in the serial planner instead infer object.
        if not pd.api.types.is_float_dtype(frame[column].dtype):
            records_frame[column] = records_frame[column].where(pd.notna(frame[column]), None)
    records = records_frame.to_dict('records')
    return records, diagnostics


def parallel_build_plan(corpus, data_root, known_missing=(), *, workers=64, chunk_size=32, progress=None):
    import pandas as pd
    if workers < 1 or chunk_size < 1 or not corpus.tracks:
        raise ValueError('Workers, chunk size and corpus track count must be positive')
    chunks = [corpus.tracks[start:start + chunk_size] for start in range(0, len(corpus.tracks), chunk_size)]
    # All policy ranges and source hashes are identical in each chunk. Avoid
    # replicating the unrelated all_clips table and roster in every worker.
    policy = replace(corpus, tracks=[], all_clips=[], actors=[])
    rows, issues, base = [], [], None
    changing = {'rows', 'candidates', 'issue_counts', 'issues', 'fatal_input_issues', 'source_status_counts',
                'incomplete_source_tracks', 'actors_with_incomplete_source'}
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(name, '1')
    with ProcessPoolExecutor(max_workers=min(workers, len(chunks)), mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize_worker,
                             initargs=(policy, Path(data_root), set(known_missing), os.getpid())) as pool:
        # map yields input order even when workers finish in a different order.
        for count, (records, diagnostics) in enumerate(pool.map(plan_chunk, chunks), 1):
            constants = {key: value for key, value in diagnostics.items() if key not in changing}
            if base is not None and constants != base:
                raise ValueError('Planner policy diagnostics differ between chunks')
            base = constants
            offset = len(rows)
            for index, row in enumerate(records):
                row['iid'] = offset + index
            rows.extend(records)
            issues.extend(diagnostics['issues'])
            if progress:
                progress(min(count * chunk_size, len(corpus.tracks)), len(corpus.tracks))
    plan = pd.DataFrame(rows)
    plan['frame_index'] = plan['frame_index'].astype('Int64')
    diagnostics = {
        'rows': len(rows), 'candidates': int(plan['candidate'].sum()),
        'issue_counts': dict(Counter(issue['kind'] for issue in issues)), 'issues': issues,
        'fatal_input_issues': sum(issue['kind'] != 'confirmed_hf_missing' for issue in issues),
        'source_status_counts': {str(key): int(value) for key, value in plan.source_status.value_counts().items()},
        'incomplete_source_tracks': sorted(set(plan.loc[plan.source_incomplete, 'track'])),
        'actors_with_incomplete_source': sorted(set(plan.loc[plan.source_incomplete, 'uuid'])),
        **base,
    }
    return plan, diagnostics


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv-root', type=Path, default=CSV_ROOT)
    parser.add_argument('--blackbands-csv', type=Path, default=BLACKBANDS_CSV)
    parser.add_argument('--reserved-csv', type=Path, action='append', default=[])
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--missing-report', type=Path, action='append', default=[])
    parser.add_argument('--workers', type=int, default=64)
    parser.add_argument('--chunk-size', type=int, default=32)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'parallel-plan.complete.json').unlink(missing_ok=True)
    started = time.monotonic()
    corpus = load_corpus(args.csv_root, args.blackbands_csv, args.reserved_csv)
    write_inventory(corpus, args.output_dir)
    known_missing = {key for path in args.missing_report for key in json.loads(path.read_text())['missing']}
    last = 0

    def progress(done, total):
        nonlocal last
        if done - last >= 256 or done == total:
            print(json.dumps({'planned_tracks': done, 'tracks': total, 'seconds': time.monotonic() - started}), flush=True)
            last = done

    plan, diagnostics = parallel_build_plan(corpus, args.data_root, known_missing, workers=args.workers,
                                             chunk_size=args.chunk_size, progress=progress)
    write_json(args.output_dir / 'input-diagnostics.json', diagnostics)
    if diagnostics['fatal_input_issues']:
        raise ValueError(f"{diagnostics['fatal_input_issues']} incomplete/mismatched inputs; see input-diagnostics.json")
    temporary = args.output_dir / 'plan.parquet.tmp'
    plan.to_parquet(temporary, index=False)
    temporary.replace(args.output_dir / 'plan.parquet')
    receipt = {'complete': True, 'tracks': len(corpus.tracks), 'rows': len(plan), 'workers': args.workers,
               'chunk_size': args.chunk_size, 'seconds': time.monotonic() - started,
               'source_hashes': corpus.source_hashes,
               'implementation_sha256': {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in (
                   ('upstream.py', Path(upstream.__file__)), ('parallel_plan.py', Path(__file__)))},
               'plan_sha256': hashlib.sha256((args.output_dir / 'plan.parquet').read_bytes()).hexdigest()}
    write_json(args.output_dir / 'parallel-plan.complete.json', receipt)
    print(json.dumps({key: value for key, value in diagnostics.items() if key != 'issues'}), flush=True)


if __name__ == '__main__':
    main()
