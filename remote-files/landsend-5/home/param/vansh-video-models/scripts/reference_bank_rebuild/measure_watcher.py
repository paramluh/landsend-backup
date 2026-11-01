"""Process verified upstream tracks while HF downloads proceed; resumable and fail closed."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from measure_upstream import initialize, measure_track, measurement_config, checkpoint_valid, ARCHIVE, MODELS
from upstream import REFERENCE_PREFIX, SELECTOR_FOLDER
from hf_sources import write_json


def shard_report_dir(output_dir, shard, num_shards):
    return Path(output_dir) / 'watch-shards' / f'shard-{shard:05d}-of-{num_shards:05d}'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tracks', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--receipts', type=Path, action='append', required=True)
    p.add_argument('--workers', type=int, default=32)
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--run-id', default='standalone')
    p.add_argument('--report-dir', type=Path)
    args = p.parse_args()
    if args.workers < 1 or args.num_shards < 1 or not 0 <= args.shard < args.num_shards:
        p.error('workers and num-shards must be positive; shard must be in [0, num-shards)')
    report_dir = args.report_dir or (args.output_dir if args.num_shards == 1 else
                                    shard_report_dir(args.output_dir, args.shard, args.num_shards))
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / 'watch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return watch(args, report_dir)


def watch(args, report_dir):
    tracks = [line for line in args.tracks.read_text().splitlines() if line]
    if len(set(tracks)) != len(tracks):
        raise ValueError('Duplicate tracks in measurement plan')
    tracks = tracks[args.shard::args.num_shards]
    identity = {'run_id': args.run_id, 'shard': args.shard, 'num_shards': args.num_shards,
                'tracks_sha256': hashlib.sha256(args.tracks.read_bytes()).hexdigest(),
                'assigned_tracks': len(tracks)}
    reports = [Path(str(path) + '.missing.json') for path in args.receipts]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'tracks').mkdir(exist_ok=True)
    (report_dir / 'watch-complete.json').unlink(missing_ok=True)
    write_json(report_dir / 'watch-failures.json', [])
    write_json(report_dir / 'watch-runtime.json', {'pid': os.getpid(), 'hostname': socket.gethostname(), **identity})
    # Missing-source inventories are generated before downloading each stage.
    while not all(path.is_file() for path in reports):
        time.sleep(5)
    missing = {key for path in reports for key in json.loads(path.read_text())['missing']}
    config = measurement_config()
    write_json(report_dir / 'watch-code-inputs.json', config)
    completed = {track for track in tracks if checkpoint_valid(track, args.data_root, args.output_dir, config)}
    remaining, active, failures = set(tracks) - completed, {}, []
    verified, offsets = set(), {path: 0 for path in args.receipts}
    manifests = {}
    started, last = time.monotonic(), 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize,
                             initargs=(ARCHIVE, MODELS, args.data_root, args.output_dir, reports)) as pool:
        while remaining or active:
            for path in args.receipts:
                if not path.is_file():
                    continue
                with path.open() as handle:
                    handle.seek(offsets[path])
                    while True:
                        before = handle.tell()
                        line = handle.readline()
                        if not line or not line.endswith('\n'):
                            handle.seek(before)
                            break
                        row = json.loads(line)
                        verified.add(row['source_key'])
                    offsets[path] = handle.tell()
            available = verified | missing
            for track in sorted(remaining):
                if len(active) >= args.workers * 2:
                    break
                prefix = f'{REFERENCE_PREFIX}/{track}/{SELECTOR_FOLDER}'
                manifest_key = f'{prefix}/manifest.json'
                if not {manifest_key, f'{prefix}/selected_poses.parquet'} <= available:
                    continue
                if track not in manifests:
                    manifests[track] = ([] if manifest_key in missing else
                        json.loads((args.data_root / manifest_key).read_text())['images'])
                needed = {f'{prefix}/{row["category"]}.png' for row in manifests[track]}
                # Audit visits all nine files, so wait for all existing HF slot files.
                from upstream import SLOTS
                needed |= {f'{prefix}/{slot}.png' for slot in SLOTS}
                if not needed <= available:
                    continue
                active[pool.submit(measure_track, track)] = track
                remaining.remove(track)
            done, _ = wait(active, timeout=2, return_when=FIRST_COMPLETED) if active else (set(), set())
            for future in done:
                track = active.pop(future)
                try:
                    future.result()
                    completed.add(track)
                except Exception as error:
                    failures.append({'track': track, 'error': str(error), 'type': type(error).__name__})
                    write_json(report_dir / 'watch-failures.json', failures)
            if time.monotonic() - last > 30:
                status = {'completed': len(completed), 'active': len(active), 'waiting_for_download': len(remaining),
                          'failed': len(failures), 'verified_source_files': len(verified),
                          'seconds': round(time.monotonic() - started), **identity}
                print(json.dumps(status), flush=True)
                write_json(report_dir / 'watch-status.json', status)
                last = time.monotonic()
            if not active:
                if remaining and all(Path(str(path) + '.summary.json').is_file() for path in args.receipts):
                    # A downloader can append its final receipts and publish
                    # its summary while we wait for the last measurement. Our
                    # offsets are from the loop's start; drain that new tail
                    # before concluding that the remaining tracks lack inputs.
                    if any(path.is_file() and path.stat().st_size > offsets[path] for path in args.receipts):
                        continue
                    raise RuntimeError(f'Download stages finished with {len(remaining)} tracks not ready')
                time.sleep(5)
    if failures:
        raise SystemExit(f'{len(failures)} tracks failed; fix and resume')
    complete = {'complete': True, 'tracks': len(completed), **identity}
    write_json(report_dir / 'watch-status.json', {'completed': len(completed), 'active': 0,
               'waiting_for_download': 0, 'failed': 0, 'verified_source_files': len(verified),
               'seconds': round(time.monotonic() - started), **identity})
    write_json(report_dir / 'watch-complete.json', complete)
    print(json.dumps(complete), flush=True)


if __name__ == '__main__':
    main()
