"""Overlap one owned HF download with serial five-node scans of verified tracks.

Only receipts appended by this invocation authorize early work. A pass keeps
each original track dictionary unchanged and waits for its video and every pose
chunk. Its isolated plan/candidates cannot race the final plan or other passes.
The final scan preserves unavailable-target diagnostics while strictly requiring
all selected inputs and all selected chunks. No partial output is promoted.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from distributed_scan import DEFAULT_NODES, snapshot_code
from hf_sources import BUCKET, read_download_list, source_key, write_json
from import_scan_checkpoints import import_caches, track_map
from measurements import ARCHIVE, MODELS
from owned_subprocess import stop_group
from process_output import start_logged
from video_pipeline import code_config, digest, plan_identity


def utc():
    return datetime.now(timezone.utc).isoformat()


def file_identity(path):
    stat = Path(path).stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def track_inputs(track):
    yield source_key(track['source']), Path(track['video_path']), track['video_source_metadata']
    for chunk in track['usable_chunks']:
        key = str(chunk)
        yield source_key(track['pose_sources'].get(key, track['pose_sources'].get(chunk))), Path(
            track['pose_paths'].get(key, track['pose_paths'].get(chunk))), None


class VerifiedInputs:
    """Follow current-run receipts without trusting old files or partial JSONL.

    The owned downloader checks size, Xet hash and SHA-256 before atomic publish.
    Its receipt is appended after that check and includes the verified file's
    stat identity. We bind it to the visible inode/ctime and pinned video identity,
    avoiding a redundant full-video hash pass.
    """
    def __init__(self, receipts, tracks, offset=0):
        self.receipts = Path(receipts)
        self.offset = offset
        self.expected = {}
        self.verified = {}
        for track in tracks:
            for key, path, pinned in track_inputs(track):
                if path in self.expected and self.expected[path] != (key, pinned):
                    raise ValueError(f'Conflicting planned source: {path}')
                self.expected[path] = (key, pinned)

    def update(self):
        if not self.receipts.exists():
            return
        if self.receipts.stat().st_size < self.offset:
            raise ValueError('Current download receipts were truncated')
        with self.receipts.open() as handle:
            handle.seek(self.offset)
            while True:
                before = handle.tell()
                line = handle.readline()
                if not line or not line.endswith('\n'):
                    handle.seek(before)
                    break
                row = json.loads(line)
                path = Path(row['local_path'])
                if path not in self.expected:
                    raise ValueError(f'Unplanned download receipt: {path}')
                key, pinned = self.expected[path]
                if (row.get('status') not in ('downloaded', 'reused') or row.get('bucket') != BUCKET
                        or row.get('source_key') != key or len(row.get('sha256', '')) != 64
                        or not row.get('xet_hash')):
                    raise ValueError(f'Invalid verification receipt: {path}')
                if pinned and any(row.get(field) != pinned.get(field) for field in ('size', 'xet_hash')):
                    raise ValueError(f'HF video changed since target planning: {key}')
                identity = file_identity(path)
                verified = datetime.fromisoformat(row['verified_at'])
                if verified.tzinfo is None:
                    raise ValueError('Verification timestamp needs a timezone')
                # Use the recorded stat, not wall-clock ordering: filesystem
                # and controller clocks need not have identical offsets.
                if list(identity) != row.get('local_stat') or identity[2] != row['size']:
                    raise ValueError(f'Input changed after downloader verification: {path}')
                prior = self.verified.get(path)
                if prior and prior['identity'] != identity:
                    raise ValueError(f'Verified input was replaced during scanning: {path}')
                self.verified[path] = {'identity': identity, 'receipt': row}
            self.offset = handle.tell()

    def ready(self, track):
        paths = [path for _, path, _ in track_inputs(track)]
        return all(path in self.verified for path in paths)

    def assert_unchanged(self):
        for path, value in self.verified.items():
            if file_identity(path) != value['identity']:
                raise ValueError(f'Verified input changed during scanning: {path}')

    def missing(self):
        return [{'path': str(path), 'source_key': key} for path, (key, _) in self.expected.items()
                if path not in self.verified]


def create_pass(output_dir, destination, tracks):
    """Keep policy/model/source fingerprint inputs byte-for-byte equivalent."""
    destination.mkdir(parents=True)
    write_json(destination / 'tracks.json', tracks)
    for path in sorted(output_dir.glob('targets-*.json')):
        (destination / path.name).write_bytes(path.read_bytes())
    return destination


def require_selected_complete(report, tracks, identity):
    expected = sum(len(track['usable_chunks']) for track in tracks)
    if (report.get('plan_sha256') != identity or report.get('completed_chunks') != expected
            or report.get('missing_sources') != 0 or report.get('failures') != 0
            or report.get('available_targets_complete') is not True):
        raise ValueError('Final scan did not verify every selected input/chunk')


def scan_command(args, scripts, output_dir):
    # Incomplete target coverage is distinct from missing selected inputs. The
    # final report is checked strictly by require_selected_complete above.
    return [args.python, '-u', scripts / 'distributed_scan.py', '--output-dir', output_dir,
            '--workers', args.workers, '--python', args.python, '--nodes', *args.nodes,
            '--archive-root', args.archive_root, '--model-root', args.model_root, '--allow-incomplete']


def run(args):
    output = args.output_dir
    tracks = list(track_map(output / 'tracks.json').values())
    identity = plan_identity(output)
    config = code_config(args.archive_root, args.model_root)
    pairs = read_download_list(output / 'downloads.tsv', args.source_root)
    expected = {(key, path) for track in tracks for key, path, _ in track_inputs(track)}
    if set(pairs) != expected or len(pairs) != len(expected):
        raise ValueError('Download list does not exactly match selected video/pose inputs')
    run_dir = output / 'overlap' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:12])
    run_dir.mkdir(parents=True)
    scripts, hashes = snapshot_code(Path(__file__).resolve().parent, run_dir)
    state_path = output / 'download-scan-status.json'
    state = {'run_dir': str(run_dir), 'started_at': utc(), 'pid': os.getpid(), 'complete': False,
             'available_targets_complete': False, 'source_hashes': hashes, 'plan_sha256': identity,
             'passes': [], 'nodes': args.nodes, 'expected_tracks': len(tracks),
             'expected_chunks': sum(len(track['usable_chunks']) for track in tracks)}
    stopped = []
    previous = {}
    for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        previous[number] = signal.signal(number, lambda number, frame: stopped.append(number))
    children = []

    def status(stage, **values):
        state.update(stage=stage, updated_at=utc(), **values)
        write_json(state_path, state)
        write_json(run_dir / 'status.json', state)
        print(json.dumps({'stage': stage, **values}), flush=True)

    def launch(argv, name):
        child, stream = start_logged(list(map(str, argv)), run_dir / f'{name}.log', start_new_session=True)
        children.append((child, stream))
        return child

    def check():
        if stopped:
            raise RuntimeError('Owned download/scan pipeline was stopped')
        if plan_identity(output) != identity:
            raise ValueError('Final plan changed')
        verified.update()

    # Read only new receipts from our own downloader, never stale receipts from
    # a prior attempt. Resume still works: it emits fresh checks for valid files.
    offset = args.receipts.stat().st_size if args.receipts.exists() else 0
    verified = VerifiedInputs(args.receipts, tracks, offset)
    Path(str(args.receipts) + '.summary.json').unlink(missing_ok=True)
    caches = sorted({path.resolve() for path in args.cache} | {
        path.parent for path in (output / 'overlap').glob('*/pass-*/tracks.json')})
    try:
        download = launch([args.python, '-u', scripts / 'hf_sources.py', '--token-name', args.token_name,
                           'download', '--list', output / 'downloads.tsv', '--root', args.source_root,
                           '--receipts', args.receipts, '--workers', args.download_workers,
                           '--processes', args.download_processes, '--batch-size', args.download_batch_size,
                           '--allow-missing'], 'download')
        status('downloading-and-scanning', download_pid=download.pid)
        scanned = set()
        active, active_tracks, active_dir = None, [], None
        last_status = last_pass = 0
        while True:
            check()
            code = download.poll()
            if code not in (None, 0):
                raise RuntimeError(f'HF download failed ({code}); see {run_dir / "download.log"}')
            if active is not None and active.poll() is not None:
                if active.returncode:
                    raise RuntimeError(f'Early scan failed ({active.returncode}); see {active_dir}')
                report = json.loads((active_dir / 'scan-summary.json').read_text())
                require_selected_complete(report, active_tracks, plan_identity(active_dir))
                scanned.update(track['track'] for track in active_tracks)
                caches.append(active_dir)
                state['passes'][-1].update(finished_at=utc(), completed_chunks=report['completed_chunks'])
                active = None
            if code == 0 and active is None:
                break
            ready = [track for track in tracks if track['track'] not in scanned and verified.ready(track)]
            if active is None and ready and time.monotonic() - last_pass >= args.scan_interval:
                verified.assert_unchanged()
                active_dir = create_pass(output, run_dir / f'pass-{len(state["passes"]):04d}', ready)
                if caches:
                    import_caches(active_dir, caches, config)
                active_tracks = ready
                active = launch(scan_command(args, scripts, active_dir), active_dir.name)
                state['passes'].append({'output_dir': str(active_dir), 'tracks': len(ready),
                                        'started_at': utc(), 'pid': active.pid})
                last_pass = time.monotonic()
            if time.monotonic() - last_status >= 30:
                status('downloading-and-scanning', verified_files=len(verified.verified),
                       scanned_tracks=len(scanned), active_tracks=len(active_tracks) if active else 0,
                       download_returncode=code)
                last_status = time.monotonic()
            time.sleep(args.poll_seconds)
        check()
        verified.assert_unchanged()
        if code_config(args.archive_root, args.model_root) != config:
            raise ValueError('Fingerprinted scan implementation or model changed')
        summary = json.loads(Path(str(args.receipts) + '.summary.json').read_text())
        missing = verified.missing()
        write_json(run_dir / 'selected-inputs-missing.json', missing)
        if summary['failures'] or summary['missing'] or missing or summary['requested'] != len(pairs):
            raise ValueError('Selected HF inputs are incomplete; inspect download summary and selected-inputs-missing.json')
        status('importing-completed-scans', verified_files=len(verified.verified))
        if caches:
            import_caches(output, caches, config)
        status('final-selected-input-scan')
        final = launch(scan_command(args, scripts, output), 'final-scan')
        while final.poll() is None:
            check()
            time.sleep(args.poll_seconds)
        if final.returncode:
            raise RuntimeError(f'Final selected-input scan failed ({final.returncode})')
        check()
        verified.assert_unchanged()
        if code_config(args.archive_root, args.model_root) != config:
            raise ValueError('Fingerprinted scan implementation or model changed')
        report = json.loads((output / 'scan-summary.json').read_text())
        require_selected_complete(report, tracks, identity)
        status('finished', complete=report['complete'], available_targets_complete=True,
               finished_at=utc(), selected_inputs_verified=len(verified.verified), scan_summary=report)
        return 0
    except BaseException as error:
        status('failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        # Signal only children we created. Scan supervisors first receive a
        # chance to close their remote wrappers before local group cleanup.
        for child, _ in children:
            if child.poll() is None:
                child.terminate()
        for child, stream in children:
            stop_group(child, grace=25)
            stream.join()
        for number, handler in previous.items():
            signal.signal(number, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--receipts', type=Path, required=True)
    parser.add_argument('--token-name', default='oauth-paramluhadiya')
    parser.add_argument('--nodes', nargs='+', default=list(DEFAULT_NODES))
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--workers', type=int, default=64)
    parser.add_argument('--download-workers', type=int, default=8)
    parser.add_argument('--download-processes', type=int, default=8)
    parser.add_argument('--download-batch-size', type=int, default=16)
    parser.add_argument('--poll-seconds', type=float, default=5)
    parser.add_argument('--scan-interval', type=float, default=60)
    parser.add_argument('--archive-root', type=Path, default=ARCHIVE)
    parser.add_argument('--model-root', type=Path, default=MODELS)
    parser.add_argument('--cache', type=Path, action='append', default=[])
    args = parser.parse_args()
    if (min(args.workers, args.download_workers, args.download_processes, args.download_batch_size) < 1
            or args.poll_seconds <= 0 or args.scan_interval < 0 or not args.nodes
            or len(set(args.nodes)) != len(args.nodes) or any(node.startswith('-') for node in args.nodes)):
        parser.error('Invalid workers, intervals, or nodes')
    for name in ('output_dir', 'source_root', 'receipts', 'archive_root', 'model_root'):
        setattr(args, name, getattr(args, name).resolve())
    with ExitStack() as locks:
        # Separate supervisor lock avoids contention with serial pass/import/
        # final-scan locks; one receipt stream has exactly one owned downloader.
        for path in (args.output_dir / 'download-scan.lock', Path(str(args.receipts) + '.owner.lock')):
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = locks.enter_context(path.open('a'))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
