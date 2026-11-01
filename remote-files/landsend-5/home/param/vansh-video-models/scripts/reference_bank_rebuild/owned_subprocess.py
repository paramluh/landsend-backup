"""Run one owned process group; stop it on controller loss or an expired lease."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time


def stop_group(process, grace=10):
    """The caller must have created this process with start_new_session=True."""
    if process.poll() == 0:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    # The leader can exit before its children. The group remains owned until
    # reaped here; kill remaining workers even when the leader already exited.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run(argv, lease, run_id, lease_seconds=90):
    stopped = []
    for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(number, lambda number, frame: stopped.append(number))
    child = subprocess.Popen(argv, start_new_session=True)
    try:
        while True:
            code = child.poll()
            if code is not None:
                return code
            if stopped:
                raise RuntimeError('Owned scan wrapper received a stop signal')
            current = json.loads(Path(lease).read_text())
            if current['run_id'] != run_id or time.time() - current['time'] > lease_seconds:
                raise RuntimeError('Owned scan controller lease expired')
            if select.select([sys.stdin], [], [], 0)[0] and not os.read(sys.stdin.fileno(), 1):
                raise RuntimeError('Owned scan controller disconnected')
            time.sleep(.5)
    finally:
        stop_group(child)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lease', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--lease-seconds', type=float, default=90)
    parser.add_argument('argv', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    argv = args.argv[1:] if args.argv[:1] == ['--'] else args.argv
    if not argv:
        parser.error('A child command is required')
    return run(argv, args.lease, args.run_id, args.lease_seconds)


if __name__ == '__main__':
    raise SystemExit(main())
