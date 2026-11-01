#!/usr/bin/env python3
"""Run identical CPU video-scan shards on the allocated nodes, then validate all.

This foreground supervisor is intended to be owned by the root build process.
It keeps SSH child processes open, logs each node on shared storage, and returns
only after all nodes finish and their candidate receipts pass aggregation.
Resume reuses individually verified candidate checkpoints. No video acquisition
is performed here; supply only verified inputs (download_and_scan.py can overlap
acquisition with immutable plans containing the currently ready tracks).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid

from hf_sources import write_json
from measurements import ARCHIVE, MODELS
from process_output import CPU_ENV, start_logged

DEFAULT_NODES = ("192.168.0.4", "192.168.0.9", "192.168.0.7", "192.168.0.8", "192.168.0.6")
SSH = ("ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
       "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=6")


def snapshot_code(source, output_dir):
    """Publish one immutable, content-addressed snapshot on shared paramfs."""
    payloads = {path.name: path.read_bytes() for path in sorted(source.glob("*.py"))}
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    snapshots = output_dir / "distributed-code"
    snapshots.mkdir(parents=True, exist_ok=True)
    destination = snapshots / identity
    if not destination.exists():
        temporary = Path(tempfile.mkdtemp(prefix=".snapshot-", dir=snapshots))
        try:
            for name, data in payloads.items():
                (temporary / name).write_bytes(data)
            write_json(temporary / "source-hashes.json", hashes)
            temporary.rename(destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    for name, expected in hashes.items():
        if hashlib.sha256((destination / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Shared code snapshot changed: {destination / name}")
    return destination, hashes


def remote_command(node, argv):
    return [*SSH, node, shlex.join(list(map(str, argv)))]


def preflight(node, args, snapshot):
    code = ("from pathlib import Path; import json, socket, mediapipe, pandas; "
            f"assert Path({str(snapshot / 'video_pipeline.py')!r}).is_file(); "
            f"assert Path({str(args.output_dir / 'tracks.json')!r}).is_file(); "
            f"assert Path({str(args.model_root / 'face_landmarker.task')!r}).is_file(); "
            "print(json.dumps({'hostname': socket.gethostname(), 'mediapipe': mediapipe.__version__, "
            "'pandas': pandas.__version__}))")
    result = subprocess.run(remote_command(node, ['env', *CPU_ENV, args.python, "-c", code]), stdout=subprocess.PIPE,
                            text=True, timeout=60)
    print(result.stdout, end='', flush=True)
    if result.returncode:
        raise RuntimeError(f"Node {node} preflight failed ({result.returncode})")
    return {"node": node, **json.loads(result.stdout.strip().splitlines()[-1])}


def run(args):
    snapshot, hashes = snapshot_code(Path(__file__).resolve().parent, args.output_dir)
    with ThreadPoolExecutor(max_workers=len(args.nodes)) as pool:
        nodes = list(pool.map(lambda node: preflight(node, args, snapshot), args.nodes))
    logs = args.output_dir / "distributed-logs"
    logs.mkdir(exist_ok=True)
    run_id = uuid.uuid4().hex
    lease = logs / f"lease-{run_id}.json"
    stopped = []
    previous_handlers = {}
    for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        previous_handlers[number] = signal.signal(number, lambda number, frame: stopped.append(number))
    write_json(lease, {"run_id": run_id, "time": time.time()})
    state = {"started_at": datetime.now(timezone.utc).isoformat(), "supervisor_pid": os.getpid(),
             "run_id": run_id,
             "snapshot": str(snapshot), "source_hashes": hashes, "nodes": nodes,
             "workers_per_node": args.workers, "complete": False, "returncodes": {}}
    state_path = args.output_dir / "distributed-scan.json"
    write_json(state_path, state)
    # Shards deliberately never write global reports. Remove an old aggregate
    # so interrupted new work cannot masquerade as this invocation's result.
    (args.output_dir / "scan-summary.json").unlink(missing_ok=True)
    children = []
    try:
        for shard, node in enumerate(args.nodes):
            argv = [args.python, "-u", snapshot / "video_pipeline.py", "scan", "--output-dir", args.output_dir,
                    "--archive-root", args.archive_root, "--model-root", args.model_root,
                    "--workers", args.workers, "--shard", shard, "--num-shards", len(args.nodes)]
            if args.allow_incomplete:
                argv.append("--allow-incomplete")
            log_path = logs / f"scan-{shard:05d}-{node}.log"
            wrapped = ['env', *CPU_ENV, args.python, "-u", snapshot / "owned_subprocess.py", "--lease", lease,
                       "--run-id", run_id, "--", *argv]
            process, output = start_logged(remote_command(node, wrapped), log_path, stdin=subprocess.PIPE)
            children.append((node, process, output))
            nodes[shard].update(shard=shard, ssh_pid=process.pid, log=str(log_path))
        write_json(state_path, state)
        print(json.dumps({"nodes": nodes, "snapshot": str(snapshot), "stage": "scanning"}), flush=True)
        while True:
            if stopped:
                raise RuntimeError("Owned distributed scan was stopped")
            write_json(lease, {"run_id": run_id, "time": time.time()})
            state["returncodes"] = {node: process.poll() for node, process, _ in children}
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            write_json(state_path, state)
            if any(code not in (None, 0) for code in state["returncodes"].values()):
                raise RuntimeError(f"Video scan shard failed: {state['returncodes']}")
            if all(code is not None for code in state["returncodes"].values()):
                break
            time.sleep(10)
        if any(state["returncodes"].values()):
            raise RuntimeError(f"Video scan shard failed; inspect {logs}: {state['returncodes']}")
        argv = [args.python, "-u", snapshot / "video_pipeline.py", "aggregate-scan",
                "--output-dir", args.output_dir, "--num-shards", len(args.nodes),
                "--archive-root", args.archive_root, "--model-root", args.model_root]
        if args.allow_incomplete:
            argv.append("--allow-incomplete")
        subprocess.run(list(map(str, argv)), check=True)
        report = json.loads((args.output_dir / "scan-summary.json").read_text())
        state.update(complete=report["complete"], available_targets_complete=report["available_targets_complete"],
                     finished_at=datetime.now(timezone.utc).isoformat(), aggregation_verified=True)
        write_json(state_path, state)
        print(json.dumps(state), flush=True)
        return 0
    except BaseException as exc:
        state.update(error=f"{type(exc).__name__}: {exc}", complete=False)
        write_json(state_path, state)
        raise
    finally:
        write_json(lease, {"run_id": run_id, "time": 0})
        # EOF and the shared lease reach only this run's wrappers. Each wrapper
        # terminates the exact process group that it created, including MP pools.
        for _, process, _ in children:
            if process.stdin:
                process.stdin.close()
        deadline = time.monotonic() + 15
        while any(process.poll() is None for _, process, _ in children) and time.monotonic() < deadline:
            time.sleep(.1)
        for _, process, output in children:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            output.join()
        for number, handler in previous_handlers.items():
            signal.signal(number, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nodes", nargs="+", default=list(DEFAULT_NODES))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--archive-root", type=Path, default=ARCHIVE)
    parser.add_argument("--model-root", type=Path, default=MODELS)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if len(set(args.nodes)) != len(args.nodes):
        parser.error("--nodes must be unique")
    if any(node.startswith("-") for node in args.nodes):
        parser.error("Invalid SSH node")
    args.output_dir = args.output_dir.resolve()
    args.archive_root, args.model_root = args.archive_root.resolve(), args.model_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "distributed-scan.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
