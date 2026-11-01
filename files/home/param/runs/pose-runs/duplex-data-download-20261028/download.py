#!/usr/bin/env python3
"""Download pinned HF dataset files and verify their repository hashes."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import socket
import time

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
os.environ.setdefault("HF_XET_NUM_CONCURRENT_RANGE_GETS", "16")
os.environ.setdefault("HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS", "128")
os.environ.setdefault("HF_XET_CLIENT_MAX_IDLE_CONNECTIONS", "128")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
import httpx
from huggingface_hub import hf_hub_download, set_client_factory
from huggingface_hub.utils._http import hf_request_event_hook


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def digest(path, entry):
    h = hashlib.sha256() if entry.get("sha256") else hashlib.sha1()
    if not entry.get("sha256"):
        h.update(f"blob {entry['size']}\0".encode())
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fetch(root, entry):
    dest = root / entry["directory"] / entry["path"]
    expected = entry.get("sha256") or entry["git_blob_sha1"]
    force = False
    for attempt in range(8):
        try:
            if not dest.is_file() or dest.stat().st_size != entry["size"] or force:
                hf_hub_download(
                    repo_id=entry["repo"], repo_type="dataset", revision=entry["revision"],
                    filename=entry["path"], local_dir=root / entry["directory"],
                    force_download=force,
                )
            if dest.stat().st_size != entry["size"] or digest(dest, entry) != expected:
                force = True
                raise ValueError("File size or repository checksum mismatch")
            return entry
        except Exception as exc:
            print(json.dumps({"event": "retry", "repo": entry["repo"], "path": entry["path"],
                              "attempt": attempt + 1, "error_type": type(exc).__name__}), flush=True)
            if attempt == 7:
                raise RuntimeError(f"Failed after 8 attempts: {entry['repo']}/{entry['path']}") from None
            time.sleep(min(60, 2 ** attempt))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    assert args.workers > 0
    set_client_factory(lambda: httpx.Client(
        event_hooks={"request": [hf_request_event_hook]}, follow_redirects=True, timeout=None,
        limits=httpx.Limits(max_connections=max(256, args.workers * 2),
                           max_keepalive_connections=max(128, args.workers)),
    ))
    args.root.mkdir(parents=True, exist_ok=True)
    args.control.mkdir(parents=True, exist_ok=True)
    lock = (args.control / "download.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads(args.manifest.read_text())
    entries = manifest["files"]
    for entry in entries:
        for key in ("directory", "path"):
            path = PurePosixPath(entry[key])
            assert not path.is_absolute() and ".." not in path.parts, entry[key]
    total = sum(e["size"] for e in entries)
    remaining = sum(e["size"] for e in entries
                    if not (args.root / e["directory"] / e["path"]).is_file())
    assert shutil.disk_usage(args.root).free > remaining + 80 * 10**9, "Insufficient free space"
    state = {"hostname": socket.gethostname(), "pid": os.getpid(), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "download_state": "running", "training_ready": False,
             "pending_preparation": manifest.get("pending_preparation", ["Rebuild s2h-m1-samsep stores from the original stores and SAM tokens"]),
             "missing_components": manifest.get("missing_components", []),
             "root": str(args.root), "files_total": len(entries), "bytes_total": total,
             "max_workers": args.workers, "workers": args.workers,
             "cpu_count_available": len(os.sched_getaffinity(0)),
             "xet_high_performance": os.environ["HF_XET_HIGH_PERFORMANCE"],
             "xet_concurrent_range_gets": os.environ["HF_XET_NUM_CONCURRENT_RANGE_GETS"],
             "xet_max_concurrent_files": os.environ["HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS"],
             "files_verified": 0, "bytes_verified": 0, "failed": [], "sources": manifest["sources"]}
    (args.control / "pid").write_text(str(os.getpid()) + "\n")
    shutil.copyfile(args.manifest, args.root / "SOURCE_MANIFEST.json")

    def report():
        state["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        atomic_json(args.control / "status.json", state)
        print(json.dumps({k: state[k] for k in ["hostname", "download_state", "files_verified", "files_total", "bytes_verified", "bytes_total", "updated_utc"]}), flush=True)

    report()
    # Metadata, SAM shards, and row stores first; audio follows. All are downloaded directly from HF on this node.
    entries.sort(key=lambda e: (e["path"].startswith("flac/"), e["size"] > 10**7, e["repo"], e["path"]))
    with (args.control / "verified.jsonl").open("a") as journal, concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {}
        queue = iter(entries)
        exhausted = False
        last_report = time.monotonic()
        while pending or not exhausted:
            # Bound queued work so concurrency can be tuned without restarting transfers.
            try:
                requested = int(json.loads((args.control / "concurrency.json").read_text())["workers"])
                state["workers"] = max(1, min(args.workers, requested))
            except (FileNotFoundError, ValueError, KeyError):
                pass
            while len(pending) < state["workers"] and not exhausted:
                entry = next(queue, None)
                if entry is None:
                    exhausted = True
                else:
                    pending[pool.submit(fetch, args.root, entry)] = entry
            done, _ = concurrent.futures.wait(pending, timeout=15, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                entry = pending.pop(future)
                try:
                    future.result()
                except Exception as exc:
                    state["failed"].append({"repo": entry["repo"], "path": entry["path"], "error_type": type(exc).__name__})
                else:
                    state["files_verified"] += 1
                    state["bytes_verified"] += entry["size"]
                    journal.write(json.dumps({"repo": entry["repo"], "path": entry["path"], "size": entry["size"],
                                              "sha256": entry.get("sha256"), "git_blob_sha1": entry.get("git_blob_sha1")}) + "\n")
            if time.monotonic() - last_report > 15 or not pending:
                journal.flush()
                report()
                last_report = time.monotonic()
    state["download_state"] = "failed" if state["failed"] else "complete"
    state["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report()
    if not state["failed"]:
        atomic_json(args.control / "DOWNLOADS_COMPLETE.json", state)
    raise SystemExit(bool(state["failed"]))


if __name__ == "__main__":
    main()
