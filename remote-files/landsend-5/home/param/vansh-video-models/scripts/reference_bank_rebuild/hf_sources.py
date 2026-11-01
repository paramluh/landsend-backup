#!/usr/bin/env python3
"""Acquire reference-bank sources from Hugging Face with content receipts.

The current HF login is used unless --token-name selects an existing named login.
This never changes login state. Download lists contain either a bucket-relative
key, or ``key<TAB>destination``; relative destinations are under --root. Missing
objects and failed downloads have separate reports. Existing files are content
checked, and replacements are published only after size and Xet-hash validation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import fcntl
import json
import multiprocessing
import os
from pathlib import Path, PurePosixPath
import re
import threading
import time
from typing import Iterable

BUCKET = "InternalCan/aloe-conversationa-data"
IMAGE_PREFIX = "reference-images-512x960-v1"
VIDEO_PREFIX = "reference-videos-selected-actors"
_FORK_BATCH_FUNCTION = None


def _initialize_download_process():
    # The parent has completed metadata requests before forking. Create fresh
    # HTTP clients in each child; Xet is first initialized inside that child.
    from huggingface_hub import set_client_factory
    from huggingface_hub.utils._http import default_client_factory
    set_client_factory(default_client_factory)


def _run_forked_batch(batch):
    if _FORK_BATCH_FUNCTION is None:
        raise RuntimeError("Download worker was not initialized by its parent")
    return _FORK_BATCH_FUNCTION(batch)


def get_api(token_name: str | None = None):
    from huggingface_hub import HfApi

    if token_name is None:
        return HfApi()
    from huggingface_hub.utils import get_stored_tokens

    tokens = get_stored_tokens()
    if token_name not in tokens:
        raise ValueError(f"No stored Hugging Face login named {token_name!r}")
    return HfApi(token=tokens[token_name])


def source_key(value: str, bucket: str = BUCKET) -> str:
    prefix = f"hf://buckets/{bucket}/"
    key = value.removeprefix(prefix)
    parts = PurePosixPath(key).parts
    if not parts or key.startswith("/") or "://" in key or any(p in (".", "..") for p in parts):
        raise ValueError(f"Expected a key in HF bucket {bucket}: {value!r}")
    return key


def metadata_record(item, bucket: str = BUCKET) -> dict:
    row = {"bucket": bucket, "path": item.path, "type": item.type}
    for name in ("size", "xet_hash", "mtime", "uploaded_at"):
        value = getattr(item, name, None)
        if value is not None:
            row[name] = value.isoformat() if isinstance(value, datetime) else value
    return row


def write_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def inventory_prefix(api, prefix: str, output: str | Path, *, bucket: str = BUCKET,
                     recursive: bool = True) -> dict:
    """Stream an inventory to disk; keep an interrupted .part for diagnosis."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = size = directories = 0
    part = output.with_name(output.name + ".part")
    with part.open("w") as handle:
        for item in api.list_bucket_tree(bucket, prefix=source_key(prefix, bucket), recursive=recursive):
            row = metadata_record(item, bucket)
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            if row["type"] == "file":
                count += 1
                size += row["size"]
            else:
                directories += 1
    os.replace(part, output)
    summary = {"bucket": bucket, "prefix": prefix, "recursive": recursive,
               "files": count, "directories": directories, "bytes": size,
               "inventoried_at": datetime.now(timezone.utc).isoformat()}
    write_json(str(output) + ".summary.json", summary)
    return summary


def read_inventory(path: str | Path) -> list[dict]:
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_video_index(rows: Iterable[dict] | str | Path) -> dict[str, list[dict]]:
    """Map recording/speaker to original HF object(s); never pick ambiguously."""
    if isinstance(rows, (str, Path)):
        rows = read_inventory(rows)
    index: dict[str, list[dict]] = {}
    pattern = re.compile(r"^" + re.escape(VIDEO_PREFIX) + r"/([^/]+)/.+?/(speaker_\d+)/video/[^/]+/full\.mp4$")
    for row in rows:
        match = pattern.match(row["path"])
        if match:
            index.setdefault("/".join(match.groups()), []).append(row)
    return index


def resolve_video(recording_id: str, speaker: int | str, index: dict[str, list[dict]],
                  original_blob: str | None = None) -> dict:
    speaker = str(speaker).removeprefix("speaker_")
    track = f"{recording_id}/speaker_{speaker}"
    candidates = index.get(track, [])
    if original_blob:
        # An archived gs:// URI is provenance only: this always resolves an HF key.
        blob = original_blob.removeprefix("gs://aloe-processed-data/").split("#", 1)[0]
        parts = blob.split("/")
        if recording_id in parts:
            blob = "/".join(parts[parts.index(recording_id):])
        key = blob if blob.startswith(VIDEO_PREFIX + "/") else VIDEO_PREFIX + "/" + blob
        candidates = [row for row in candidates if row["path"] == key]
    if len(candidates) != 1:
        raise ValueError(f"Expected one HF source video for {track}; found {len(candidates)}")
    return candidates[0]


def read_download_list(path: str | Path, root: str | Path, *, bucket: str = BUCKET) -> list[tuple[str, Path]]:
    root = Path(root).expanduser().resolve()
    pairs = []
    destinations: dict[Path, str] = {}
    with Path(path).open() as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) not in (1, 2):
                raise ValueError(f"Invalid download list line {number}")
            key = source_key(parts[0], bucket)
            destination = Path(parts[1] if len(parts) == 2 else key).expanduser()
            if not destination.is_absolute():
                destination = Path(os.path.abspath(root / destination))
                if not destination.is_relative_to(root):
                    raise ValueError(f"Destination escapes --root on line {number}")
            destination = Path(os.path.abspath(destination))
            if destination in destinations:
                if destinations[destination] != key:
                    raise ValueError(f"Different sources share destination {destination}")
                continue
            destinations[destination] = key
            pairs.append((key, destination))
    return pairs


def _check_content(path: Path, info) -> tuple[bool, str | None]:
    import hf_xet

    if not path.is_file() or path.stat().st_size != info.size:
        return False, None
    hashed = hf_xet.hash_files([str(path)])[0]
    if hashed.file_size != info.size or hashed.hash != info.xet_hash:
        return False, None
    with path.open("rb") as handle:
        sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    return True, sha256


def download_pairs(api, pairs: Iterable[tuple[str, str | Path]], receipts_path: str | Path,
                   *, bucket: str = BUCKET, workers: int = 8, retries: int = 4,
                   batch_size: int = 32, processes: int = 1,
                   expected_inventory: Iterable[dict] | None = None,
                   max_total_bytes: int | None = None) -> dict:
    """Download and verify sources; missing keys are separate from failed files.

    Return counts plus missing/failures. Callers must inspect both, or use the
    CLI which fails by default. Pass expected_inventory to reject source changes
    since a previously saved inventory. No source uploads are ever performed.
    """
    if workers < 1 or retries < 1 or batch_size < 1 or processes < 1:
        raise ValueError("workers, retries, batch_size and processes must be positive")
    pairs = [(source_key(key, bucket), Path(os.path.abspath(Path(dest).expanduser()))) for key, dest in pairs]
    if len({dest for _, dest in pairs}) != len(pairs):
        raise ValueError("Each download must have a unique local destination")
    receipts = Path(receipts_path)
    receipts.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for key, _ in pairs))
    metadata = {}
    def lookup(chunk):
        for attempt in range(retries):
            try:
                return list(api.get_bucket_paths_info(bucket, chunk))
            except Exception:
                if attempt + 1 == retries:
                    raise
                time.sleep(min(2 ** attempt, 8))
    chunks = [keys[start:start + 500] for start in range(0, len(keys), 500)]
    with ThreadPoolExecutor(max_workers=min(workers, 8)) as pool:
        for items in pool.map(lookup, chunks):
            metadata.update((item.path, item) for item in items)
    print(json.dumps({"metadata_found": len(metadata), "metadata_requested": len(keys)}), flush=True)
    missing = [key for key in keys if key not in metadata]
    write_json(str(receipts) + ".missing.json", {"bucket": bucket, "missing": missing})
    expected = {row["path"]: row for row in expected_inventory or [] if row.get("type", "file") == "file"}
    changed = []
    if expected_inventory is not None:
        for key, item in metadata.items():
            prior = expected.get(key)
            if not prior or (prior.get("size"), prior.get("xet_hash")) != (item.size, item.xet_hash):
                changed.append(key)
        if changed:
            raise ValueError(f"{len(changed)} source objects missing from or changed since pinned inventory: {changed[:3]}")
    total = sum(metadata[key].size for key, _ in pairs if key in metadata)
    if max_total_bytes is not None and total > max_total_bytes:
        raise ValueError(f"Requested {total} bytes exceeds --max-total-bytes={max_total_bytes}")
    with Path(str(receipts) + ".request-metadata.jsonl").open("w") as handle:
        for key in sorted(metadata):
            handle.write(json.dumps(metadata_record(metadata[key], bucket), sort_keys=True) + "\n")
    lock = threading.Lock()
    failures = []
    downloaded = reused = recovered_parts = verified_bytes = 0

    def record_for(key, destination, status, digest):
        info = metadata[key]
        stat = destination.stat()
        return {"status": status, "bucket": bucket, "source_key": key,
                "hf_uri": f"hf://buckets/{bucket}/{key}", "local_path": str(destination),
                "size": info.size, "xet_hash": info.xet_hash, "sha256": digest,
                "local_stat": [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns],
                "verified_at": datetime.now(timezone.utc).isoformat(),
                "remote_metadata": metadata_record(info, bucket)}

    def get_batch(batch):
        records, pending = [], []
        recorded = 0

        def flush_records():
            nonlocal recorded
            if recorded == len(records):
                return
            with lock, receipts.open("a") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                handle.writelines(json.dumps(record, sort_keys=True) + "\n" for record in records[recorded:])
                handle.flush()
                os.fsync(handle.fileno())
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            recorded = len(records)

        for key, destination in batch:
            destination.parent.mkdir(parents=True, exist_ok=True)
            valid, digest = _check_content(destination, metadata[key])
            if valid:
                records.append(record_for(key, destination, "reused", digest))
            else:
                part = destination.with_name(destination.name + ".hf-part")
                valid, digest = _check_content(part, metadata[key])
                if valid:
                    # A stopped download group can leave complete files behind.
                    # Publish only after the same size/Xet/SHA checks as a fresh
                    # transfer; incomplete or corrupt parts still get fetched.
                    os.replace(part, destination)
                    record = record_for(key, destination, "reused", digest)
                    record["recovered_partial"] = True
                    records.append(record)
                else:
                    pending.append((key, destination))
        # Publish verified files before waiting for any slow request in this
        # batch, so incremental measurement workers can consume them promptly.
        flush_records()
        error = None
        for attempt in range(retries):
            if not pending:
                break
            parts = {destination: destination.with_name(destination.name + ".hf-part") for _, destination in pending}
            try:
                # One refresh-route HTTP request per batch, not per source file.
                api.download_bucket_files(bucket, [(metadata[key], str(parts[destination])) for key, destination in pending],
                                          raise_on_missing_files=True)
            except Exception as exc:
                error = {"download_error_type": type(exc).__name__,
                         "download_http_status": getattr(getattr(exc, "response", None), "status_code", None)}
                print(json.dumps({"download_retry": attempt + 1, "files": len(pending), **error}), flush=True)
            remaining = []
            for key, destination in pending:
                try:
                    valid, digest = _check_content(parts[destination], metadata[key])
                    if not valid:
                        raise ValueError("Downloaded bytes fail size or Xet-hash verification")
                    os.replace(parts[destination], destination)
                    records.append(record_for(key, destination, "downloaded", digest))
                except Exception as exc:
                    error = {**(error or {}), "verification_error_type": type(exc).__name__}
                    remaining.append((key, destination))
            pending = remaining
            flush_records()
            if pending and attempt + 1 < retries:
                time.sleep(min(2 ** attempt, 8))
        records.extend({"status": "failed", "source_key": key, "local_path": str(dest), **(error or {})}
                       for key, dest in pending)
        return records

    todo = [pair for pair in pairs if pair[0] in metadata]
    batches = [todo[start:start + batch_size] for start in range(0, len(todo), batch_size)]
    completed = 0
    if processes > 1:
        global _FORK_BATCH_FUNCTION
        _FORK_BATCH_FUNCTION = get_batch
        executor = ProcessPoolExecutor(max_workers=processes, mp_context=multiprocessing.get_context("fork"),
                                       initializer=_initialize_download_process)
        work = _run_forked_batch
    else:
        executor = ThreadPoolExecutor(max_workers=workers)
        work = get_batch
    with executor as pool:
        futures = {pool.submit(work, batch): batch for batch in batches}
        for future in as_completed(futures):
            try:
                records = future.result()
            except Exception as exc:
                records = [{"status": "failed", "source_key": key, "local_path": str(dest),
                            "error_type": type(exc).__name__} for key, dest in futures[future]]
            for record in records:
                if record["status"] == "failed":
                    failures.append(record)
                else:
                    downloaded += record["status"] == "downloaded"
                    reused += record["status"] == "reused"
                    recovered_parts += record.get("recovered_partial", False)
                    verified_bytes += record["size"]
            previous = completed
            completed += len(records)
            if completed // 100 != previous // 100:
                print(json.dumps({"completed": completed, "total": len(todo), "failed": len(failures)}), flush=True)
    summary = {"requested": len(pairs), "downloaded": downloaded, "reused": reused,
               "recovered_parts": recovered_parts, "verified_bytes": verified_bytes,
               "missing": missing, "failures": failures}
    write_json(str(receipts) + ".failures.json", {"bucket": bucket, "failures": failures})
    write_json(str(receipts) + ".summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default=BUCKET)
    parser.add_argument("--token-name", help="Existing named HF login; no login-state changes")
    commands = parser.add_subparsers(dest="command", required=True)
    inv = commands.add_parser("inventory")
    inv.add_argument("--prefix", required=True)
    inv.add_argument("--output", required=True)
    inv.add_argument("--shallow", action="store_true")
    dl = commands.add_parser("download")
    dl.add_argument("--list", required=True)
    dl.add_argument("--root", required=True)
    dl.add_argument("--receipts", required=True)
    dl.add_argument("--workers", type=int, default=8)
    dl.add_argument("--retries", type=int, default=4)
    dl.add_argument("--batch-size", type=int, default=32)
    dl.add_argument("--processes", type=int, default=1,
                    help="Independent download processes; one shared metadata inventory and locked receipts")
    dl.add_argument("--inventory", help="Frozen JSONL inventory; reject changed/missing source hashes")
    dl.add_argument("--allow-missing", action="store_true")
    dl.add_argument("--max-total-bytes", type=int)
    vi = commands.add_parser("video-index")
    vi.add_argument("--inventory", required=True)
    vi.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "video-index":
        index = build_video_index(args.inventory)
        write_json(args.output, index)
        print(json.dumps({"tracks": len(index), "ambiguous_tracks": sum(len(v) != 1 for v in index.values())}))
        return 0
    api = get_api(args.token_name)
    if args.command == "inventory":
        result = inventory_prefix(api, args.prefix, args.output, bucket=args.bucket, recursive=not args.shallow)
    else:
        pairs = read_download_list(args.list, args.root, bucket=args.bucket)
        result = download_pairs(api, pairs, args.receipts, bucket=args.bucket, workers=args.workers,
                                retries=args.retries, batch_size=args.batch_size, processes=args.processes,
                                max_total_bytes=args.max_total_bytes,
                                expected_inventory=read_inventory(args.inventory) if args.inventory else None)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.command == "download" and (result["failures"] or (result["missing"] and not args.allow_missing)):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
