#!/usr/bin/env python3
"""Async, high-throughput downloader for many small GCS JSON files.

Reads a plain-text file where each line is a GCS object reference of the form
``<bucket>/<object/key/path>`` (optionally prefixed with ``gs://``) and
downloads every file to a single local directory, flattening to its basename.

Why not the ``google-cloud-storage`` ``transfer_manager``?
    For tiny files (few KB) it tops out around 50 files/s because of per-thread
    auth/handshake overhead and process-pool startup costs per chunk. With
    aiohttp + a shared bearer token we get ~500 files/s at 256 concurrency.

Usage:
    uv run python scripts/download_jsons.py upperbody_ideal_virtual_bucket_json_paths.txt
    uv run python scripts/download_jsons.py paths.txt --dest /mnt/somfs/hands/clip_jsons
    uv run python scripts/download_jsons.py paths.txt -c 512
"""
from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
import time
from pathlib import Path

import aiohttp

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEST = Path("/mnt/somfs/hands/clip_jsons")
PROGRESS_EVERY = 2000  # files between progress prints


def get_access_token() -> str:
    """Fetch a fresh ADC bearer token via gcloud."""
    return subprocess.check_output(
        ["gcloud", "auth", "application-default", "print-access-token"],
        text=True,
    ).strip()


def read_paths(txt_path: Path) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    with open(txt_path) as f:
        for line in f:
            p = line.strip()
            if not p or p in seen:
                continue
            if p.startswith("gs://"):
                p = p[5:]
            seen.add(p)
            out.append(p)
    return out


async def fetch_one(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    gcs_path: str,
    dest_dir: Path,
    token_ref: list[str],
    counters: dict,
) -> None:
    bucket, _, key = gcs_path.partition("/")
    if not key:
        counters["err"] += 1
        return
    url = f"https://storage.googleapis.com/{bucket}/{key}"
    local = dest_dir / key.rsplit("/", 1)[-1]
    async with sem:
        for attempt in range(3):
            try:
                headers = {"Authorization": f"Bearer {token_ref[0]}"}
                async with session.get(url, headers=headers) as r:
                    if r.status == 200:
                        data = await r.read()
                        local.write_bytes(data)
                        counters["ok"] += 1
                        return
                    if r.status == 401:
                        token_ref[0] = get_access_token()
                        continue
                    if r.status in (429, 500, 502, 503, 504):
                        await asyncio.sleep(0.5 * (attempt + 1))
                        continue
                    counters["err"] += 1
                    return
            except (aiohttp.ClientError, asyncio.TimeoutError):
                await asyncio.sleep(0.3 * (attempt + 1))
                continue
        counters["err"] += 1


async def run(paths: list[str], dest_dir: Path, conc: int, total: int) -> dict:
    token_ref = [get_access_token()]
    counters = {"ok": 0, "err": 0}
    sem = asyncio.Semaphore(conc)
    timeout = aiohttp.ClientTimeout(total=60, connect=15)
    conn = aiohttp.TCPConnector(
        limit=conc,
        limit_per_host=conc,
        ttl_dns_cache=300,
        force_close=False,
    )

    last_done = 0
    last_t = time.monotonic()
    start = last_t

    async def progress_reporter():
        nonlocal last_done, last_t
        while True:
            await asyncio.sleep(2.0)
            done = counters["ok"] + counters["err"]
            if done == last_done:
                continue
            now = time.monotonic()
            inst = (done - last_done) / max(now - last_t, 1e-3)
            avg = done / max(now - start, 1e-3)
            eta = (total - done) / avg if avg > 0 else 0
            print(
                f"  [{done:>7}/{total}] {counters['ok']} ok, {counters['err']} err "
                f"| {inst:.0f} files/s now | {avg:.0f} avg | ~{eta:.0f}s left",
                file=sys.stderr,
            )
            last_done, last_t = done, now

    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
        prog_task = asyncio.create_task(progress_reporter())
        try:
            await asyncio.gather(
                *(fetch_one(session, sem, p, dest_dir, token_ref, counters) for p in paths)
            )
        finally:
            prog_task.cancel()
            try:
                await prog_task
            except asyncio.CancelledError:
                pass

    return counters


def main():
    parser = argparse.ArgumentParser(
        description="Async downloader for many small GCS JSON files."
    )
    parser.add_argument(
        "paths_file",
        type=Path,
        help="Plain-text file of GCS paths (one per line, format <bucket>/<key>).",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"Local destination directory (default: {DEFAULT_DEST}).",
    )
    parser.add_argument(
        "-c", "--concurrency", type=int, default=256,
        help="Concurrent in-flight requests (default: 256; 512 sometimes hurts).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Optional cap on number of paths to download (for smoke-testing).",
    )
    args = parser.parse_args()

    if not args.paths_file.exists():
        parser.error(f"paths file not found: {args.paths_file}")

    gcs_paths = read_paths(args.paths_file)
    if args.limit:
        gcs_paths = gcs_paths[: args.limit]
    print(f"Read {len(gcs_paths):,} unique paths from {args.paths_file}", file=sys.stderr)

    dest_dir: Path = args.dest
    dest_dir.mkdir(parents=True, exist_ok=True)

    existing = {p.name for p in dest_dir.iterdir() if p.is_file()}
    to_download = [p for p in gcs_paths if p.rsplit("/", 1)[-1] not in existing]
    total = len(to_download)
    skipped = len(gcs_paths) - total
    print(
        f"Downloading {total:,} files to {dest_dir}/ "
        f"({skipped:,} already exist, concurrency={args.concurrency})",
        file=sys.stderr,
    )
    if not to_download:
        print("Nothing new to download.", file=sys.stderr)
        return

    t0 = time.monotonic()
    counters = asyncio.run(run(to_download, dest_dir, args.concurrency, total))
    elapsed = time.monotonic() - t0
    print(
        f"\nDone in {elapsed:.1f}s: {counters['ok']:,} downloaded, "
        f"{counters['err']:,} failed ({counters['ok']/elapsed:.0f} files/s avg)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
