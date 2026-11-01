#!/usr/bin/env python3
"""Convert pose NPZ files to plain memmappable NPY format.

Source: pose_files/<stem>_poses.npz  (compressed npz, ``poses`` is float32 (T, 133, 3))
Target: pose_files_npy/<stem>_poses.npy  (uncompressed, memmappable)

Properties:
- Multiprocessing across many cores, plus a thread pool *inside* each worker.
  Both the network-FS reads/writes and ``zlib`` decompression release the GIL,
  so threads overlap I/O latency and let decompression run on multiple cores.
- Fast decompression path: reads only the ``poses`` member's raw DEFLATE stream
  and inflates it with ``zlib`` directly, skipping the ZIP CRC32 verification and
  the other (tiny) members that ``np.load`` would also touch. Falls back to
  ``np.load`` if the archive layout is unexpected.
- Resumable: skips files whose output already exists with non-zero size.
- Atomic writes: serialise to memory, write to ``.partial``, then rename.
- Skips malformed files (object dtype, wrong shape, etc.).
- Logs all errors to convert_errors.txt
"""

import argparse
import io
import multiprocessing as mp
import os
import struct
import sys
import time
import zipfile
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from numpy.lib import format as npformat
from tqdm import tqdm

SRC = Path("pose_files")
DST = Path("/mnt/somfs/pose_conditioning_uncompressed")

_SRC_ROOT: Path | None = None
_DST_ROOT: Path | None = None
_THREADS: int = 1
_OVERWRITE: bool = False


def init_worker(src_root: str, dst_root: str, threads: int, overwrite: bool = False) -> None:
    global _SRC_ROOT, _DST_ROOT, _THREADS, _OVERWRITE
    _SRC_ROOT = Path(src_root)
    _DST_ROOT = Path(dst_root)
    _THREADS = max(1, threads)
    _OVERWRITE = bool(overwrite)


def output_name(name: str) -> str:
    return name.replace(".npz", ".npy")


def _read_poses_raw(src: Path) -> np.ndarray:
    """Read just the ``poses`` member and raw-inflate it (no CRC check).

    An NPZ is a ZIP archive whose members are individual ``.npy`` blobs. We read
    the whole (small) file once, locate the ``poses`` local header, slice out its
    compressed bytes, and inflate with a raw DEFLATE call. This skips the CRC32
    pass and avoids decompressing the other metadata members.
    """
    with open(src, "rb") as fh:
        data = fh.read()

    with zipfile.ZipFile(io.BytesIO(data)) as z:
        try:
            info = z.getinfo("poses.npy")
        except KeyError:
            info = next(
                (i for i in z.infolist() if i.filename.startswith("poses")),
                None,
            )
            if info is None:
                raise KeyError("no 'poses' member in archive")

    off = info.header_offset
    # Local file header: 30 fixed bytes, then filename (n) and extra (m).
    n = struct.unpack("<H", data[off + 26:off + 28])[0]
    m = struct.unpack("<H", data[off + 28:off + 30])[0]
    start = off + 30 + n + m
    comp = data[start:start + info.compress_size]

    if info.compress_type == zipfile.ZIP_DEFLATED:
        raw = zlib.decompress(comp, -15)
    elif info.compress_type == zipfile.ZIP_STORED:
        raw = comp
    else:
        raise ValueError(f"unsupported compress_type {info.compress_type}")

    return npformat.read_array(io.BytesIO(raw), allow_pickle=False)


def _load_poses(src: Path) -> np.ndarray:
    try:
        return _read_poses_raw(src)
    except Exception:
        # Robust fallback for any unexpected archive layout.
        with np.load(src, allow_pickle=True) as d:
            return d["poses"]


def convert_one(name: str) -> tuple[str, str]:
    """Convert one NPZ to plain NPY. Returns (name, status)."""
    if _SRC_ROOT is None or _DST_ROOT is None:
        raise RuntimeError("worker was not initialized")

    src = _SRC_ROOT / name
    out = _DST_ROOT / output_name(name)

    if not _OVERWRITE:
        try:
            if out.stat().st_size > 0:
                return name, "skip"
        except FileNotFoundError:
            pass

    try:
        poses = _load_poses(src)

        if poses.dtype == object or poses.ndim != 3 or poses.shape[1:] != (133, 3):
            return name, f"malformed:dtype={poses.dtype},shape={poses.shape}"

        if poses.dtype != np.float32:
            poses = poses.astype(np.float32, copy=False)

        # Serialise in memory first so the network FS sees a single write().
        buf = io.BytesIO()
        npformat.write_array(buf, poses, allow_pickle=False)

        tmp = out.with_name(out.name + ".partial")
        with open(tmp, "wb") as f:
            f.write(buf.getbuffer())
        os.replace(tmp, out)
        return name, "ok"
    except Exception as e:
        return name, f"err:{type(e).__name__}:{e}"


def convert_batch(names: list[str]) -> tuple[dict[str, int], list[tuple[str, str]]]:
    """Convert a batch in one worker call.

    Uses a thread pool so reads/inflate/writes for different files overlap.
    """
    if _THREADS <= 1:
        results = [convert_one(name) for name in names]
    else:
        with ThreadPoolExecutor(_THREADS) as ex:
            results = list(ex.map(convert_one, names))

    counts: Counter[str] = Counter()
    errors: list[tuple[str, str]] = []
    for name, status in results:
        key = status if status in {"ok", "skip"} else status.split(":", 1)[0]
        counts[key] += 1
        if status not in {"ok", "skip"}:
            errors.append((name, status))
    return dict(counts), errors


def batched(names: list[str], batch_size: int) -> list[list[str]]:
    return [names[i:i + batch_size] for i in range(0, len(names), batch_size)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16,
                    help="Number of worker processes")
    ap.add_argument("--threads-per-worker", type=int, default=8,
                    help="I/O+inflate threads inside each worker process")
    ap.add_argument("--manifest", default="manifest.txt")
    ap.add_argument("--chunksize", type=int, default=4)
    ap.add_argument(
        "--task-batch-size", type=int, default=64,
        help="How many files each worker converts per multiprocessing task",
    )
    ap.add_argument("--errors-log", default="convert_errors.txt")
    ap.add_argument("--src", default=str(SRC), help="Directory containing *_poses.npz files")
    ap.add_argument("--dst", default=str(DST), help="Directory to write *_poses.npy files")
    ap.add_argument(
        "--prefilter-existing", action="store_true",
        help="Skip already-converted outputs in the parent before starting workers",
    )
    ap.add_argument(
        "--overwrite", action="store_true",
        help="Reconvert every manifest entry even if the .npy already exists "
        "(use after re-downloading reprocessed .npz so stale .npy are refreshed)",
    )
    args = ap.parse_args()

    if args.workers < 1:
        ap.error("--workers must be >= 1")
    if args.threads_per_worker < 1:
        ap.error("--threads-per-worker must be >= 1")
    if args.chunksize < 1:
        ap.error("--chunksize must be >= 1")
    if args.task_batch_size < 1:
        ap.error("--task-batch-size must be >= 1")

    src_root = Path(args.src)
    dst_root = Path(args.dst)
    dst_root.mkdir(parents=True, exist_ok=True)

    # Accept either .npz or .npy manifests; normalise to the .npz source name.
    names: list[str] = []
    with open(args.manifest) as f:
        for line in f:
            s = line.strip()
            if s.endswith("_poses.npz"):
                names.append(s)
            elif s.endswith("_poses.npy"):
                names.append(s[:-4] + ".npz")

    n_manifest = len(names)
    n_prefiltered = 0
    if args.prefilter_existing and not args.overwrite:
        # One directory scan instead of one stat() per manifest entry. On a
        # high-latency network FS, 1.6M individual stats can take tens of
        # minutes; a single scandir is O(existing outputs) and near-instant
        # when the destination is empty.
        before = len(names)
        existing: set[str] = set()
        try:
            with os.scandir(dst_root) as it:
                for entry in it:
                    if entry.name.endswith("_poses.npy") and entry.stat().st_size > 0:
                        existing.add(entry.name)
        except FileNotFoundError:
            pass
        if existing:
            names = [name for name in names if output_name(name) not in existing]
        n_prefiltered = before - len(names)

    batches = batched(names, args.task_batch_size)

    print(f"Manifest entries (npz): {n_manifest:,}")
    if args.prefilter_existing:
        print(f"Prefiltered existing:    {n_prefiltered:,}")
        print(f"Remaining to submit:     {len(names):,}")
    print(f"Workers:                {args.workers}")
    print(f"Threads per worker:     {args.threads_per_worker}")
    print(f"Task batch size:        {args.task_batch_size}")
    print(f"MP chunksize:           {args.chunksize}")
    print(f"Source:                 {src_root.resolve()}")
    print(f"Destination:            {dst_root.resolve()}")
    if not args.prefilter_existing:
        print("(Resumability is handled per-file inside the worker — no upfront scan.)")
    print(flush=True)

    t0 = time.perf_counter()
    counts = {"ok": 0, "skip": 0, "malformed": 0, "err": 0}
    errors: list[tuple[str, str]] = []

    with mp.Pool(
        args.workers,
        initializer=init_worker,
        initargs=(str(src_root), str(dst_root), args.threads_per_worker, args.overwrite),
    ) as pool:
        for batch_counts, batch_errors in tqdm(
            pool.imap_unordered(
                convert_batch,
                batches,
                chunksize=args.chunksize,
            ),
            total=len(batches),
            smoothing=0.05,
            mininterval=2.0,
            unit="batch",
        ):
            counts["ok"] += batch_counts.get("ok", 0)
            counts["skip"] += batch_counts.get("skip", 0)
            counts["malformed"] += batch_counts.get("malformed", 0)
            counts["err"] += sum(
                count for key, count in batch_counts.items()
                if key not in {"ok", "skip", "malformed"}
            )
            errors.extend(batch_errors)

    elapsed = time.perf_counter() - t0
    print()
    print(f"Done in {elapsed/60:.1f} min")
    print(f"  ok:        {counts['ok']:,}")
    print(f"  skip:      {counts['skip']:,}")
    print(f"  malformed: {counts['malformed']:,}")
    print(f"  err:       {counts['err']:,}")
    done = counts["ok"] + counts["skip"] + counts["malformed"] + counts["err"]
    if elapsed > 0:
        print(f"  rate:      {done/elapsed:,.0f} files/s")

    if errors:
        with open(args.errors_log, "w") as f:
            for name, status in errors:
                f.write(f"{name}\t{status}\n")
        print(f"Errors written to: {args.errors_log}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
