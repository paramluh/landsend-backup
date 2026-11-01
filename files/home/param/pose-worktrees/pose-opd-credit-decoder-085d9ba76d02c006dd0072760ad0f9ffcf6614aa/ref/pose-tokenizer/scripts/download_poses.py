#!/usr/bin/env python3
"""Export clip and pose GCS paths for a Stage 3 bucket to CSV, or download them.

Queries the database for clips matching the given bucket (or virtual bucket),
then writes a CSV with columns: clip_id, gcs_clip_path, gcs_pose_path.

If gcs_pose_path is NULL in the DB, it is derived from the clip path by
replacing .mp4 with _poses.npz.

By default, whenever the DB is queried the resulting CSV is saved to
``data/<bucket>_paths.csv`` (or the path given by ``-o``) so it can be
reused later via ``--from-csv`` without re-querying the database.

Usage:
    # CSV export (default behaviour - just queries DB and writes CSV)
    uv run python scripts/download_poses.py upperbody_ideal
    uv run python scripts/download_poses.py upperbody_ideal_no_captions_no_objects
    uv run python scripts/download_poses.py upperbody_ideal -o my_export.csv

    # Bulk download from DB query (CSV is also saved automatically)
    uv run python scripts/download_poses.py upperbody_ideal --download poses
    uv run python scripts/download_poses.py upperbody_ideal --download both --dest ./my_data
    uv run python scripts/download_poses.py upperbody_ideal --download clips -w 64

    # Skip the DB and download directly from a previously-saved CSV
    uv run python scripts/download_poses.py --from-csv data/upperbody_ideal_paths.csv \\
        --download poses --dest /mnt/somfs/pose_conditioning/

    uv run python scripts/download_poses.py --list-buckets
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dataclasses import dataclass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VIRTUAL_BUCKETS = {
    "upperbody_ideal_no_captions_no_objects": {
        "bucket": "upperbody_ideal",
        "join_videos": True,
        "where": (
            "(v.coarse_filter_has_captions = FALSE OR v.caption_temporal_pattern = 'intro_only') "
            "AND c.object_in_hand_detected IS NOT TRUE"
        ),
    },
    "upperbody_ideal_no_captions_with_objects": {
        "bucket": "upperbody_ideal",
        "join_videos": True,
        "where": (
            "(v.coarse_filter_has_captions = FALSE OR v.caption_temporal_pattern = 'intro_only') "
            "AND c.object_in_hand_detected = TRUE"
        ),
    },
    "upperbody_ideal_with_captions_no_objects": {
        "bucket": "upperbody_ideal",
        "join_videos": True,
        "where": (
            "v.coarse_filter_has_captions = TRUE "
            "AND v.caption_temporal_pattern = 'persistent' "
            "AND c.object_in_hand_detected IS NOT TRUE"
        ),
    },
}

@dataclass(frozen=True)
class DBConfig:
    """Postgres connection settings.

    Configure via env vars (preferred for secrets):
      - TIKTOK_DB_DSN (full psycopg2 DSN) OR
      - TIKTOK_DB_HOST, TIKTOK_DB_PORT, TIKTOK_DB_NAME, TIKTOK_DB_USER, TIKTOK_DB_PASSWORD
    """

    dsn: str | None = None
    host: str | None = None
    port: int = 5432
    dbname: str = "tiktok_pipeline"
    user: str = "postgres"
    password: str | None = None
    sslmode: str = "require"

    @classmethod
    def from_env(cls) -> DBConfig:
        dsn = os.getenv("TIKTOK_DB_DSN") or None
        host = os.getenv("TIKTOK_DB_HOST") or None
        port = int(os.getenv("TIKTOK_DB_PORT") or "5432")
        dbname = os.getenv("TIKTOK_DB_NAME") or "tiktok_pipeline"
        user = os.getenv("TIKTOK_DB_USER") or "postgres"
        password = os.getenv("TIKTOK_DB_PASSWORD") or None
        sslmode = os.getenv("TIKTOK_DB_SSLMODE") or "require"
        return cls(dsn=dsn, host=host, port=port, dbname=dbname, user=user, password=password, sslmode=sslmode)

    def is_configured(self) -> bool:
        return bool(self.dsn or self.host)


def connect(cfg: DBConfig | None = None):
    """Create a psycopg2 connection if configured; returns None otherwise."""
    cfg = cfg or DBConfig.from_env()
    if not cfg.is_configured():
        return None

    import psycopg2

    if cfg.dsn:
        return psycopg2.connect(cfg.dsn)
    return psycopg2.connect(
        host=cfg.host,
        port=cfg.port,
        dbname=cfg.dbname,
        user=cfg.user,
        password=cfg.password,
        sslmode=cfg.sslmode,
    )


def derive_pose_path(clip_path: str) -> str:
    """Derive the .npz pose path from a .mp4 clip path."""
    if clip_path.endswith(".mp4"):
        return clip_path[:-4] + "_poses.npz"
    return clip_path + "_poses.npz"


def export_paths(
    bucket_name: str,
    tags: list[str] | None = None,
    min_duration: float = 3.0,
) -> list[tuple[str, str, str, bool]]:
    """Return (clip_id, gcs_clip_path, gcs_pose_path, has_objects) for clips in a bucket.

    ``has_objects`` is True iff ``clips.object_in_hand_detected`` is TRUE.
    NULL is treated as False (i.e. "no objects detected").
    """

    conn = connect(DBConfig.from_env())
    if conn is None:
        print("ERROR: DB not configured. Set TIKTOK_DB_DSN.", file=sys.stderr)
        sys.exit(1)

    cur = conn.cursor()
    vb = VIRTUAL_BUCKETS.get(bucket_name)
    real_bucket = vb["bucket"] if vb else bucket_name
    needs_join = bool(vb and vb.get("join_videos"))

    if needs_join:
        base = """
            SELECT c.clip_id, c.gcs_clip_path, c.gcs_pose_path, c.object_in_hand_detected
            FROM clips c
            JOIN videos v ON c.video_id = v.video_id
            WHERE c.bucket = %s AND c.gcs_clip_path IS NOT NULL
        """
    else:
        base = """
            SELECT clip_id, gcs_clip_path, gcs_pose_path, object_in_hand_detected
            FROM clips
            WHERE bucket = %s AND gcs_clip_path IS NOT NULL
        """

    params: list = [real_bucket]

    if vb and vb.get("where"):
        base += f" AND {vb['where']}"

    if tags:
        col = "c.tags" if needs_join else "tags"
        base += f" AND {col} @> %s"
        params.append(tags)

    dur_col = "c.duration" if needs_join else "duration"
    base += f" AND ({dur_col} >= %s OR {dur_col} IS NULL)"
    params.append(min_duration)

    vid_col = "c.video_id" if needs_join else "video_id"
    base += f" ORDER BY {vid_col}, clip_id" if not needs_join else " ORDER BY c.video_id, c.clip_id"

    cur.execute(base, tuple(params))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    results = []
    for clip_id, clip_path, pose_path, has_objects in rows:
        if not pose_path:
            pose_path = derive_pose_path(clip_path)
        results.append((clip_id, clip_path, pose_path, bool(has_objects)))

    return results


CSV_HEADER = ["clip_id", "gcs_clip_path", "gcs_pose_path", "has_objects"]


def write_rows_csv(path: Path, rows: list[tuple[str, str, str, bool]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for clip_id, clip_path, pose_path, has_objects in rows:
            writer.writerow([clip_id, clip_path, pose_path, "true" if has_objects else "false"])


def read_rows_csv(path: Path) -> list[tuple[str, str, str, bool]]:
    rows: list[tuple[str, str, str, bool]] = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header != CSV_HEADER:
            print(
                f"WARNING: unexpected CSV header in {path}: {header}",
                file=sys.stderr,
            )
        for row in reader:
            if len(row) < 3:
                continue
            clip_id, clip_path, pose_path = row[0], row[1], row[2]
            if not pose_path:
                pose_path = derive_pose_path(clip_path)
            has_objects = False
            if len(row) >= 4:
                has_objects = row[3].strip().lower() in ("true", "1", "t", "yes")
            rows.append((clip_id, clip_path, pose_path, has_objects))
    return rows


def list_buckets():
    from pipeline.db import DBConfig, connect

    conn = connect(DBConfig.from_env())
    if conn is None:
        print("ERROR: DB not configured.", file=sys.stderr)
        sys.exit(1)

    cur = conn.cursor()
    cur.execute(
        "SELECT bucket, COUNT(*) FROM clips "
        "WHERE bucket IS NOT NULL AND gcs_clip_path IS NOT NULL "
        "GROUP BY bucket ORDER BY COUNT(*) DESC"
    )
    print(f"{'Bucket':<45} {'Clips':>8}")
    print("-" * 55)
    for row in cur.fetchall():
        print(f"{row[0]:<45} {row[1]:>8}")

    print(f"\n{'Virtual Buckets':<45}")
    print("-" * 55)
    for vb_name, vb_def in VIRTUAL_BUCKETS.items():
        cur.execute(
            f"""SELECT COUNT(*) FROM clips c
                JOIN videos v ON c.video_id = v.video_id
                WHERE c.bucket = %s AND c.gcs_clip_path IS NOT NULL
                AND {vb_def['where']}""",
            (vb_def["bucket"],),
        )
        count = cur.fetchone()[0]
        print(f"{vb_name:<45} {count:>8}")

    cur.close()
    conn.close()


def main():
    env_path = ROOT / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    val = val.strip().strip("'\"")
                    os.environ.setdefault(key.strip(), val)

    parser = argparse.ArgumentParser(
        description="Export clip (.mp4) and pose (.npz) GCS paths to CSV"
    )
    parser.add_argument("bucket", nargs="?", help="Bucket name (e.g. upperbody_ideal)")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output CSV path (default: data/<bucket>_paths.csv)",
    )
    parser.add_argument(
        "--from-csv", type=Path, default=None,
        help="Load rows from a previously exported CSV instead of querying the DB. "
             "When set, the DB is not contacted and 'bucket' becomes optional.",
    )
    parser.add_argument(
        "--no-csv", action="store_true",
        help="Don't save a CSV after querying the DB (only meaningful with --download).",
    )
    parser.add_argument("--tags", nargs="+", help="Filter by tags (e.g. --tags speech:active lang:en)")
    parser.add_argument("--min-duration", type=float, default=3.0, help="Minimum clip duration in seconds (default: 3.0)")
    parser.add_argument("--list-buckets", action="store_true", help="List available buckets and exit")
    parser.add_argument(
        "--download", choices=["poses", "clips", "both"], default=None,
        help="Download artifacts from GCS in parallel (default: CSV export only)",
    )
    parser.add_argument(
        "--dest", type=str, default=None,
        help="Local destination directory for --download mode (default: data/downloads/<bucket>)",
    )
    parser.add_argument(
        "-w", "--workers", type=int, default=64,
        help="Parallel download workers (default: 64)",
    )
    parser.add_argument(
        "--worker-type", choices=["thread", "process"], default="thread",
        help=(
            "google-cloud-storage transfer_manager worker type "
            "(default: thread; usually best for many small files)"
        ),
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help=(
            "Number of objects submitted per transfer_manager call "
            "(default: max(2000, workers * 16))"
        ),
    )
    parser.add_argument(
        "--checksum", choices=["auto", "md5", "crc32c", "none"], default="auto",
        help="Checksum mode passed to blob downloads (default: auto)",
    )
    parser.add_argument(
        "--single-shot-download", action="store_true",
        help="Force each object download to use a single request.",
    )
    parser.add_argument(
        "--num-shards", type=int, default=1,
        help=(
            "Split the work into N disjoint shards for multi-node fan-out. "
            "Run one process per node, each with a different --shard-index, "
            "all pointing --dest at the same shared filesystem."
        ),
    )
    parser.add_argument(
        "--shard-index", type=int, default=0,
        help="Which shard this process handles (0-based, < --num-shards).",
    )
    args = parser.parse_args()

    if args.num_shards < 1:
        parser.error("--num-shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        parser.error("--shard-index must satisfy 0 <= shard-index < num-shards")

    if args.list_buckets:
        list_buckets()
        return

    if args.from_csv:
        if not args.from_csv.exists():
            parser.error(f"--from-csv path does not exist: {args.from_csv}")
        print(f"Loading rows from {args.from_csv}...", file=sys.stderr)
        rows = read_rows_csv(args.from_csv)
        print(f"Loaded {len(rows)} clips from CSV", file=sys.stderr)
        bucket_label = args.bucket or args.from_csv.stem
    else:
        if not args.bucket:
            parser.error("bucket name required (or use --from-csv / --list-buckets)")

        print(
            f"Querying clips for bucket '{args.bucket}' "
            f"(min duration: {args.min_duration}s)...",
            file=sys.stderr,
        )
        rows = export_paths(args.bucket, tags=args.tags, min_duration=args.min_duration)
        print(f"Found {len(rows)} clips", file=sys.stderr)
        bucket_label = args.bucket

        if not args.no_csv and rows:
            csv_path = args.output or ROOT / "data" / f"{args.bucket}_paths.csv"
            write_rows_csv(csv_path, rows)
            print(f"Wrote {len(rows)} rows to {csv_path}", file=sys.stderr)
            print(
                f"  (re-run with --from-csv {csv_path} to skip the DB next time)",
                file=sys.stderr,
            )

    if not rows:
        print("Nothing to do.", file=sys.stderr)
        return

    if args.num_shards > 1:
        # Deterministic, balanced split by row order so every node selects a
        # disjoint slice from the same CSV without coordination.
        all_rows = rows
        rows = rows[args.shard_index :: args.num_shards]
        print(
            f"Shard {args.shard_index}/{args.num_shards}: handling "
            f"{len(rows)} of {len(all_rows)} clips",
            file=sys.stderr,
        )

    if args.download:
        from google.cloud import storage as gcs_storage
        from google.cloud.storage import transfer_manager

        dest_dir = Path(args.dest) if args.dest else ROOT / "data" / "downloads" / bucket_label
        dest_dir.mkdir(parents=True, exist_ok=True)

        gcs_paths: list[str] = []
        if args.download in ("clips", "both"):
            gcs_paths.extend(clip for _, clip, _, _ in rows)
        if args.download in ("poses", "both"):
            gcs_paths.extend(pose for _, _, pose, _ in rows)

        # Ignore zero-byte files, which are usually partial outputs left by an
        # interrupted transfer rather than valid pose archives.
        existing = {
            p.name for p in dest_dir.iterdir()
            if p.is_file() and p.stat().st_size > 0
        }
        to_download = [p for p in gcs_paths if p.rsplit("/", 1)[-1] not in existing]

        total = len(to_download)
        skipped = len(gcs_paths) - total
        workers = args.workers
        print(
            f"Downloading {total} files to {dest_dir}/ "
            f"({skipped} already exist, {workers} workers)",
            file=sys.stderr,
        )
        if not to_download:
            print("Nothing new to download.", file=sys.stderr)
            return

        client = gcs_storage.Client()

        # Group by GCS bucket (almost always one, but be safe)
        by_bucket: dict[str, list[str]] = {}
        for gcs_path in to_download:
            without_prefix = gcs_path[5:]
            bucket_name, _, key = without_prefix.partition("/")
            by_bucket.setdefault(bucket_name, []).append(key)

        downloaded = 0
        failed = 0
        downloaded_bytes = 0
        start = time.monotonic()
        chunk_size = args.batch_size or max(2000, workers * 16)
        checksum = None if args.checksum == "none" else args.checksum
        download_kwargs = {
            "checksum": checksum,
            "single_shot_download": args.single_shot_download,
        }

        first_chunk_checked = False
        for bucket_name, keys in by_bucket.items():
            bucket = client.bucket(bucket_name)

            for chunk_start in range(0, len(keys), chunk_size):
                chunk = keys[chunk_start : chunk_start + chunk_size]
                blob_file_pairs = [
                    (bucket.blob(key), str(dest_dir / key.rsplit("/", 1)[-1]))
                    for key in chunk
                ]

                results = transfer_manager.download_many(
                    blob_file_pairs,
                    download_kwargs=download_kwargs,
                    max_workers=workers,
                    worker_type=args.worker_type,
                    raise_exception=False,
                    skip_if_exists=True,
                )

                chunk_ok = 0
                chunk_err = 0
                errors_shown = 0
                for (blob, local_path), result in zip(blob_file_pairs, results):
                    if isinstance(result, Exception):
                        failed += 1
                        chunk_err += 1
                        if errors_shown < 3:
                            print(
                                f"    FAIL gs://{bucket_name}/{blob.name} -> {local_path}: {result!r}",
                                file=sys.stderr,
                            )
                            errors_shown += 1
                        Path(local_path).unlink(missing_ok=True)
                    else:
                        downloaded += 1
                        chunk_ok += 1
                        try:
                            downloaded_bytes += Path(local_path).stat().st_size
                        except FileNotFoundError:
                            pass

                elapsed = time.monotonic() - start
                done = downloaded + failed
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                mb_s = (downloaded_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0
                print(
                    f"  [{done:>7}/{total}] {downloaded} ok, {failed} err "
                    f"| {rate:.0f} files/s | ~{mb_s:.0f} MB/s | ~{eta:.0f}s left",
                    file=sys.stderr,
                )

                if not first_chunk_checked:
                    first_chunk_checked = True
                    if chunk_ok == 0 and chunk_err > 0:
                        print(
                            f"\nERROR: first chunk of {chunk_err} downloads all failed. "
                            f"Aborting early — see exceptions above. "
                            f"Check GOOGLE_APPLICATION_CREDENTIALS, bucket permissions, "
                            f"and that --dest is writable.",
                            file=sys.stderr,
                        )
                        sys.exit(1)

        elapsed = time.monotonic() - start
        print(
            f"\nDone in {elapsed:.1f}s: {downloaded} downloaded, {failed} failed",
            file=sys.stderr,
        )
        return


if __name__ == "__main__":
    main()
