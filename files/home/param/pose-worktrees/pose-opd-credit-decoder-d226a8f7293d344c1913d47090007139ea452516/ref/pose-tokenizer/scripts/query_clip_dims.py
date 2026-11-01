#!/usr/bin/env python3
"""Query the DB for clip dimensions (height, width, video_id, duration) for all poses on disk.

Reads clip_ids from a manifest file (one filename per line), batches them into
SQL queries against the ``clips`` table, and writes a CSV compatible with
``PoseDataset(dims_csv=...)``. The ``duration`` column (seconds, from ``clips.duration``)
is included for histograms / planning ``max_frames`` (see ``plot_clip_length_histogram.py``).

Usage:
    uv run python scripts/query_clip_dims.py manifest.txt -o dims.csv
    uv run python scripts/query_clip_dims.py manifest.txt          # default: dims.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BATCH_SIZE = 5000


@dataclass(frozen=True)
class DBConfig:
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
    cfg = cfg or DBConfig.from_env()
    if not cfg.is_configured():
        return None
    import psycopg2
    if cfg.dsn:
        return psycopg2.connect(cfg.dsn)
    return psycopg2.connect(
        host=cfg.host, port=cfg.port, dbname=cfg.dbname,
        user=cfg.user, password=cfg.password, sslmode=cfg.sslmode,
    )


def clip_id_from_filename(fname: str) -> str:
    """``150131099837509632_00-01.27_to_00-15.00_poses.npz`` → ``150131099837509632_00-01.27_to_00-15.00``"""
    stem = fname.removesuffix(".npz")
    if stem.endswith("_poses"):
        stem = stem.removesuffix("_poses")
    return stem


def main() -> None:
    env_path = ROOT / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    val = val.strip().strip("'\"")
                    os.environ.setdefault(key.strip(), val)

    parser = argparse.ArgumentParser(description="Query clip dims from DB for downloaded pose files")
    parser.add_argument("manifest", type=Path, help="Manifest .txt (one filename per line)")
    parser.add_argument("-o", "--output", type=Path, default=ROOT / "dims.csv", help="Output CSV (default: dims.csv)")
    args = parser.parse_args()

    with open(args.manifest) as f:
        filenames = [l.strip() for l in f if l.strip()]

    clip_id_to_stem: dict[str, str] = {}
    for fname in filenames:
        cid = clip_id_from_filename(fname)
        clip_stem = fname.removesuffix(".npz")
        clip_id_to_stem[cid] = clip_stem

    all_clip_ids = list(clip_id_to_stem.keys())
    print(f"Manifest has {len(all_clip_ids)} clips", file=sys.stderr)

    conn = connect(DBConfig.from_env())
    if conn is None:
        print("ERROR: DB not configured. Set TIKTOK_DB_DSN or TIKTOK_DB_HOST.", file=sys.stderr)
        sys.exit(1)

    cur = conn.cursor()
    # clip_id → (video_id, height, width, duration_seconds_or_None)
    results: dict[str, tuple[str, int, int, float | None]] = {}
    start = time.monotonic()

    for i in range(0, len(all_clip_ids), BATCH_SIZE):
        batch = all_clip_ids[i : i + BATCH_SIZE]
        placeholders = ",".join(["%s"] * len(batch))
        cur.execute(
            f"SELECT clip_id, video_id, height, width, duration FROM clips "
            f"WHERE clip_id IN ({placeholders})",
            batch,
        )
        for clip_id, video_id, h, w, dur in cur.fetchall():
            dur_s: float | None
            if dur is None:
                dur_s = None
            elif hasattr(dur, "total_seconds"):
                dur_s = float(dur.total_seconds())
            else:
                dur_s = float(dur)
            results[clip_id] = (video_id, h, w, dur_s)

        done = min(i + BATCH_SIZE, len(all_clip_ids))
        elapsed = time.monotonic() - start
        print(f"  Queried {done}/{len(all_clip_ids)} ({len(results)} found, {elapsed:.1f}s)", file=sys.stderr)

    cur.close()
    conn.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    matched = 0
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["clip_id", "video_id", "height", "width", "duration", "clip_stem"])
        for cid in all_clip_ids:
            if cid in results:
                vid, h, w, dur = results[cid]
                writer.writerow([cid, vid, h, w, "" if dur is None else dur, clip_id_to_stem[cid]])
                matched += 1

    elapsed = time.monotonic() - start
    missing = len(all_clip_ids) - matched
    print(f"\nWrote {matched} rows to {args.output} ({missing} clips not found in DB, {elapsed:.1f}s)", file=sys.stderr)


if __name__ == "__main__":
    main()
