#!/usr/bin/env python3
"""Export per-clip dimensions and fps for a Stage 3 bucket to dims.csv.

Mirrors the DB-connection pattern from `download_poses.py`. The output CSV
has columns ``clip_stem,width,height,fps`` and is consumed by
``pose_tokenizer/data/dataset.py`` via its dims_csv argument.

The ``clip_stem`` matches manifest entries minus the ``_poses.npy`` suffix,
e.g. manifest line ``7615522494455368982_00-00.00_to_00-04.70_poses.npy``
→ stem ``7615522494455368982_00-00.00_to_00-04.70``.

Usage:
    # Requires TIKTOK_DB_DSN or TIKTOK_DB_HOST/PASSWORD/... env vars
    # (same connection pattern as scripts/download_poses.py)
    uv run python scripts/export_dims.py upperbody_ideal
    uv run python scripts/export_dims.py upperbody_ideal -o dims.csv

If you want to be sure the bucket matches what you trained on, you can
pass the virtual buckets defined in download_poses.py too.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Reuse the exact connection-and-bucket plumbing from download_poses.py so
# DB creds, bucket filters and join logic stay consistent.
from scripts.download_poses import (  # noqa: E402
    DBConfig, VIRTUAL_BUCKETS, connect,
)


def stem_from_pose_path(pose_path: str) -> str:
    """Derive the clip_stem (manifest key) from gcs pose/clip path."""
    base = pose_path.rsplit("/", 1)[-1]
    for suf in ("_poses.npz", "_poses.npy", ".npz", ".npy", ".mp4"):
        if base.endswith(suf):
            return base[: -len(suf)]
    return base


def export_dims(bucket_name: str, min_duration: float = 3.0) -> list[tuple[str, float, float, float]]:
    """Return (clip_stem, width, height, fps) for clips in *bucket_name*."""
    conn = connect(DBConfig.from_env())
    if conn is None:
        print("ERROR: DB not configured. Set TIKTOK_DB_DSN or TIKTOK_DB_HOST/...", file=sys.stderr)
        sys.exit(1)

    cur = conn.cursor()
    vb = VIRTUAL_BUCKETS.get(bucket_name)
    real_bucket = vb["bucket"] if vb else bucket_name
    needs_join = bool(vb and vb.get("join_videos"))

    # Per-clip dim columns. Adjust here if the schema uses different names.
    if needs_join:
        base = """
            SELECT c.clip_id, c.gcs_pose_path, c.gcs_clip_path,
                   c.orig_width, c.orig_height, c.fps
            FROM clips c
            JOIN videos v ON c.video_id = v.video_id
            WHERE c.bucket = %s AND c.gcs_clip_path IS NOT NULL
        """
    else:
        base = """
            SELECT clip_id, gcs_pose_path, gcs_clip_path,
                   orig_width, orig_height, fps
            FROM clips
            WHERE bucket = %s AND gcs_clip_path IS NOT NULL
        """

    params: list = [real_bucket]
    if vb and vb.get("where"):
        base += f" AND {vb['where']}"

    dur_col = "c.duration" if needs_join else "duration"
    base += f" AND ({dur_col} >= %s OR {dur_col} IS NULL)"
    params.append(min_duration)

    print(f"Querying clips for bucket '{bucket_name}' (min duration: {min_duration}s)...", file=sys.stderr)
    cur.execute(base, tuple(params))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    print(f"Got {len(rows):,} rows", file=sys.stderr)

    out: list[tuple[str, float, float, float]] = []
    missing_dims = 0
    for clip_id, pose_path, clip_path, w, h, fps in rows:
        path_for_stem = pose_path or clip_path
        if not path_for_stem:
            continue
        stem = stem_from_pose_path(path_for_stem)
        # Skip rows where dims are unset (preferably the DB has them all)
        if w is None or h is None:
            missing_dims += 1
            continue
        out.append((stem, float(w), float(h), float(fps) if fps is not None else 30.0))
    if missing_dims:
        print(f"  WARNING: {missing_dims:,} rows skipped (NULL width or height)", file=sys.stderr)
    return out


def main() -> int:
    env_path = ROOT / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    val = val.strip().strip("'\"")
                    os.environ.setdefault(key.strip(), val)

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bucket", help="Bucket name (e.g. upperbody_ideal)")
    p.add_argument("-o", "--output", type=Path, default=Path("dims.csv"))
    p.add_argument("--min-duration", type=float, default=3.0)
    args = p.parse_args()

    rows = export_dims(args.bucket, min_duration=args.min_duration)
    if not rows:
        print("Nothing to write.", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["clip_stem", "width", "height", "fps"])
        writer.writerows(rows)
    print(f"Wrote {len(rows):,} rows to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
