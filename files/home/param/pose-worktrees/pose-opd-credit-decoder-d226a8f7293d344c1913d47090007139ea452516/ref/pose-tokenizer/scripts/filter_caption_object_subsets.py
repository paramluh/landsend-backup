#!/usr/bin/env python3
"""Intersect two DB subsets (caption / object combinations) with the locally
available upperbody_ideal clips.

Usage:
    uv run python scripts/filter_caption_object_subsets.py
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.download_poses import export_paths

LOCAL_CSV = ROOT / "upperbody_ideal_virtual_bucket_gcs_paths.csv"

SUBSETS = [
    ("upperbody_ideal_with_captions_no_objects", ROOT / "with_captions_no_objects.csv"),
    ("upperbody_ideal_no_captions_with_objects", ROOT / "no_captions_with_objects.csv"),
]


def load_env():
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def main():
    load_env()

    with open(LOCAL_CSV) as f:
        local_paths = {row["gcs_clip_path"] for row in csv.DictReader(f)}
    print(f"Local set: {len(local_paths):,} clips", file=sys.stderr)

    for bucket, out_path in SUBSETS:
        print(f"\nQuerying {bucket}...", file=sys.stderr)
        rows = export_paths(bucket)
        kept = [r for r in rows if r[1] in local_paths]
        print(f"  DB returned {len(rows):,}, kept {len(kept):,} present locally",
              file=sys.stderr)

        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["clip_id", "gcs_clip_path", "gcs_pose_path", "has_objects"])
            for clip_id, clip_path, pose_path, has_objects in kept:
                w.writerow([clip_id, clip_path, pose_path, "true" if has_objects else "false"])
        print(f"  → {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
