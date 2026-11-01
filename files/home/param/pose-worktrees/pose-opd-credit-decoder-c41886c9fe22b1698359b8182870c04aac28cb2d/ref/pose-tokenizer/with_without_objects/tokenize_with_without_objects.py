#!/usr/bin/env python3
"""Run pose tokenization for with/without-object splits.

This is a thin wrapper over ``scripts/tokenize_chunks.py`` that runs it twice:
1) with objects
2) without objects
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESOURCES = Path(__file__).resolve().parent / "resources"
DEFAULT_OUT_ROOT = Path(__file__).resolve().parent / "tokenized_pose_splits"
TOKENIZE_SCRIPT = ROOT / "scripts" / "tokenize_chunks.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Pose tokenizer checkpoint path or HF repo id.",
    )
    parser.add_argument(
        "--pose-dir",
        required=True,
        help="Directory containing <clip_id>_poses.npy files referenced by manifests.",
    )
    parser.add_argument(
        "--resources-dir",
        default=str(DEFAULT_RESOURCES),
        help=f"Resources dir with manifests (default: {DEFAULT_RESOURCES}).",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUT_ROOT),
        help=f"Output root for split token files (default: {DEFAULT_OUT_ROOT}).",
    )
    parser.add_argument(
        "--dims-csv",
        default="",
        help="Optional dims CSV passed through to tokenize_chunks.py.",
    )
    parser.add_argument(
        "--skip-with-objects",
        action="store_true",
        help="Skip tokenizing the with-objects split.",
    )
    parser.add_argument(
        "--skip-without-objects",
        action="store_true",
        help="Skip tokenizing the without-objects split.",
    )
    parser.add_argument(
        "--extra-args",
        default="",
        help=(
            "Extra args appended to tokenize_chunks.py, quoted as one string. "
            "Example: \"--format parquet --num-workers 8 --batch-size 128\""
        ),
    )
    return parser.parse_args()


def _run_one(
    split_name: str,
    *,
    checkpoint: str,
    pose_dir: str,
    manifest: Path,
    output_dir: Path,
    dims_csv: str,
    extra_args: str,
) -> None:
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest}")

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(TOKENIZE_SCRIPT),
        "--checkpoint",
        checkpoint,
        "--manifest",
        str(manifest),
        "--pose-dir",
        pose_dir,
        "--output-dir",
        str(output_dir),
        "--format",
        "parquet",
    ]
    if dims_csv:
        cmd.extend(["--dims-csv", dims_csv])
    if extra_args:
        cmd.extend(shlex.split(extra_args))

    print(f"\n=== Tokenizing split: {split_name} ===")
    print(" ".join(shlex.quote(part) for part in cmd))
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def main() -> int:
    args = parse_args()
    resources_dir = Path(args.resources_dir).resolve()
    output_root = Path(args.output_root).resolve()
    with_manifest = resources_dir / "with_objects_pose_manifest.txt"
    without_manifest = resources_dir / "without_objects_pose_manifest.txt"

    if not args.skip_with_objects:
        _run_one(
            "with_objects",
            checkpoint=args.checkpoint,
            pose_dir=args.pose_dir,
            manifest=with_manifest,
            output_dir=output_root / "with_objects",
            dims_csv=args.dims_csv,
            extra_args=args.extra_args,
        )
    if not args.skip_without_objects:
        _run_one(
            "without_objects",
            checkpoint=args.checkpoint,
            pose_dir=args.pose_dir,
            manifest=without_manifest,
            output_dir=output_root / "without_objects",
            dims_csv=args.dims_csv,
            extra_args=args.extra_args,
        )

    print("\nDone.")
    print(f"outputs_root={output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
