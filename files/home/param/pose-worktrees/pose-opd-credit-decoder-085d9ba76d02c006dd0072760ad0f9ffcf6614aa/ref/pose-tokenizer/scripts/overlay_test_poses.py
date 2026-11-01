#!/usr/bin/env python3
"""Overlay precomputed test_set pose .npz files on their source videos.

Sanity-check utility for the regenerated test_set: reads
``test_set/actor_*/segment_*_poses.npz`` (already-computed RTMW-X + One
Euro Filter poses), draws the standard skeleton over each frame of the
matching ``.mp4``, and writes the result to ``pose_overlays/test_set/``.

Reuses ``overlay_skeleton`` from ``scripts/rtmwx_overlay.py`` for visual
consistency with the live-detection overlay videos.

Usage:
    # Default: 6 random clips
    uv run --no-sync python scripts/overlay_test_poses.py

    # Specific clip
    uv run --no-sync python scripts/overlay_test_poses.py --only actor_1/segment_01.mp4

    # Every clip in test_set
    uv run --no-sync python scripts/overlay_test_poses.py --all
"""
from __future__ import annotations

import argparse
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

# Allow `import rtmwx_overlay` without scripts/ being a package.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rtmwx_overlay import overlay_skeleton  # noqa: E402


def load_poses_as_xy_conf(
    npz_path: Path,
) -> tuple[np.ndarray, np.ndarray, float, int, int]:
    """Read a test_set ``_poses.npz`` and return (xy, conf, fps, h, w).

    The npz schema is ``(T, 133, 3)`` with column layout ``(y_pix, x_pix, conf)``
    (see ``pose_tokenizer.data.dataset.load_clip``). ``overlay_skeleton``
    expects ``(x, y)`` per joint, so we swap.
    """
    d = np.load(npz_path)
    poses = d["poses"]                       # (T, 133, 3)
    fps = float(d["fps"])
    h = int(float(d["orig_h"]))
    w = int(float(d["orig_w"]))

    T = poses.shape[0]
    xy = np.empty((T, 133, 2), dtype=np.float32)
    xy[..., 0] = poses[..., 1]               # x
    xy[..., 1] = poses[..., 0]               # y
    conf = poses[..., 2].astype(np.float32)
    return xy, conf, fps, h, w


def render_overlay(
    video: Path,
    npz: Path,
    out: Path,
    *,
    conf_thresh: float = 0.3,
) -> None:
    xy, conf, _, npz_h, npz_w = load_poses_as_xy_conf(npz)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {video}")
    T_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if (h, w) != (npz_h, npz_w):
        print(f"  WARN: video {h}x{w} != npz {npz_h}x{npz_w}; skeleton may misalign",
              flush=True)

    n_frames = min(T_video, xy.shape[0])

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = tmp.name
    tmp.close()

    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))

    for fi in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        img = overlay_skeleton(frame, xy[fi], conf[fi], conf_thresh=conf_thresh)
        writer.write(img)

    cap.release()
    writer.release()

    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", tmp_path,
            "-i", str(video),
            "-map", "0:v",
            "-map", "1:a?",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-c:a", "aac",
            "-movflags", "+faststart",
            str(out),
        ],
        check=True, capture_output=True,
    )
    Path(tmp_path).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("test_set"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("pose_overlays/test_set"))
    parser.add_argument("--n", type=int, default=6,
                        help="Random sample size when neither --all nor --only is set")
    parser.add_argument("--all", action="store_true",
                        help="Render every clip in --root")
    parser.add_argument("--only", type=str, default=None,
                        help="Relative path under --root, e.g. actor_1/segment_01.mp4")
    parser.add_argument("--conf-thresh", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.only:
        videos = [args.root / args.only]
    else:
        videos = sorted(args.root.glob("actor_*/segment_*.mp4"))
        if not args.all and len(videos) > args.n:
            rng = random.Random(args.seed)
            videos = sorted(rng.sample(videos, args.n))

    if not videos:
        print(f"No videos found under {args.root}", file=sys.stderr)
        sys.exit(1)

    print(f"Rendering {len(videos)} overlay(s) to {args.out_dir}/", flush=True)

    n_ok = n_skip = 0
    for i, video in enumerate(videos, 1):
        npz = video.with_name(video.stem + "_poses.npz")
        rel = video.relative_to(args.root)
        out_name = f"{rel.parent.name}__{rel.stem}_overlay.mp4"
        out = args.out_dir / out_name
        print(f"[{i}/{len(videos)}] {rel}  ->  {out}", flush=True)
        if not npz.exists():
            print(f"  SKIP: missing {npz}", flush=True)
            n_skip += 1
            continue
        try:
            render_overlay(video, npz, out, conf_thresh=args.conf_thresh)
            n_ok += 1
            print(f"  OK", flush=True)
        except Exception as e:
            print(f"  FAIL: {e!r}", flush=True)
            n_skip += 1

    print(f"\nDone: {n_ok} written, {n_skip} skipped/failed. "
          f"Outputs in {args.out_dir}/", flush=True)


if __name__ == "__main__":
    main()
