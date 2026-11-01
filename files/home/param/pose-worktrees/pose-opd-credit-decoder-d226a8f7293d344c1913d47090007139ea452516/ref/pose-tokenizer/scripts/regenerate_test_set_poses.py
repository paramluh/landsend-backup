#!/usr/bin/env python3
"""Recompute test_set pose .npz files with RTMW-X + One Euro Filter.

Walks ``test_set/actor_*/segment_*.mp4``, runs RTMW-X (rtmlib PoseTracker) on
every frame, applies confidence-aware One Euro smoothing, and overwrites the
matching ``segment_*_poses.npz`` file in place.

The output schema matches what ``pose_tokenizer.data.dataset.load_clip``
expects:

    poses    : (T, 133, 3)  float32   layout = (y_pixel, x_pixel, confidence)
    orig_h   : ()           float32   original frame height
    orig_w   : ()           float32   original frame width
    fps      : ()           float32   source video fps

NOTE: rtmlib returns keypoints as ``(x, y)`` — we swap to ``(y, x)`` because
``load_clip`` divides column 0 by orig_h and column 1 by orig_w.

Run with the GPU build of onnxruntime active. The dev group installs both
``onnxruntime`` (CPU) and ``onnxruntime-gpu`` which collide; if
``ort.get_available_providers()`` doesn't include ``CUDAExecutionProvider``,
uninstall the CPU one (``uv pip uninstall onnxruntime``) first.

Usage:
    uv run --no-sync python scripts/regenerate_test_set_poses.py
    uv run --no-sync python scripts/regenerate_test_set_poses.py --root test_set --device cuda
    uv run --no-sync python scripts/regenerate_test_set_poses.py --only actor_1/segment_01.mp4
"""
from __future__ import annotations

import argparse
import ctypes
import glob
import os
import site
import sys
import time
from pathlib import Path


def _preload_cuda12_libs() -> list[str]:
    """Preload the pip-installed nvidia-*-cu12 shared libs so onnxruntime can find them.

    System CUDA may be 13.x (which onnxruntime-gpu 1.26 cannot use). We
    bundle 12.x via the ``nvidia-*-cu12`` wheels and dlopen them here so
    onnxruntime's CUDAExecutionProvider can load.
    """
    loaded: list[str] = []
    sp_dirs = site.getsitepackages() + [site.getusersitepackages()]
    seen: set[str] = set()
    for sp in sp_dirs:
        for lib in sorted(glob.glob(os.path.join(sp, "nvidia/*/lib/*.so*"))):
            base = os.path.basename(lib)
            if base in seen:
                continue
            seen.add(base)
            try:
                ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                loaded.append(lib)
            except OSError:
                pass
    return loaded


_PRELOADED = _preload_cuda12_libs()

import cv2  # noqa: E402
import numpy as np  # noqa: E402


# ---------------------------------------------------------------------------
# One Euro Filter
# ---------------------------------------------------------------------------

class OneEuroFilter:
    """Vectorized One Euro Filter (Casiez et al., 2012).

    Adaptive low-pass filter: low cutoff when signal is slow (kills jitter),
    high cutoff when signal is fast (preserves intentional motion).

    Holds internal state shaped like the first sample it sees, so it works on
    any per-frame array (e.g. ``(133, 2)`` keypoints).

    Args:
        freq:       Sampling rate of the input (Hz, = video fps).
        min_cutoff: Cutoff frequency for slow motion (lower = smoother).
        beta:       Cutoff slope vs. speed (higher = follows fast motion more
                    aggressively).
        d_cutoff:   Cutoff used when smoothing the derivative estimate.
    """

    def __init__(
        self,
        freq: float = 30.0,
        min_cutoff: float = 1.0,
        beta: float = 0.05,
        d_cutoff: float = 1.0,
    ):
        self.freq = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev: np.ndarray | None = None
        self.dx_prev: np.ndarray | None = None

    @staticmethod
    def _alpha(t_e: float, cutoff: np.ndarray | float) -> np.ndarray | float:
        r = 2.0 * np.pi * cutoff * t_e
        return r / (r + 1.0)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32, copy=False)
        if self.x_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            return x.copy()

        t_e = 1.0 / self.freq

        a_d = self._alpha(t_e, self.d_cutoff)
        dx = (x - self.x_prev) * self.freq
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._alpha(t_e, cutoff)
        x_hat = a * x + (1.0 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


def smooth_sequence(
    keypoints: np.ndarray,
    scores: np.ndarray,
    *,
    fps: float,
    min_cutoff: float,
    beta: float,
    conf_thresh: float,
    conf_decay: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """Confidence-aware One Euro smoothing.

    1. Joints below ``conf_thresh`` in a frame are held at the previous
       frame's position (avoids One Euro reacting to detection dropouts as
       if they were fast motion). Confidence decays by ``conf_decay`` per
       held frame.
    2. The cleaned sequence is passed through ``OneEuroFilter``.

    Args:
        keypoints: (T, 133, 2)  in (x, y) pixel coords.
        scores:    (T, 133)     raw rtmlib confidences.

    Returns:
        smoothed_xy: (T, 133, 2)
        smoothed_sc: (T, 133)
    """
    T, J, D = keypoints.shape
    assert D == 2

    filled_xy = keypoints.astype(np.float32, copy=True)
    filled_sc = scores.astype(np.float32, copy=True)

    for t in range(1, T):
        low = filled_sc[t] < conf_thresh
        had = filled_sc[t - 1] >= conf_thresh
        carry = low & had
        if carry.any():
            filled_xy[t, carry] = filled_xy[t - 1, carry]
            filled_sc[t, carry] = filled_sc[t - 1, carry] * conf_decay

    oef = OneEuroFilter(freq=fps, min_cutoff=min_cutoff, beta=beta)
    smoothed = np.empty_like(filled_xy)
    for t in range(T):
        smoothed[t] = oef(filled_xy[t])

    return smoothed, filled_sc


# ---------------------------------------------------------------------------
# Pose extraction
# ---------------------------------------------------------------------------

def _pick_best_person(
    keypoints_list: list[np.ndarray] | np.ndarray,
    scores_list: list[np.ndarray] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Return the (133, 2) / (133,) tracks for the highest-mean-confidence person."""
    if keypoints_list is None or len(keypoints_list) == 0:
        return None, None
    if len(keypoints_list) == 1:
        return np.asarray(keypoints_list[0]), np.asarray(scores_list[0])
    best = int(np.argmax([np.asarray(s).mean() for s in scores_list]))
    return np.asarray(keypoints_list[best]), np.asarray(scores_list[best])


def extract_raw_poses(
    video_path: Path,
    pose_tracker,
    progress_prefix: str = "",
) -> tuple[np.ndarray, np.ndarray, float, int, int]:
    """Run RTMW-X per-frame; return raw (T,133,2) xy + (T,133) scores + meta.

    Missing detections are filled with NaN for xy and 0 for scores.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {video_path}")
    T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pose_tracker.reset()

    all_xy: list[np.ndarray] = []
    all_sc: list[np.ndarray] = []

    fi = 0
    t0 = time.monotonic()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        kps, scs = pose_tracker(frame)
        best_kp, best_sc = _pick_best_person(kps, scs)
        if best_kp is None:
            all_xy.append(np.full((133, 2), np.nan, dtype=np.float32))
            all_sc.append(np.zeros(133, dtype=np.float32))
        else:
            all_xy.append(best_kp.astype(np.float32))
            all_sc.append(best_sc.astype(np.float32))

        fi += 1
        if fi % 100 == 0:
            elapsed = time.monotonic() - t0
            rate = fi / max(elapsed, 1e-9)
            print(f"  {progress_prefix}{fi}/{T} frames  ({rate:.1f} fps)", flush=True)

    cap.release()

    xy = np.stack(all_xy, axis=0) if all_xy else np.zeros((0, 133, 2), dtype=np.float32)
    sc = np.stack(all_sc, axis=0) if all_sc else np.zeros((0, 133), dtype=np.float32)
    return xy, sc, fps, h, w


def save_compatible_npz(
    out_path: Path,
    xy: np.ndarray,   # (T, 133, 2) in (x, y) pixel coords
    sc: np.ndarray,   # (T, 133)
    *,
    orig_h: int,
    orig_w: int,
    fps: float,
) -> None:
    """Save in the schema load_clip() expects: (T, 133, 3) with (y, x, conf)."""
    T, J, D = xy.shape
    assert D == 2 and J == 133

    # rtmlib emits (x, y) — load_clip divides col0 by orig_h, col1 by orig_w,
    # so we must put y in col 0 and x in col 1.
    poses = np.empty((T, J, 3), dtype=np.float32)
    poses[..., 0] = xy[..., 1]  # y
    poses[..., 1] = xy[..., 0]  # x
    poses[..., 2] = sc

    # NaN positions (no detection) get zeroed so load_clip's per-frame
    # shoulder-width logic doesn't blow up. Confidence is already 0 there.
    nan_mask = ~np.isfinite(poses[..., 0]) | ~np.isfinite(poses[..., 1])
    if nan_mask.any():
        poses[nan_mask, 0] = 0.0
        poses[nan_mask, 1] = 0.0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        poses=poses,
        orig_h=np.float32(orig_h),
        orig_w=np.float32(orig_w),
        fps=np.float32(fps),
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def list_test_videos(root: Path) -> list[Path]:
    return sorted(root.glob("actor_*/segment_*.mp4"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("test_set"),
                        help="Directory containing actor_*/segment_*.mp4 (default: test_set)")
    parser.add_argument("--mode", default="balanced",
                        choices=["performance", "balanced", "lightweight"])
    parser.add_argument("--backend", default="onnxruntime")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--det-frequency", type=int, default=1,
                        help="Re-detect bounding boxes every N frames (1 = every frame)")
    parser.add_argument("--min-cutoff", type=float, default=1.0,
                        help="One Euro min_cutoff (lower = smoother)")
    parser.add_argument("--beta", type=float, default=0.05,
                        help="One Euro beta (higher = react harder to fast motion)")
    parser.add_argument("--conf-thresh", type=float, default=0.1,
                        help="Joints below this conf in a frame are held from previous frame")
    parser.add_argument("--only", type=str, default=None,
                        help="Process only this relative path under --root, e.g. actor_1/segment_01.mp4")
    parser.add_argument("--dry-run", action="store_true",
                        help="List the files that would be processed and exit")
    args = parser.parse_args()

    import onnxruntime as ort
    providers = ort.get_available_providers()
    print(f"preloaded {len(_PRELOADED)} CUDA libs from nvidia-*-cu12 wheels",
          flush=True)
    print(f"onnxruntime providers: {providers}", flush=True)
    if args.device == "cuda" and "CUDAExecutionProvider" not in providers:
        print("ERROR: --device cuda requested but CUDAExecutionProvider is not "
              "available. Run `uv pip uninstall onnxruntime` to remove the CPU "
              "build, leaving only onnxruntime-gpu.", file=sys.stderr)
        sys.exit(2)

    if not args.root.is_dir():
        print(f"ERROR: --root {args.root} is not a directory", file=sys.stderr)
        sys.exit(2)

    if args.only:
        videos = [args.root / args.only]
        if not videos[0].exists():
            print(f"ERROR: {videos[0]} does not exist", file=sys.stderr)
            sys.exit(2)
    else:
        videos = list_test_videos(args.root)

    if not videos:
        print(f"No videos found under {args.root}/actor_*/segment_*.mp4", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(videos)} videos under {args.root}/", flush=True)
    if args.dry_run:
        for v in videos:
            print(f"  {v}")
        return

    print(f"Initializing RTMW-X (mode={args.mode}, det_freq={args.det_frequency}, "
          f"backend={args.backend}, device={args.device})", flush=True)
    from rtmlib import PoseTracker, Wholebody
    pose_tracker = PoseTracker(
        Wholebody,
        det_frequency=args.det_frequency,
        to_openpose=False,
        mode=args.mode,
        backend=args.backend,
        device=args.device,
    )

    print(f"One Euro Filter: min_cutoff={args.min_cutoff}, beta={args.beta}, "
          f"conf_thresh={args.conf_thresh}", flush=True)

    n_ok = 0
    n_skip = 0
    t_total = time.monotonic()

    for idx, video in enumerate(videos, 1):
        out_path = video.with_name(video.stem + "_poses.npz")
        print(f"\n[{idx}/{len(videos)}] {video.relative_to(args.root)}  ->  "
              f"{out_path.relative_to(args.root)}", flush=True)

        try:
            t0 = time.monotonic()
            xy, sc, fps, h, w = extract_raw_poses(
                video, pose_tracker,
                progress_prefix=f"[{idx}/{len(videos)}] ",
            )
            t_extract = time.monotonic() - t0
            if xy.shape[0] == 0:
                print(f"  WARN: 0 frames decoded, skipping", flush=True)
                n_skip += 1
                continue

            smoothed_xy, smoothed_sc = smooth_sequence(
                xy, sc, fps=fps,
                min_cutoff=args.min_cutoff,
                beta=args.beta,
                conf_thresh=args.conf_thresh,
            )

            save_compatible_npz(
                out_path,
                smoothed_xy, smoothed_sc,
                orig_h=h, orig_w=w, fps=fps,
            )
            n_ok += 1
            frames = xy.shape[0]
            n_missed = int((sc.max(axis=1) < args.conf_thresh).sum())
            print(f"  OK: {frames} frames, {n_missed} no-detection frames, "
                  f"{t_extract:.1f}s extract ({frames/t_extract:.1f} fps)", flush=True)
        except Exception as e:
            print(f"  FAIL: {e!r}", flush=True)
            n_skip += 1

    t_total = time.monotonic() - t_total
    print(f"\nDone in {t_total:.1f}s: {n_ok} written, {n_skip} skipped/failed",
          flush=True)


if __name__ == "__main__":
    main()
