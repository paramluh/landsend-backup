"""Comprehensive data profiling for the pose dataset.

Generates:
  1. Per-joint confidence histograms
  2. Co-occlusion matrix (P(j < thresh | i < thresh))
  3. Temporal occlusion run-length distributions
  4. Per-joint jitter conditioned on confidence buckets
  5. Fraction of clean frames / clips
  6. Shoulder-width distribution

For each clip we need image dimensions to normalise coordinates. Newer .npz
files store ``orig_h`` / ``orig_w`` directly; older ones don't, so we fall
back to a ``dims.csv`` lookup (columns: ``clip_stem,width,height``).

Usage:
    python scripts/profile_data.py \
        --data-dir /mnt/somfs/pose-conditioning-files/pose_data/ \
        --manifest /mnt/somfs/pose-conditioning-files/available_keypoints.txt \
        --dims-csv dims.csv \
        --output-dir scripts/profiling_output \
        --num-clips 100000 \
        --workers 16
"""

from __future__ import annotations

import argparse
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from pose_tokenizer.data.dataset import _pick_best_person
from pose_tokenizer.data.kinematic import (
    SELECTED_INDICES,
    NUM_JOINTS,
    ROOT_LEFT,
    ROOT_RIGHT,
)

JOINT_NAMES = [
    "nose", "L_eye", "R_eye", "L_ear", "R_ear",
    "L_shldr", "R_shldr", "L_elbow", "R_elbow",
    "L_wrist", "R_wrist", "L_hip", "R_hip",
    "LH_wrst",
    "LH_th1", "LH_th2", "LH_th3", "LH_th4",
    "LH_ix1", "LH_ix2", "LH_ix3", "LH_ix4",
    "LH_md1", "LH_md2", "LH_md3", "LH_md4",
    "LH_rn1", "LH_rn2", "LH_rn3", "LH_rn4",
    "LH_pk1", "LH_pk2", "LH_pk3", "LH_pk4",
    "RH_wrst",
    "RH_th1", "RH_th2", "RH_th3", "RH_th4",
    "RH_ix1", "RH_ix2", "RH_ix3", "RH_ix4",
    "RH_md1", "RH_md2", "RH_md3", "RH_md4",
    "RH_rn1", "RH_rn2", "RH_rn3", "RH_rn4",
    "RH_pk1", "RH_pk2", "RH_pk3", "RH_pk4",
]
assert len(JOINT_NAMES) == NUM_JOINTS, f"{len(JOINT_NAMES)} names vs {NUM_JOINTS} joints"


# ---------------------------------------------------------------------------
# Parallel loading
# ---------------------------------------------------------------------------

# Populated per worker process by _init_worker (or in-process for workers<=1).
# Maps clip stem (without "_poses" suffix) -> (height, width).
_DIMS: dict[str, tuple[float, float]] = {}


def _load_dims_csv(dims_csv: str | Path) -> dict[str, tuple[float, float]]:
    """Load dims.csv into a dict for O(1) fallback lookup.

    Columns: ``clip_stem,width,height``. Returned dict maps stem -> (h, w)
    to match the ``orig_h, orig_w`` ordering used elsewhere.
    """
    import csv as _csv
    out: dict[str, tuple[float, float]] = {}
    with open(dims_csv, newline="") as f:
        for row in _csv.DictReader(f):
            out[row["clip_stem"]] = (float(row["height"]), float(row["width"]))
    return out


def _init_worker(dims_csv: str | None) -> None:
    global _DIMS
    if dims_csv:
        _DIMS = _load_dims_csv(dims_csv)


def _process_one_clip(path: Path) -> dict | None:
    """Load a single .npz and extract the stats we need. Runs in a worker."""
    try:
        data = np.load(path, allow_pickle=True)
    except Exception:
        return None

    poses = data["poses"]
    # Newer .npz files embed orig_h/orig_w; older ones need the dims.csv fallback.
    try:
        orig_h, orig_w = float(data["orig_h"]), float(data["orig_w"])
    except KeyError:
        stem = path.stem
        if stem.endswith("_poses"):
            stem = stem[: -len("_poses")]
        dims = _DIMS.get(stem)
        if dims is None:
            return None
        orig_h, orig_w = dims

    T = poses.shape[0]
    positions = np.empty((T, NUM_JOINTS, 2), dtype=np.float32)
    confidence = np.empty((T, NUM_JOINTS), dtype=np.float32)

    for t in range(T):
        frame = np.asarray(poses[t], dtype=np.float32)
        if frame.ndim == 2:
            frame = frame[np.newaxis, ...]
        if frame.shape[0] == 0:
            positions[t] = 0.0
            confidence[t] = 0.0
            continue
        person = _pick_best_person(frame)
        selected = person[SELECTED_INDICES]
        selected[:, 0] /= orig_h
        selected[:, 1] /= orig_w
        positions[t] = selected[:, :2]
        confidence[t] = selected[:, 2]

    return {
        "positions": positions,
        "confidence": confidence,
        "orig_w": orig_w,
    }


def load_file_list(data_dir: Path, manifest: Path | None, num_clips: int) -> list[Path]:
    if manifest is not None:
        with open(manifest) as f:
            all_paths = [data_dir / line.strip() for line in f if line.strip()]
    else:
        all_paths = sorted(data_dir.rglob("*_keypoints.npz"))

    if num_clips > 0 and num_clips < len(all_paths):
        random.seed(42)
        all_paths = random.sample(all_paths, num_clips)

    return all_paths


def load_raw_clips(
    paths: list[Path], workers: int, dims_csv: str | None
) -> list[dict]:
    clips: list[dict] = []

    if workers <= 1:
        _init_worker(dims_csv)
        for path in tqdm(paths, desc="Loading clips"):
            result = _process_one_clip(path)
            if result is not None:
                clips.append(result)
        return clips

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(dims_csv,),
    ) as pool:
        futures = {pool.submit(_process_one_clip, p): p for p in paths}
        for future in tqdm(
            as_completed(futures), total=len(futures), desc=f"Loading clips ({workers} workers)"
        ):
            result = future.result()
            if result is not None:
                clips.append(result)

    return clips


# ---------------------------------------------------------------------------
# 1. Per-joint confidence histograms
# ---------------------------------------------------------------------------

def plot_confidence_histograms(all_conf: np.ndarray, output_dir: Path):
    fig, axes = plt.subplots(5, 11, figsize=(44, 20))
    fig.suptitle("Per-joint confidence distributions", fontsize=20)

    flat = axes.flatten()
    for i, ax in enumerate(flat):
        if i >= NUM_JOINTS:
            ax.axis("off")
            continue
        ax.hist(all_conf[:, i], bins=50, range=(0, 1), color="steelblue",
                edgecolor="none", alpha=0.8)
        ax.set_title(f"{i} {JOINT_NAMES[i]}", fontsize=10)
        ax.set_xlim(0, 1)
        ax.tick_params(labelsize=7)

    fig.tight_layout()
    fig.savefig(output_dir / "01_confidence_histograms.png", dpi=200)
    plt.close(fig)
    print("  [1/6] Confidence histograms saved")


# ---------------------------------------------------------------------------
# 2. Co-occlusion matrix
# ---------------------------------------------------------------------------

def plot_coocclusion_matrix(all_conf: np.ndarray, output_dir: Path, threshold: float):
    below = all_conf < threshold
    count_i = below.sum(axis=0).astype(np.float64)  # (50,)
    count_i[count_i == 0] = 1  # avoid div-by-zero

    # Vectorised: cooccl[i,j] = (below[:,i] & below[:,j]).sum() / count_i[i]
    cooccl = (below.T.astype(np.float64) @ below.astype(np.float64)) / count_i[:, None]

    fig, ax = plt.subplots(figsize=(18, 16))
    im = ax.imshow(cooccl, cmap="inferno", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_JOINTS))
    ax.set_yticks(range(NUM_JOINTS))
    ax.set_xticklabels(JOINT_NAMES, rotation=90, fontsize=8)
    ax.set_yticklabels(JOINT_NAMES, fontsize=8)
    ax.set_xlabel("Joint j", fontsize=12)
    ax.set_ylabel("Joint i (conditioned on i < threshold)", fontsize=12)
    ax.set_title(f"P(j < {threshold} | i < {threshold})  —  {below.shape[0]:,} frames", fontsize=14)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "02_coocclusion_matrix.png", dpi=200)
    plt.close(fig)
    print("  [2/6] Co-occlusion matrix saved")


# ---------------------------------------------------------------------------
# 3. Temporal occlusion run lengths
# ---------------------------------------------------------------------------

def _run_lengths(mask: np.ndarray) -> list[int]:
    if not mask.any():
        return []
    diffs = np.diff(mask.astype(np.int8))
    starts = np.where(diffs == 1)[0] + 1
    ends = np.where(diffs == -1)[0] + 1
    if mask[0]:
        starts = np.concatenate([[0], starts])
    if mask[-1]:
        ends = np.concatenate([ends, [len(mask)]])
    return (ends - starts).tolist()


def plot_temporal_run_lengths(clips: list[dict], output_dir: Path, threshold: float):
    joints_to_plot = {
        "nose (0)": 0,
        "L_shoulder (5)": 5,
        "L_wrist (9)": 9,
        "LH_wrist (13)": 13,
        "LH_index_tip (21)": 21,
        "RH_index_tip (42)": 42,
    }

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f"Temporal occlusion run lengths (conf < {threshold})", fontsize=16)

    for ax, (label, j) in zip(axes.flatten(), joints_to_plot.items()):
        all_runs = []
        for clip in clips:
            runs = _run_lengths(clip["confidence"][:, j] < threshold)
            all_runs.extend(runs)

        if all_runs:
            ax.hist(all_runs, bins=range(1, min(max(all_runs) + 2, 60)),
                    color="coral", edgecolor="none", alpha=0.8)
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("consecutive frames", fontsize=10)
        ax.set_ylabel("count", fontsize=10)

    fig.tight_layout()
    fig.savefig(output_dir / "03_temporal_run_lengths.png", dpi=200)
    plt.close(fig)
    print("  [3/6] Temporal run lengths saved")


# ---------------------------------------------------------------------------
# 4. Per-joint jitter vs confidence
# ---------------------------------------------------------------------------

def plot_jitter_vs_confidence(clips: list[dict], output_dir: Path):
    buckets = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]

    # Accumulate sums online to avoid storing all jitter values.
    sums = {b: 0.0 for b in buckets}
    sq_sums = {b: 0.0 for b in buckets}
    counts = {b: 0 for b in buckets}

    for clip in clips:
        pos = clip["positions"]
        conf = clip["confidence"]
        if pos.shape[0] < 2:
            continue
        jitter = np.linalg.norm(np.diff(pos, axis=0), axis=-1)  # (T-1, 50)
        c = conf[:-1]

        for lo, hi in buckets:
            mask = (c >= lo) & (c < hi)
            vals = jitter[mask]
            if vals.size > 0:
                sums[(lo, hi)] += vals.sum()
                sq_sums[(lo, hi)] += (vals ** 2).sum()
                counts[(lo, hi)] += vals.size

    labels = [f"{lo:.1f}-{hi:.1f}" for lo, hi in buckets]
    means = []
    stds = []
    for b in buckets:
        if counts[b] > 0:
            m = sums[b] / counts[b]
            means.append(m)
            stds.append(np.sqrt(sq_sums[b] / counts[b] - m ** 2))
        else:
            means.append(0)
            stds.append(0)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.bar(labels, means, yerr=stds, color="mediumpurple", edgecolor="none",
           capsize=4, alpha=0.8)
    ax.set_xlabel("Confidence bucket", fontsize=12)
    ax.set_ylabel("Mean frame-to-frame jitter (normalised coords)", fontsize=12)
    ax.set_title("Coordinate jitter vs confidence", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "04_jitter_vs_confidence.png", dpi=200)
    plt.close(fig)
    print("  [4/6] Jitter vs confidence saved")


# ---------------------------------------------------------------------------
# 5. Clean frames / clips
# ---------------------------------------------------------------------------

def plot_clean_stats(clips: list[dict], output_dir: Path):
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    frame_pct, clip_pct, clip_90_pct = [], [], []

    for thresh in thresholds:
        total_frames = 0
        clean_frames = 0
        clean_clips = 0
        mostly_clean_clips = 0

        for clip in clips:
            conf = clip["confidence"]
            T = conf.shape[0]
            total_frames += T

            all_above = (conf >= thresh).all(axis=1)
            clean_frames += int(all_above.sum())
            if all_above.all():
                clean_clips += 1

            mostly_above = (conf >= thresh).mean(axis=1) >= 0.9
            if mostly_above.mean() >= 0.9:
                mostly_clean_clips += 1

        n = len(clips)
        frame_pct.append(clean_frames / total_frames * 100)
        clip_pct.append(clean_clips / n * 100)
        clip_90_pct.append(mostly_clean_clips / n * 100)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(thresholds, frame_pct, "o-", label="All joints clean (frames)", linewidth=2)
    ax.plot(thresholds, clip_pct, "s-", label="All joints clean (full clips)", linewidth=2)
    ax.plot(thresholds, clip_90_pct, "^-", label="90% joints × 90% frames (clips)", linewidth=2)
    ax.set_xlabel("Confidence threshold", fontsize=12)
    ax.set_ylabel("Percentage", fontsize=12)
    ax.set_title("Fraction of clean data vs confidence threshold", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "05_clean_stats.png", dpi=200)
    plt.close(fig)
    print("  [5/6] Clean stats saved")


# ---------------------------------------------------------------------------
# 6. Shoulder width distribution
# ---------------------------------------------------------------------------

def plot_shoulder_width(clips: list[dict], output_dir: Path):
    median_sw_px = []
    median_sw_norm = []

    for clip in clips:
        pos = clip["positions"]
        sw_norm = np.abs(pos[:, ROOT_LEFT, 1] - pos[:, ROOT_RIGHT, 1])
        med = float(np.median(sw_norm))
        median_sw_norm.append(med)
        median_sw_px.append(med * clip["orig_w"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    ax1.hist(median_sw_px, bins=80, color="teal", edgecolor="none", alpha=0.8)
    ax1.axvline(10, color="red", linestyle="--", label="10px filter line")
    ax1.set_xlabel("Median shoulder width (pixels)", fontsize=12)
    ax1.set_ylabel("Clips", fontsize=12)
    ax1.set_title("Shoulder width distribution (pixels)", fontsize=14)
    ax1.legend(fontsize=11)

    ax2.hist(median_sw_norm, bins=80, color="teal", edgecolor="none", alpha=0.8)
    ax2.set_xlabel("Median shoulder width (normalised by img width)", fontsize=12)
    ax2.set_ylabel("Clips", fontsize=12)
    ax2.set_title("Shoulder width distribution (normalised)", fontsize=14)

    below_10px = sum(1 for s in median_sw_px if s < 10)
    n = len(clips)
    fig.suptitle(
        f"{n:,} clips  |  {below_10px} ({below_10px / n * 100:.1f}%) "
        f"with median shoulder width < 10px",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "06_shoulder_width.png", dpi=200)
    plt.close(fig)
    print("  [6/6] Shoulder width distribution saved")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Profile pose dataset quality")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Text file with one relative .npz path per line")
    parser.add_argument("--dims-csv", type=Path, default=None,
                        help="CSV (clip_stem,width,height) used as fallback "
                             "when an .npz doesn't embed orig_h/orig_w.")
    parser.add_argument("--output-dir", type=Path, default=Path("scripts/profiling_output"))
    parser.add_argument("--num-clips", type=int, default=100_000,
                        help="Random sample size (0 = use all)")
    parser.add_argument("--workers", type=int, default=16,
                        help="Number of parallel workers for loading")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Confidence threshold for co-occlusion / run-length analysis")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Building file list...")
    paths = load_file_list(args.data_dir, args.manifest, args.num_clips)
    print(f"  {len(paths):,} clips selected")
    dims_csv = str(args.dims_csv) if args.dims_csv else None
    if dims_csv:
        print(f"  dims.csv fallback: {dims_csv}")
    print()

    clips = load_raw_clips(paths, args.workers, dims_csv)
    total_frames = sum(c["confidence"].shape[0] for c in clips)
    skipped = len(paths) - len(clips)
    print(
        f"\nLoaded {len(clips):,} clips, {total_frames:,} frames"
        f"  (skipped {skipped:,} / {len(paths):,} — unreadable or no dims)\n"
    )

    all_conf = np.concatenate([c["confidence"] for c in clips], axis=0)

    print("Generating plots...")
    plot_confidence_histograms(all_conf, args.output_dir)
    plot_coocclusion_matrix(all_conf, args.output_dir, args.threshold)
    plot_temporal_run_lengths(clips, args.output_dir, args.threshold)
    plot_jitter_vs_confidence(clips, args.output_dir)
    plot_clean_stats(clips, args.output_dir)
    plot_shoulder_width(clips, args.output_dir)

    print(f"\nAll plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
