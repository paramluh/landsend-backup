"""Render an external (notebook-style) pose npz through a trained checkpoint.

Mirrors the notebook's Cell 4 preprocessing (raw 133-keypoint detections →
kinematic-chain offsets normalised by median shoulder width), then runs the
model with the same fixes as the training-time render pipeline:

  • Resamples from the npz's fps to the checkpoint's target_fps (matches training)
  • Chunked inference at cfg.segment_frames (matches training-time T)
  • Time-aligned source video underlay (handled by render_comparison_video)

Outputs side-by-side comparison + a single-overlay version (red recon on
original video) to pose_comparisons/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from pose_tokenizer import PoseTokenizerModel
from pose_tokenizer.config import TrainConfig
from pose_tokenizer.data.kinematic import (
    ROOT_LEFT,
    ROOT_RIGHT,
    SELECTED_INDICES,
    positions_to_offsets,
)
from pose_tokenizer.data.rendering import _reconstruct_clip, render_comparison_video


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--pose-npz", default="pose_overlays/IMG_1665_30fps_poses.npz")
    p.add_argument("--video", default="IMG_1665_30fps.mp4")
    p.add_argument("--checkpoint", default="checkpoints/hard-conf-0.3/step_9000")
    p.add_argument("--out", default=None,
                   help="Output mp4 (default: pose_comparisons/<clip>__<run>.mp4)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-seconds", type=float, default=30.0)
    p.add_argument("--chunk-stride", type=int, default=0,
                   help="Inference chunk stride. 0 → chunk_size // 2 (50% overlap, "
                        "Hann-blended). Set = chunk_size for non-overlapping.")
    return p.parse_args()


def preprocess_to_kinematic(raw_kp: np.ndarray, raw_sc: np.ndarray,
                             res_h: float, res_w: float):
    """Mirror notebook Cell 4 (current-kinematic, 55-joint / 110-feature path).

    Returns:
        keypoints_flat:  (T, 110)  kinematic-chain offsets, shoulder-width-normalised
        confidence:      (T, 55)
        global_state:    (T, 3)    [root_y, root_x, median_shoulder_width per frame]
        positions:       (T, 55, 2) absolute normalised positions (for visualisation)
    """
    sel_kp = raw_kp[:, SELECTED_INDICES]            # (T, 55, 2) in (x, y) px
    sel_sc = raw_sc[:, SELECTED_INDICES]            # (T, 55)
    # Normalise to image coords; flip to (y, x) order to match the rest of the pipeline.
    positions = np.empty_like(sel_kp, dtype=np.float32)
    positions[..., 0] = sel_kp[..., 1] / res_h
    positions[..., 1] = sel_kp[..., 0] / res_w
    confidence = sel_sc.astype(np.float32)

    shoulder_width = np.abs(positions[:, ROOT_LEFT, 1] - positions[:, ROOT_RIGHT, 1])
    median_sw = float(np.median(shoulder_width))
    root = (positions[:, ROOT_LEFT, :] + positions[:, ROOT_RIGHT, :]) / 2.0

    _, kc_offsets = positions_to_offsets(positions)
    kc_offsets = kc_offsets / median_sw
    T = positions.shape[0]
    keypoints_flat = kc_offsets.reshape(T, -1).astype(np.float32)
    global_state = np.stack(
        [root[:, 0], root[:, 1], np.full(T, median_sw, dtype=np.float32)], axis=-1,
    )
    return keypoints_flat, confidence, global_state, positions


def resample(arr: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return arr[idx]


def main():
    args = parse()
    ckpt_dir = Path(args.checkpoint)
    run_dir = ckpt_dir.parent
    cfg, _ = TrainConfig.from_yaml(run_dir / "config.yaml")
    target_fps = cfg.target_fps or cfg.source_fps
    chunk = cfg.segment_frames if cfg.segment_frames > 0 else None

    # Load model
    model = PoseTokenizerModel.from_pretrained(str(ckpt_dir)).to(args.device).eval()
    print(f"Loaded {ckpt_dir.parent.name}/{ckpt_dir.name}")
    print(f"  input_features={model.config.input_features}  segment_frames={chunk}  target_fps={target_fps}")
    if model.config.input_features != 110:
        raise SystemExit(
            f"This script targets the 55-joint / 110-feature schema. "
            f"Checkpoint has input_features={model.config.input_features}. "
            f"Use a different preprocessing path (notebook Cell 4's legacy branch)."
        )

    # Load + preprocess the external npz
    data = np.load(args.pose_npz)
    raw_kp = data["keypoints"]
    raw_sc = data["scores"]
    src_fps = float(data["fps"])
    res_w, res_h = int(data["resolution"][0]), int(data["resolution"][1])
    print(f"Loaded {args.pose_npz}: T={raw_kp.shape[0]}  fps={src_fps}  res={res_w}x{res_h}")

    kp_flat, conf, gs, _ = preprocess_to_kinematic(raw_kp, raw_sc, res_h, res_w)

    # Resample to checkpoint's target_fps
    if abs(src_fps - target_fps) > 0.01:
        T_src = kp_flat.shape[0]
        T_tgt = round(T_src * target_fps / src_fps)
        idx = np.round(np.linspace(0, T_src - 1, T_tgt)).astype(int)
        kp_flat = resample(kp_flat, idx)
        conf = resample(conf, idx)
        gs = resample(gs, idx)
        print(f"Resampled {src_fps} → {target_fps} fps: T {T_src} → {T_tgt}")

    # Chunked inference (overlap-and-add with Hann blending)
    stride = args.chunk_stride if args.chunk_stride > 0 else None
    recon = _reconstruct_clip(model, kp_flat, args.device,
                              chunk_size=chunk, stride=stride)
    l1 = np.abs(kp_flat - recon).mean()
    stride_str = f"stride={stride}" if stride else f"stride={chunk // 2 if chunk else None} (default = chunk/2)"
    print(f"Reconstruction L1 (all features, no mask): {l1:.6f}   [chunk={chunk}  {stride_str}]")

    # Render
    out_path = Path(args.out) if args.out else (
        Path("pose_comparisons") / f"{Path(args.pose_npz).stem}__{run_dir.name}-{ckpt_dir.name}.mp4"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Rendering → {out_path}")
    render_comparison_video(
        orig_kp=kp_flat,
        recon_kp=recon,
        global_state=gs,
        confidence=conf,
        out_path=out_path,
        video_path=Path(args.video) if Path(args.video).exists() else None,
        fps=target_fps,
        max_seconds=args.max_seconds,
    )
    print(f"Done → {out_path}")


if __name__ == "__main__":
    main()
