"""Render the IMG_1665 reconstruction via single full-clip forward pass.

Purpose: validate that for our *bidirectional* TCN, chunked inference is
unnecessary — the model is translation-equivariant, so feeding the entire clip
in one pass should produce identical outputs to chunked inference for any
interior frame, and *better* outputs at boundaries (because reflection padding
gives synthetic neighbours).

Compares three modes side-by-side via L1 + writes mp4s:
  1. Chunked + Hann overlap + reflect pad   (current production)
  2. Chunked, NO Hann overlap (stride=chunk) (the "old" non-overlap mode)
  3. Full-clip single forward pass, reflect-padded edges only
"""

from __future__ import annotations

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


CHECKPOINT = "checkpoints/overfit-img1665-rand/step_3000"
POSE_NPZ = "pose_overlays/IMG_1665_30fps_poses.npz"
VIDEO = "IMG_1665_30fps.mp4"
DEVICE = "cuda:0"
OUT_DIR = Path("pose_comparisons/fullclip_test")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def preprocess(raw_kp, raw_sc, res_h, res_w):
    sel_kp = raw_kp[:, SELECTED_INDICES]
    sel_sc = raw_sc[:, SELECTED_INDICES]
    positions = np.empty_like(sel_kp, dtype=np.float32)
    positions[..., 0] = sel_kp[..., 1] / res_h
    positions[..., 1] = sel_kp[..., 0] / res_w
    conf = sel_sc.astype(np.float32)
    sw = np.abs(positions[:, ROOT_LEFT, 1] - positions[:, ROOT_RIGHT, 1])
    msw = float(np.median(sw))
    root = (positions[:, ROOT_LEFT, :] + positions[:, ROOT_RIGHT, :]) / 2.0
    _, kc = positions_to_offsets(positions)
    kc = kc / msw
    T = positions.shape[0]
    kp_flat = kc.reshape(T, -1).astype(np.float32)
    gs = np.stack([root[:, 0], root[:, 1], np.full(T, msw, dtype=np.float32)], axis=-1)
    return kp_flat, conf, gs


@torch.no_grad()
def fullclip_reflectpad(model, kp, device, pad: int):
    """Single forward pass on the entire clip with reflection padding at the
    absolute sequence boundaries (so the first/last few frames see synthetic
    smooth context instead of zero padding)."""
    kp_padded = np.pad(kp, ((pad, pad), (0, 0)), mode="reflect")
    inp = torch.from_numpy(kp_padded).float().unsqueeze(0).to(device)
    out = model(inp)["reconstruction"].squeeze(0).cpu().numpy()
    return out[pad:pad + kp.shape[0]]


def main():
    ckpt = Path(CHECKPOINT)
    cfg, _ = TrainConfig.from_yaml(ckpt.parent / "config.yaml")
    chunk = cfg.segment_frames

    model = PoseTokenizerModel.from_pretrained(str(ckpt)).to(DEVICE).eval()
    print(f"Loaded {ckpt.parent.name}/{ckpt.name}  chunk={chunk}  target_fps={cfg.target_fps}")

    data = np.load(POSE_NPZ)
    raw_kp, raw_sc = data["keypoints"], data["scores"]
    src_fps = float(data["fps"])
    res_w, res_h = int(data["resolution"][0]), int(data["resolution"][1])
    kp, conf, gs = preprocess(raw_kp, raw_sc, res_h, res_w)
    # Resample to model fps
    T_src = kp.shape[0]
    T_tgt = round(T_src * cfg.target_fps / src_fps)
    idx = np.round(np.linspace(0, T_src - 1, T_tgt)).astype(int)
    kp, conf, gs = kp[idx], conf[idx], gs[idx]
    print(f"Clip: T={T_tgt} @ {cfg.target_fps}fps  ({T_tgt/cfg.target_fps:.1f}s)")

    pad = chunk // 2
    print(f"\nReconstruction variants:")

    # 1. Current production: chunked + Hann + reflect (already includes reflect-pad)
    r_chunked_hann = _reconstruct_clip(model, kp, DEVICE, chunk_size=chunk, stride=chunk // 2)
    l1 = np.abs(kp - r_chunked_hann).mean()
    print(f"  [1] chunked + Hann overlap (stride={chunk//2}) + reflect-pad:  L1={l1:.6f}")

    # 2. Chunked, NO overlap (stride=chunk)
    r_chunked_noov = _reconstruct_clip(model, kp, DEVICE, chunk_size=chunk, stride=chunk)
    l1 = np.abs(kp - r_chunked_noov).mean()
    print(f"  [2] chunked NO overlap (stride={chunk})    + reflect-pad:  L1={l1:.6f}")

    # 3. Full-clip single forward pass, reflect-padded edges only
    r_fullclip = fullclip_reflectpad(model, kp, DEVICE, pad=pad)
    l1 = np.abs(kp - r_fullclip).mean()
    print(f"  [3] FULL-CLIP single forward + reflect-pad edges (pad={pad}):  L1={l1:.6f}")

    # Cross-check: how much do the three variants disagree from each other?
    print(f"\nPairwise L1 between variants (smaller = more agreement):")
    print(f"  [1]↔[3]: {np.abs(r_chunked_hann - r_fullclip).mean():.6f}")
    print(f"  [2]↔[3]: {np.abs(r_chunked_noov - r_fullclip).mean():.6f}")
    print(f"  [1]↔[2]: {np.abs(r_chunked_hann - r_chunked_noov).mean():.6f}")

    print(f"\nRendering...")
    for tag, recon in [("chunked_hann", r_chunked_hann),
                       ("chunked_noov", r_chunked_noov),
                       ("fullclip", r_fullclip)]:
        out_path = OUT_DIR / f"IMG_1665__overfit-rand-step_3000__{tag}.mp4"
        render_comparison_video(
            orig_kp=kp, recon_kp=recon, global_state=gs, confidence=conf,
            out_path=out_path, video_path=Path(VIDEO) if Path(VIDEO).exists() else None,
            fps=cfg.target_fps, max_seconds=30.0,
        )
        print(f"  → {out_path}")


if __name__ == "__main__":
    main()
