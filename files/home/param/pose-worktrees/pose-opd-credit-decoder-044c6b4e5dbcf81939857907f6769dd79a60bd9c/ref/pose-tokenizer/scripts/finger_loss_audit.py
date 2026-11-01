"""Audit per-region L1 reconstruction error on a single test clip.

Tests the hypothesis that the kinematic-chain encoding + shoulder-width
normalisation makes finger-tip offsets so small that they contribute
negligibly to the training L1 loss — even when their relative error is
large or their absolute (rendered) position is far off.

Reports L1 in two spaces side-by-side:
  - "offset" : what the model trains on (kinematic offsets / median_sw)
  - "abs"    : reconstructed absolute positions in normalised image coords,
               obtained by walking the kinematic chain. This is what the
               viewer perceives in the comparison videos.

Usage:
    python -m scripts.finger_loss_audit --ckpt checkpoints/baseline/step_12000 \
                                        --clip test_set/actor_2/segment_01_poses.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pose_tokenizer.config import PoseTokenizerConfig
from pose_tokenizer.data.dataset import _process_poses_to_clip, _resample_clip
from pose_tokenizer.data.kinematic import (
    NUM_JOINTS,
    PARENT_INDEX,
    TOPO_ORDER,
    propagate_confidence,
)
from pose_tokenizer.model import PoseTokenizerModel


# Joint groups: (label, list_of_joint_indices). Mirrors the layout in
# pose_tokenizer/data/kinematic.py — 5 face + 8 upper body + 21 left hand +
# 21 right hand. Hand layout per finger is wrist (0), then MCP/PIP/DIP/TIP for
# 5 fingers ⇒ within-hand local indices 1/5/9/13/17 are MCPs, 2/6/10/14/18
# PIPs, 3/7/11/15/19 DIPs, 4/8/12/16/20 TIPs.
def joint_groups() -> dict[str, list[int]]:
    L, R = 13, 34  # offsets of left/right hand within the 55-joint layout
    mcp_local = [1, 5, 9, 13, 17]
    pip_local = [2, 6, 10, 14, 18]
    dip_local = [3, 7, 11, 15, 19]
    tip_local = [4, 8, 12, 16, 20]
    return {
        "face":           list(range(0, 5)),
        "upper_body":     list(range(5, 13)),
        "hand_wrist":     [L, R],
        "hand_mcp":       [L + i for i in mcp_local] + [R + i for i in mcp_local],
        "hand_pip":       [L + i for i in pip_local] + [R + i for i in pip_local],
        "hand_dip":       [L + i for i in dip_local] + [R + i for i in dip_local],
        "hand_tip":       [L + i for i in tip_local] + [R + i for i in tip_local],
    }


def offsets_to_absolute(root: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """(T, 2) root + (T, 55, 2) offsets → (T, 55, 2) absolute positions."""
    pos = torch.empty_like(offsets)
    for i in TOPO_ORDER:
        parent = int(PARENT_INDEX[i])
        if parent == -1:
            pos[..., i, :] = root + offsets[..., i, :]
        else:
            pos[..., i, :] = pos[..., parent, :] + offsets[..., i, :]
    return pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to checkpoint dir (with config.json + model.safetensors)")
    ap.add_argument("--clip", required=True, help="path to *_poses.npz test clip")
    ap.add_argument("--source-fps", type=int, default=30)
    ap.add_argument("--target-fps", type=int, default=25)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--conf-threshold", type=float, default=0.3,
                    help="match training: hard-mask joints with propagated confidence below this")
    args = ap.parse_args()

    # ----- model -----
    ckpt_dir = Path(args.ckpt)
    with open(ckpt_dir / "model_config.json") as f:
        model_cfg_dict = json.load(f)
    cfg = PoseTokenizerConfig(**{
        k: v for k, v in model_cfg_dict.items()
        if k in PoseTokenizerConfig.__dataclass_fields__
    })
    model = PoseTokenizerModel.from_pretrained(str(ckpt_dir))
    model = model.to(args.device).eval()
    print(f"[model] loaded {ckpt_dir}: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params, "
          f"latent_dim={model.latent_dim}, n_codebooks={cfg.n_codebooks}")

    # ----- clip -----
    clip_path = Path(args.clip)
    with np.load(clip_path, allow_pickle=True) as data:
        poses = np.asarray(data["poses"], dtype=np.float32)
        orig_h = float(data["orig_h"])
        orig_w = float(data["orig_w"])
    print(f"[clip] {clip_path}: T_src={poses.shape[0]}, orig_h={orig_h}, orig_w={orig_w}")

    processed = _process_poses_to_clip(poses, orig_h, orig_w, None, None)
    if processed is None:
        raise RuntimeError("Clip rejected by _process_poses_to_clip (shoulder width too small).")
    if args.source_fps != args.target_fps:
        processed = _resample_clip(processed, args.source_fps, args.target_fps)

    keypoints = torch.from_numpy(processed["keypoints"])      # (T, 110) — offsets / median_sw
    confidence = torch.from_numpy(processed["confidence"])    # (T, 55)  — raw per-joint conf
    global_state = torch.from_numpy(processed["global_state"]) # (T, 3) — root_y, root_x, sw
    T = keypoints.shape[0]
    print(f"[clip] T_target={T} @ {args.target_fps}fps")

    # Pad to multiple of downsampling_factor (4 for current arch) so the encoder
    # bottleneck shape matches; trim back after.
    df = model.downsampling_factor
    pad = (df - T % df) % df
    if pad:
        keypoints_pad = F.pad(keypoints, (0, 0, 0, pad))
    else:
        keypoints_pad = keypoints

    # ----- inference -----
    with torch.no_grad():
        x = keypoints_pad.unsqueeze(0).to(args.device)        # (1, T_pad, 110)
        out = model(x)
        recon = out["reconstruction"].squeeze(0).cpu()[:T]    # (T, 110)

    # Reshape back to (T, 55, 2) for per-joint analysis.
    gt_off    = keypoints.reshape(T, NUM_JOINTS, 2)
    recon_off = recon.reshape(T, NUM_JOINTS, 2)

    # Confidence mask (matches training's "hard" mode: joints below threshold
    # contribute 0 to loss). Training propagates confidence DOWN the chain
    # (a joint can only be as confident as its least-confident ancestor) — we
    # do the same here to match the model's effective training signal.
    conf_prop = torch.from_numpy(
        propagate_confidence(confidence.numpy())
    )  # (T, 55)
    conf_mask = (conf_prop >= args.conf_threshold).float()   # (T, 55)
    valid_per_joint = conf_mask.sum(dim=0)                    # (55,) — frames retained

    # ----- absolute-position reconstruction -----
    # Roll back the shoulder-width normalisation. global_state[:, 2] is sw,
    # but training divides by the *median* sw (so a single scalar applies
    # across all frames). Recompute median from global_state to stay faithful.
    median_sw = float(np.median(global_state[:, 2].numpy()))
    root      = global_state[:, :2]                           # (T, 2) absolute root in (y, x)
    gt_pos    = offsets_to_absolute(root, gt_off    * median_sw)  # (T, 55, 2)
    recon_pos = offsets_to_absolute(root, recon_off * median_sw)

    # ----- losses -----
    # Per-joint mean L1 over confident frames. Use sum-over-frames / valid count
    # so joints with many low-confidence frames aren't unfairly punished.
    def per_joint_l1(diff_xyz: torch.Tensor) -> torch.Tensor:
        # diff_xyz: (T, 55, 2). Mask along T per-joint.
        per_frame = diff_xyz.abs().mean(dim=-1)               # (T, 55)
        masked    = per_frame * conf_mask                     # zero out low-conf frames
        denom     = valid_per_joint.clamp(min=1.0)
        return masked.sum(dim=0) / denom                      # (55,)

    l1_off = per_joint_l1(recon_off - gt_off)
    l1_abs = per_joint_l1(recon_pos - gt_pos)

    # ----- group aggregation -----
    groups = joint_groups()

    print("\n[per-region] mean L1 error, masked by confidence ≥ {:.2f}".format(args.conf_threshold))
    print(f"  median shoulder width = {median_sw:.4f}  (in normalised image coords)")
    header = f"  {'group':<14} {'n_jts':>6}  {'avg_valid_T':>12}  "\
             f"{'L1_off':>10} {'L1_abs (norm)':>14} {'L1_abs (px)':>13}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label, idxs in groups.items():
        n = len(idxs)
        avg_valid = float(valid_per_joint[idxs].mean())
        l1o = float(l1_off[idxs].mean())
        l1a = float(l1_abs[idxs].mean())
        # Convert abs L1 from normalised coords to pixels — abs is in (y, x)
        # where y was divided by orig_h and x by orig_w. We average both dims,
        # so the equivalent pixel scale is the geometric mean of the two.
        px = l1a * (orig_h * orig_w) ** 0.5
        print(f"  {label:<14} {n:>6}  {avg_valid:>12.1f}  "
              f"{l1o:>10.5f} {l1a:>14.5f} {l1a * orig_w:>10.2f}px")

    # ----- per-fingertip detail -----
    L, R = 13, 34
    tip_local = [4, 8, 12, 16, 20]
    finger_names = ["thumb", "index", "middle", "ring", "pinky"]
    print("\n[per-fingertip] left vs right (5 fingers each):")
    print(f"  {'finger':<8} {'side':<6} {'L1_off':>10} {'L1_abs (norm)':>14} {'abs (px≈x)':>12}")
    print("  " + "-" * 56)
    for name, t in zip(finger_names, tip_local):
        for side, base in [("left", L), ("right", R)]:
            j = base + t
            print(f"  {name:<8} {side:<6} "
                  f"{float(l1_off[j]):>10.5f} {float(l1_abs[j]):>14.5f} "
                  f"{float(l1_abs[j]) * orig_w:>10.2f}px")

    # ----- contribution-to-loss summary -----
    print("\n[loss-contribution] share of total L1_off contributed by each region:")
    total_off = l1_off.sum().item()
    for label, idxs in groups.items():
        share = float(l1_off[idxs].sum()) / total_off * 100
        print(f"  {label:<14}  {share:>6.2f}%   ({len(idxs)} joints)")

    # ----- offset-magnitude reference -----
    # How big are GT offsets in each region? If the model could just predict 0,
    # how good would it look in offset space? This calibrates "how small are
    # the ground-truth offsets that we're trying to fit."
    gt_off_mag = gt_off.abs().mean(dim=-1)                    # (T, 55)
    masked_mag = (gt_off_mag * conf_mask).sum(dim=0) / valid_per_joint.clamp(min=1.0)
    print("\n[reference] |GT offset| mean per region (the loss budget per joint):")
    print(f"  {'group':<14} {'|GT_off|':>10} {'L1_off':>10} {'rel_err':>10}")
    for label, idxs in groups.items():
        gtm = float(masked_mag[idxs].mean())
        l1o = float(l1_off[idxs].mean())
        rel = l1o / max(gtm, 1e-9)
        print(f"  {label:<14} {gtm:>10.5f} {l1o:>10.5f} {rel:>10.2%}")


if __name__ == "__main__":
    main()
