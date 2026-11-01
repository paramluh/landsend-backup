#!/usr/bin/env python3
"""Encode -> decode real pose keypoints with pose-tokenizer-2x-12cb-lossshift1
and assess reconstruction quality on 5 examples.

Ground-truth keypoints come from the pose-conditioning chunks:
  /mnt/som-fs/hands/pose-conditioning-keypoints/<clip>/<clip>_poses_chunkNNNN.safetensors
each holding `reconstruction (T,165)` (interleaved 55 joints x (off_y,off_x,conf))
and `global_state (T,3)`.

For each example: encode -> codes -> decode -> recon, report error metrics,
and render an original-vs-reconstructed side-by-side video. A montage of the
first-frame comparisons is also saved.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from safetensors.torch import load_file

sys.path.insert(0, "/home/james/pose-tokenizer")
from pose_tokenizer import PoseTokenizer  # noqa: E402
from pose_tokenizer.data.kinematic import (  # noqa: E402
    NUM_JOINTS, PARENT_INDEX, TOPO_ORDER,
    FACE_CONNECTIONS, BODY_CONNECTIONS,
    LEFT_HAND_CONNECTIONS, RIGHT_HAND_CONNECTIONS,
)

MODEL_DIR = "/home/james/vansh-video-processing/pose-processing/pose-tokenizer-2x-12cb-lossshift1"
KP_ROOT = Path("/mnt/som-fs/hands/pose-conditioning-keypoints")
W, H, PAD = 256, 480, 0.08
_CONN = FACE_CONNECTIONS + BODY_CONNECTIONS + LEFT_HAND_CONNECTIONS + RIGHT_HAND_CONNECTIONS
_JOINTS = sorted({i for p in _CONN for i in p})


def positions_from_offsets(offsets):
    out = np.empty_like(offsets)
    for j in TOPO_ORDER:
        p = PARENT_INDEX[j]
        base = 0 if p == -1 else out[..., p, :]
        out[..., j, :] = base + offsets[..., j, :]
    return out


def fit_params(positions):
    y, x = positions[..., 0], positions[..., 1]
    y0, y1, x0, x1 = float(y.min()), float(y.max()), float(x.min()), float(x.max())
    sy = H * (1 - 2 * PAD) / max(y1 - y0, 1e-6)
    sx = W * (1 - 2 * PAD) / max(x1 - x0, 1e-6)
    return sy, sx, (H - sy * (y0 + y1)) / 2, (W - sx * (x0 + x1)) / 2


def apply_fit(positions, p):
    sy, sx, oy, ox = p
    return np.stack([(positions[..., 0] * sy + oy) / H,
                     (positions[..., 1] * sx + ox) / W], axis=-1)


def color(c):
    c = float(np.clip(c, 0, 1)); return (int(255 * c), int(255 * (1 - c)), 0)


def draw(canvas, kn, conf):
    h, w = canvas.shape[:2]
    yp, xp = kn[:, 0] * h, kn[:, 1] * w
    for i, j in _CONN:
        cv2.line(canvas, (int(xp[i]), int(yp[i])), (int(xp[j]), int(yp[j])),
                 color((conf[i] + conf[j]) * 0.5), 2, cv2.LINE_AA)
    for idx in _JOINTS:
        cv2.circle(canvas, (int(xp[idx]), int(yp[idx])), 3, color(conf[idx]), -1, cv2.LINE_AA)
    return canvas


def panel(feat_row, fitp, label):
    f = feat_row.reshape(NUM_JOINTS, 3).astype(np.float32)
    pos = positions_from_offsets(f[None, :, :2])[0]
    kn = apply_fit(pos, fitp)
    c = np.zeros((H, W, 3), dtype=np.uint8)
    draw(c, kn, f[:, 2])
    cv2.putText(c, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return c


def render_compare(orig, recon, fps, out_path):
    L = min(len(orig), len(recon))
    # shared canvas scale from the original clip's full motion
    op = positions_from_offsets(orig[:L].reshape(L, NUM_JOINTS, 3)[..., :2])
    fitp = fit_params(op)
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False); tmp.close()
    wr = cv2.VideoWriter(tmp.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (2 * W, H))
    for t in range(L):
        left = panel(orig[t], fitp, "original")
        right = panel(recon[t], fitp, "recon (4cb)")
        wr.write(cv2.cvtColor(np.concatenate([left, right], axis=1), cv2.COLOR_RGB2BGR))
    wr.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp.name,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_path)], check=True)
    Path(tmp.name).unlink()
    return fitp, L


def metrics(orig, recon):
    L = min(len(orig), len(recon))
    o = orig[:L].reshape(L, NUM_JOINTS, 3)
    r = recon[:L].reshape(L, NUM_JOINTS, 3)
    off_mse = float(((o[..., :2] - r[..., :2]) ** 2).mean())
    off_mae = float(np.abs(o[..., :2] - r[..., :2]).mean())
    conf_mae = float(np.abs(o[..., 2] - r[..., 2]).mean())
    # per-joint Euclidean position error after kinematic chain (normalised offset space)
    po = positions_from_offsets(o[..., :2]); pr = positions_from_offsets(r[..., :2])
    joint_rmse = float(np.sqrt(((po - pr) ** 2).sum(-1)).mean())
    return dict(frames=L, off_mse=off_mse, off_mae=off_mae,
                conf_mae=conf_mae, joint_rmse=joint_rmse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=5)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--out-dir", default="assess_out")
    args = ap.parse_args()

    out_dir = Path("/home/james/pose-tokenizer") / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = PoseTokenizer.from_pretrained(MODEL_DIR, device=device)
    print(f"model n_codebooks={tok.config.n_codebooks}  downsample={tok.model.downsampling_factor}x  "
          f"input_features={tok.config.input_features}  norm_stats={tok.has_norm_stats}\n")

    # NOTE: KP_ROOT holds ~1.5M entries on a network FS, so never enumerate/sort
    # all of it. Pull just the first N directories via scandir with an early break.
    import os
    clip_dirs = []
    with os.scandir(KP_ROOT) as it:
        for entry in it:
            if entry.is_dir():
                clip_dirs.append(Path(entry.path))
                if len(clip_dirs) >= args.num:
                    break
    first_frames, rows = [], []
    for i, cd in enumerate(clip_dirs):
        chunk = sorted(cd.glob("*_poses_chunk*.safetensors"))[0]
        sd = load_file(str(chunk))
        orig = sd["reconstruction"].numpy().astype(np.float32)        # (T,165) ground truth

        x = torch.from_numpy(orig).float()                            # (T,165)
        codes = tok.encode(x)                                         # list of n_codebooks (T_codes,)
        recon = tok.decode(codes).cpu().numpy().astype(np.float32)    # (T_out,165)

        m = metrics(orig, recon)
        rows.append((cd.name, m))
        vid = out_dir / f"ex{i}_{cd.name[:18]}_compare.mp4"
        fitp, L = render_compare(orig, recon, args.fps, vid)

        # first-frame original|recon panel for the montage
        ff = np.concatenate([panel(orig[0], fitp, f"#{i} orig"),
                             panel(recon[0], fitp, "recon")], axis=1)
        first_frames.append(ff)
        print(f"[{i}] {cd.name[:28]}  Tin={len(orig)} Tcodes={codes[0].shape[-1]} Tout={len(recon)}  "
              f"off_MSE={m['off_mse']:.4e} off_MAE={m['off_mae']:.4f} "
              f"conf_MAE={m['conf_mae']:.4f} jointRMSE={m['joint_rmse']:.4f}")

    montage = np.concatenate(first_frames, axis=0)  # stack examples vertically
    cv2.imwrite(str(out_dir / "first_frame_compare_montage.png"),
                cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))

    agg = {k: float(np.mean([m[k] for _, m in rows]))
           for k in ("off_mse", "off_mae", "conf_mae", "joint_rmse")}
    print(f"\nMEAN over {len(rows)} examples: "
          f"off_MSE={agg['off_mse']:.4e}  off_MAE={agg['off_mae']:.4f}  "
          f"conf_MAE={agg['conf_mae']:.4f}  jointRMSE={agg['joint_rmse']:.4f}")
    print(f"outputs in {out_dir}/")


if __name__ == "__main__":
    main()
