#!/usr/bin/env python3
"""Save one video per truncation depth: decode(codes[d:]) rendered as a skeleton
clip, so you can watch that the FIRST still collapses no matter where the cut is.

All videos share one canvas scale (from the full decode's steady motion) so the
collapse is directly comparable across truncations.
Outputs: decoder_evidence/trunc_d{d}.mp4
"""
from __future__ import annotations
import subprocess, sys, tempfile
from pathlib import Path
import cv2, numpy as np, torch
from safetensors.torch import load_file

sys.path.insert(0, "/home/james/pose-tokenizer")
from pose_tokenizer import PoseTokenizer
from pose_tokenizer.data.kinematic import (
    NUM_JOINTS, PARENT_INDEX, TOPO_ORDER,
    FACE_CONNECTIONS, BODY_CONNECTIONS, LEFT_HAND_CONNECTIONS, RIGHT_HAND_CONNECTIONS,
)
MODEL_DIR = "/home/james/vansh-video-processing/pose-processing/pose-tokenizer-2x-12cb-lossshift1"
SRC = "/mnt/som-fs/hands/pose-conditioning-keypoints/125194882901565440_00-00.00_to_00-15.03/125194882901565440_00-00.00_to_00-15.03_poses_chunk0000.safetensors"
OUT = Path("/home/james/pose-tokenizer/decoder_evidence")
W, H, PAD = 256, 480, 0.08
FPS = 6.0   # slow so the first-frame collapse is easy to see
DEPTHS = [0, 1, 2, 3, 4, 6, 8, 10]
_CONN = FACE_CONNECTIONS + BODY_CONNECTIONS + LEFT_HAND_CONNECTIONS + RIGHT_HAND_CONNECTIONS
_JOINTS = sorted({i for p in _CONN for i in p})


def pos_off(off):
    out = np.empty_like(off)
    for j in TOPO_ORDER:
        p = PARENT_INDEX[j]
        out[..., j, :] = (0 if p == -1 else out[..., p, :]) + off[..., j, :]
    return out


def fit_params(P):
    y, x = P[..., 0], P[..., 1]
    sy = H*(1-2*PAD)/max(y.max()-y.min(), 1e-6); sx = W*(1-2*PAD)/max(x.max()-x.min(), 1e-6)
    return sy, sx, (H-sy*(y.min()+y.max()))/2, (W-sx*(x.min()+x.max()))/2


def render(feats3, fitp, label, out_path):
    sy, sx, oy, ox = fitp
    P = pos_off(feats3[:, :, :2])
    kn = np.stack([(P[..., 0]*sy+oy)/H, (P[..., 1]*sx+ox)/W], axis=-1)
    conf = feats3[:, :, 2]
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False); tmp.close()
    wr = cv2.VideoWriter(tmp.name, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for t in range(len(feats3)):
        c = np.zeros((H, W, 3), np.uint8); h, w = c.shape[:2]
        yp, xp = kn[t, :, 0]*h, kn[t, :, 1]*w
        for i, j in _CONN:
            cc = (0, 255, 0) if t > 0 else (60, 60, 255)   # frame 0 tinted red
            cv2.line(c, (int(xp[i]), int(yp[i])), (int(xp[j]), int(yp[j])), cc, 2, cv2.LINE_AA)
        for idx in _JOINTS:
            cv2.circle(c, (int(xp[idx]), int(yp[idx])), 3, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.putText(c, f"{label}  frame{t}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        wr.write(cv2.cvtColor(c, cv2.COLOR_RGB2BGR))
    wr.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp.name,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_path)], check=True)
    Path(tmp.name).unlink()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = PoseTokenizer.from_pretrained(MODEL_DIR, device=dev)
    orig = load_file(SRC)["reconstruction"].numpy().astype("float32")
    codes = tok.encode(torch.from_numpy(orig).float())
    T = codes[0].shape[-1]

    full = tok.decode(codes).cpu().numpy().reshape(-1, NUM_JOINTS, 3)
    fitp = fit_params(pos_off(full[:, :, :2]))   # shared scale across all videos

    for d in DEPTHS:
        if d >= T - 2:
            continue
        dec = tok.decode([c[d:] for c in codes]).cpu().numpy().reshape(-1, NUM_JOINTS, 3)
        out_path = OUT / f"trunc_d{d:02d}.mp4"
        render(dec, fitp, f"drop {d} tok-frames", out_path)
        print(f"drop {d:2d}: {dec.shape[0]} stills -> {out_path.name}  (frame0 is the red cold-start still)")

    print(f"\nall truncation videos in {OUT}/  (red skeleton = frame 0 = decoder cold-start)")


if __name__ == "__main__":
    main()
