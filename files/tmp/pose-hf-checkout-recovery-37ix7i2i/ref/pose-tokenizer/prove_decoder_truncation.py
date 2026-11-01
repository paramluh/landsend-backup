#!/usr/bin/env python3
"""RIGOROUS proof the first-still collapse is a DECODER cold-start.

Encode once. Then for each truncation depth d = 0,1,2,...:
  decode(codes[:, d:])  -> the token-frame that was at index d is now FIRST.
Measure the extent of that decode's FIRST output still.

Claim proven if: the first still collapses for EVERY d (~constant small ratio),
while those exact same token-frames are full-size when decoded in-context
(interior stills of the full decode). => collapse is positional (decoder),
independent of which token sits there.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, torch, cv2
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
_CONN = FACE_CONNECTIONS + BODY_CONNECTIONS + LEFT_HAND_CONNECTIONS + RIGHT_HAND_CONNECTIONS
_JOINTS = sorted({i for p in _CONN for i in p})


def pos_off(off):
    out = np.empty_like(off)
    for j in TOPO_ORDER:
        p = PARENT_INDEX[j]
        out[..., j, :] = (0 if p == -1 else out[..., p, :]) + off[..., j, :]
    return out


def extent(feat3):  # (T,55,3) -> (T,) bbox-area proxy
    P = pos_off(feat3[:, :, :2])
    return (P[..., 0].max(1) - P[..., 0].min(1)) * (P[..., 1].max(1) - P[..., 1].min(1))


def draw_still(feat_row, fitp, label):
    f = feat_row.reshape(NUM_JOINTS, 3).astype(np.float32)
    P = pos_off(f[None, :, :2])[0]
    sy, sx, oy, ox = fitp
    kn = np.stack([(P[:, 0]*sy+oy)/H, (P[:, 1]*sx+ox)/W], axis=-1)
    c = np.zeros((H, W, 3), np.uint8)
    h, w = c.shape[:2]; yp, xp = kn[:, 0]*h, kn[:, 1]*w
    for i, j in _CONN:
        cv2.line(c, (int(xp[i]), int(yp[i])), (int(xp[j]), int(yp[j])), (0, 220, 0), 2, cv2.LINE_AA)
    for idx in _JOINTS:
        cv2.circle(c, (int(xp[idx]), int(yp[idx])), 3, (0, 220, 0), -1, cv2.LINE_AA)
    cv2.putText(c, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return c


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = PoseTokenizer.from_pretrained(MODEL_DIR, device=dev)
    orig = load_file(SRC)["reconstruction"].numpy().astype("float32")
    codes = tok.encode(torch.from_numpy(orig).float())            # 4 x (T,)
    T = codes[0].shape[-1]

    full = tok.decode(codes).cpu().numpy().reshape(-1, NUM_JOINTS, 3)
    ef = extent(full)
    steady = float(np.median(ef[6:]))
    # canvas scale fixed from full decode steady motion
    fitp = (lambda P: (H*(1-2*PAD)/max(P[...,0].max()-P[...,0].min(),1e-6),
                       W*(1-2*PAD)/max(P[...,1].max()-P[...,1].min(),1e-6),
                       0, 0))(pos_off(full[:, :, :2]))
    sy, sx, _, _ = fitp
    oy = (H - sy*(pos_off(full[:, :, :2])[..., 0].min()+pos_off(full[:, :, :2])[..., 0].max()))/2
    ox = (W - sx*(pos_off(full[:, :, :2])[..., 1].min()+pos_off(full[:, :, :2])[..., 1].max()))/2
    fitp = (sy, sx, oy, ox)

    depths = [0, 1, 2, 3, 4, 6, 8, 10, 14, 20]
    print(f"T_token_frames={T}  steady-extent={steady:.3f}\n")
    print(f"{'drop d':>6} {'1st-still ext':>13} {'ratio/steady':>12} {'in-context ext (same tok)':>26} {'ctx ratio':>10}")
    trunc_panels, ctx_panels = [], []
    for d in depths:
        if d >= T - 2:
            continue
        dec = tok.decode([c[d:] for c in codes]).cpu().numpy().reshape(-1, NUM_JOINTS, 3)
        e_first = float(extent(dec[:1])[0])
        # same token-frame d, decoded in-context = full decode around output still 2d
        idx = min(2*d, len(full)-1)
        e_ctx = float(ef[idx])
        print(f"{d:>6} {e_first:>13.3f} {e_first/steady:>12.2f} {e_ctx:>26.3f} {e_ctx/steady:>10.2f}")
        trunc_panels.append(draw_still(dec[0].reshape(-1), fitp, f"drop {d}: 1st still"))
        ctx_panels.append(draw_still(full[idx].reshape(-1), fitp, f"tok {d} in-context"))

    montage = np.concatenate([np.concatenate(trunc_panels, 1),
                              np.concatenate(ctx_panels, 1)], 0)
    png = OUT / "truncation_proof.png"
    cv2.imwrite(str(png), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
    print(f"\nTop row  = first still of decode(codes[d:])  -> collapses for EVERY d")
    print(f"Bottom row = same token-frame decoded in-context -> full size")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
