#!/usr/bin/env python3
"""Visual evidence that the first-still collapse is a DECODER cold-start, not a
corrupted token.

We encode once, then decode two ways:
  A) decode ALL token-frames          -> still 0 is collapsed
  B) decode token-frames[1:] (drop the first token-frame), decode in isolation
If the corruption lived in the token, (B)'s new first still would be clean.
It is NOT -> the collapse follows the decoder's first output position.

Outputs:
  decoder_evidence_sidebyside.mp4   left=decode(all), right=decode(drop-first-token)
  decoder_evidence_firstframes.png  rows A/B x cols still0/1/2
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
_CONN = FACE_CONNECTIONS + BODY_CONNECTIONS + LEFT_HAND_CONNECTIONS + RIGHT_HAND_CONNECTIONS
_JOINTS = sorted({i for p in _CONN for i in p})


def pos_off(off):
    out = np.empty_like(off)
    for j in TOPO_ORDER:
        p = PARENT_INDEX[j]
        out[..., j, :] = (0 if p == -1 else out[..., p, :]) + off[..., j, :]
    return out


def fit(positions):
    y, x = positions[..., 0], positions[..., 1]
    y0, y1, x0, x1 = y.min(), y.max(), x.min(), x.max()
    sy = H * (1 - 2 * PAD) / max(y1 - y0, 1e-6); sx = W * (1 - 2 * PAD) / max(x1 - x0, 1e-6)
    oy = (H - sy * (y0 + y1)) / 2; ox = (W - sx * (x0 + x1)) / 2
    return sy, sx, oy, ox


def apply_fit(positions, p):
    sy, sx, oy, ox = p
    return np.stack([(positions[..., 0]*sy+oy)/H, (positions[..., 1]*sx+ox)/W], axis=-1)


def col(c):
    c = float(np.clip(c, 0, 1)); return (int(255*c), int(255*(1-c)), 0)


def draw(canvas, kn, conf, label=None):
    h, w = canvas.shape[:2]; yp, xp = kn[:, 0]*h, kn[:, 1]*w
    for i, j in _CONN:
        cv2.line(canvas, (int(xp[i]), int(yp[i])), (int(xp[j]), int(yp[j])),
                 col((conf[i]+conf[j])*0.5), 2, cv2.LINE_AA)
    for idx in _JOINTS:
        cv2.circle(canvas, (int(xp[idx]), int(yp[idx])), 3, col(conf[idx]), -1, cv2.LINE_AA)
    if label:
        cv2.putText(canvas, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = PoseTokenizer.from_pretrained(MODEL_DIR, device=dev)
    orig = load_file(SRC)["reconstruction"].numpy().astype("float32")
    codes = tok.encode(torch.from_numpy(orig).float())

    full = tok.decode(codes).cpu().numpy()                       # decode ALL
    drop1 = tok.decode([c[1:] for c in codes]).cpu().numpy()     # drop first token-frame

    fa = full.reshape(-1, NUM_JOINTS, 3); fb = drop1.reshape(-1, NUM_JOINTS, 3)
    # shared canvas scale from the full decode's steady motion
    fitp = fit(pos_off(fa[:, :, :2]))

    # ---- side-by-side video (slow fps so the first frame is visible) ----
    L = min(len(fa), len(fb))
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False); tmp.close()
    wr = cv2.VideoWriter(tmp.name, cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (2*W, H))
    for t in range(L):
        la = draw(np.zeros((H, W, 3), np.uint8), apply_fit(pos_off(fa[t, :, :2][None])[0], fitp),
                  fa[t, :, 2], f"decode ALL  t{t}")
        lb = draw(np.zeros((H, W, 3), np.uint8), apply_fit(pos_off(fb[t, :, :2][None])[0], fitp),
                  fb[t, :, 2], f"drop 1st tok  t{t}")
        wr.write(cv2.cvtColor(np.concatenate([la, lb], 1), cv2.COLOR_RGB2BGR))
    wr.release()
    sbs = OUT / "decoder_evidence_sidebyside.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp.name,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(sbs)], check=True)
    Path(tmp.name).unlink()

    # ---- first-frames montage: rows A/B, cols still0/1/2 ----
    rows = []
    for feats, tag in [(fa, "decode ALL"), (fb, "drop 1st tok")]:
        panels = []
        for t in range(3):
            c = draw(np.zeros((H, W, 3), np.uint8),
                     apply_fit(pos_off(feats[t, :, :2][None])[0], fitp),
                     feats[t, :, 2], f"{tag} still{t}")
            panels.append(c)
        rows.append(np.concatenate(panels, 1))
    montage = np.concatenate(rows, 0)
    png = OUT / "decoder_evidence_firstframes.png"
    cv2.imwrite(str(png), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))

    def ext(f):
        P = pos_off(f[:, :, :2]); return (P[..., 0].max(1)-P[..., 0].min(1))*(P[..., 1].max(1)-P[..., 1].min(1))
    print("decode ALL  still extents:", np.round(ext(fa)[:4], 3))
    print("drop 1st    still extents:", np.round(ext(fb)[:4], 3))
    print("wrote", sbs)
    print("wrote", png)


if __name__ == "__main__":
    main()
