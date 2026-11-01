#!/usr/bin/env python3
"""Decode N examples from pose-4cb-unchunked-uncut and sanity-check them.

For each example:
  - decode the 4 codebooks -> reconstruction (T,165)
  - load the ground-truth pose (pose-unchunked-uncut reconstruction)
  - report L1 (kp offsets) between GT and 4cb-decoded
  - render a side-by-side conf-colored skeleton video: GT | 4cb-decoded
"""
from __future__ import annotations

import argparse
import random
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
from safetensors import safe_open

from pose_tokenizer import PoseTokenizer
from pose_tokenizer.data.rendering import (
    _ALL_CONNECTIONS, _ALL_JOINTS, kp_flat_to_positions,
)

REPO = "xabirizar9/pose-tokenizer-2x-12cb-lossshift1"
TOK = Path("/mnt/som-fs/hands/aloe-encoded/pose-4cb-unchunked-uncut")
GT = Path("/mnt/som-fs/hands/aloe-encoded/pose-unchunked-uncut")
OUT = Path("/home/james/pose-tokenizer/decode_check_4cb")
FPS, H, W = 25.0, 768, 768


def split_165(feat: np.ndarray):
    r = feat.reshape(feat.shape[0], 55, 3)
    return r[..., :2].reshape(feat.shape[0], 110), r[..., 2]


def conf_color(c):
    c = float(np.clip(c, 0, 1))
    return (int(255 * c), int(255 * (1 - c)), 0)


def draw(canvas, pos, cf):
    yp = pos[:, 0] * H; xp = pos[:, 1] * W
    for i, j in _ALL_CONNECTIONS:
        col = conf_color((float(cf[i]) + float(cf[j])) * 0.5) if cf is not None else (0, 220, 0)
        cv2.line(canvas, (int(xp[i]), int(yp[i])), (int(xp[j]), int(yp[j])), col, 2, cv2.LINE_AA)
    for idx in _ALL_JOINTS:
        col = conf_color(float(cf[idx])) if cf is not None else (0, 220, 0)
        cv2.circle(canvas, (int(xp[idx]), int(yp[idx])), 3, col, -1, cv2.LINE_AA)


def render(kp_flat, conf, gs, out_path):
    n = min(kp_flat.shape[0], gs.shape[0])
    pos = kp_flat_to_positions(kp_flat[:n], gs[:n])
    conf = conf[:n] if conf is not None else None
    tmp = out_path.with_suffix(".tmp.mp4")
    wr = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for t in range(n):
        c = np.zeros((H, W, 3), np.uint8)
        draw(c, pos[t], None if conf is None else conf[t])
        wr.write(cv2.cvtColor(c, cv2.COLOR_RGB2BGR))
    wr.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_path)], check=True)
    tmp.unlink(missing_ok=True)


def sbs(left, right, out):
    flt = ("[0:v]scale=768:768,drawtext=text='GT':x=10:y=10:fontcolor=white:fontsize=28[l];"
           "[1:v]scale=768:768,drawtext=text='4cb decode':x=10:y=10:fontcolor=white:fontsize=28[r];"
           "[l][r]hstack=inputs=2[v]")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(left), "-i", str(right),
                    "-filter_complex", flt, "-map", "[v]", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(out)], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = args.device if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.enabled = False
    tok = PoseTokenizer.from_pretrained(REPO, device=dev)

    files = sorted(TOK.glob("*.safetensors"))
    random.seed(args.seed)
    sample = random.sample(files, args.n)
    print(f"decoding {args.n} examples -> {OUT}\n")
    for k, f in enumerate(sample):
        with safe_open(str(f), framework="pt") as h:
            tokens = h.get_tensor("tokens").long()      # (4, Ttok)
            gs = h.get_tensor("global_state").float().numpy()
        codes = [tokens[c].to(dev) for c in range(tokens.shape[0])]
        with torch.inference_mode():
            recon = tok.decode(codes).cpu().float().numpy()   # (T,165)
        dkp, dconf = split_165(recon)
        with safe_open(str(GT / f.name), framework="pt") as h:
            gtr = h.get_tensor("reconstruction").float().numpy()
        gkp, gconf = split_165(gtr)
        m = min(dkp.shape[0], gkp.shape[0])
        l1 = float(np.abs(dkp[:m] - gkp[:m]).mean())
        stem = f.name[:-len(".safetensors")]
        lp = OUT / f"{k:02d}_gt.mp4"; rp = OUT / f"{k:02d}_4cb.mp4"
        render(gkp, gconf, gs, lp)
        render(dkp, dconf, gs, rp)
        sbs(lp, rp, OUT / f"{k:02d}_sbs.mp4")
        print(f"[{k}] T_gt={gkp.shape[0]} T_dec={dkp.shape[0]} kp_L1={l1:.4f}  {stem[:46]}")
    print(f"\ndone -> {OUT} (see *_sbs.mp4)")


if __name__ == "__main__":
    main()
