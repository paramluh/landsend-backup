#!/usr/bin/env python3
"""Correlation between consecutive output stills' input-contribution profiles,
split by even-first (within a token-pair: 2k,2k+1) vs odd-first (across a token
boundary: 2k+1,2k+2)."""
import sys; sys.path.insert(0, "/home/james/pose-tokenizer")
import numpy as np, torch
from safetensors.torch import load_file
from pose_tokenizer import PoseTokenizer

tok = PoseTokenizer.from_pretrained(
    "/home/james/vansh-video-processing/pose-processing/pose-tokenizer-2x-12cb-lossshift1", device="cuda")
m = tok.model; dev = tok.device
f = "/mnt/som-fs/hands/pose-conditioning-keypoints/125194882901565440_00-00.00_to_00-15.03/125194882901565440_00-00.00_to_00-15.03_poses_chunk0000.safetensors"
o = load_file(f)["reconstruction"].numpy().astype("float32")
T_in = o.shape[0]
x0 = torch.from_numpy(o).float().unsqueeze(0).to(dev)
EPS, K = 0.05, 8


@torch.no_grad()
def recon_cont(x):
    return m.postprocess(m.decoder(m.encoder(m.preprocess(x))))


base = recon_cont(x0)[0].cpu().numpy(); T_out = base.shape[0]
M = np.zeros((T_out, T_in)); rng = np.random.default_rng(0)
for i in range(T_in):
    acc = np.zeros(T_out)
    for _ in range(K):
        x = x0.clone()
        d = torch.from_numpy(rng.standard_normal(o.shape[1]).astype("float32")).to(dev); d /= d.norm()
        x[0, i] = x[0, i] + EPS * d
        acc += np.linalg.norm(recon_cont(x)[0].cpu().numpy() - base, axis=1)
    M[:, i] = acc / K / EPS


def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


WARMUP = 6  # skip boundary stills whose window is clipped / cold-start
even_first, odd_first = [], []   # even_first = within-token pair, odd_first = across-token
for s in range(WARMUP, T_out - 1):
    c = pearson(M[s], M[s + 1])
    (even_first if s % 2 == 0 else odd_first).append(c)

print(f"chunk T_out={T_out}, comparing consecutive pairs (skipping first {WARMUP} stills)\n")
print(f"EVEN-first pairs (2k, 2k+1)  WITHIN a token : n={len(even_first):2d}  "
      f"mean corr={np.mean(even_first):.4f}  median={np.median(even_first):.4f}")
print(f"ODD-first  pairs (2k+1, 2k+2) ACROSS a token: n={len(odd_first):2d}  "
      f"mean corr={np.mean(odd_first):.4f}  median={np.median(odd_first):.4f}")
print(f"\ndifference (within - across) = {np.mean(even_first) - np.mean(odd_first):+.4f}")
