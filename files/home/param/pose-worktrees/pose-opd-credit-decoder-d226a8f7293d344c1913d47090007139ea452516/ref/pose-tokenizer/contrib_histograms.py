#!/usr/bin/env python3
"""For each of the first 8 output stills (= first 4 token-frames decoded),
histogram which INPUT stills inform it (continuous Jacobian-Frobenius
sensitivity). Saves a 2x4 grid of bar charts + prints the matrix.
"""
import sys; sys.path.insert(0, "/home/james/pose-tokenizer")
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from safetensors.torch import load_file
from pose_tokenizer import PoseTokenizer

tok = PoseTokenizer.from_pretrained(
    "/home/james/vansh-video-processing/pose-processing/pose-tokenizer-2x-12cb-lossshift1", device="cuda")
m = tok.model; dev = tok.device
f = "/mnt/som-fs/hands/pose-conditioning-keypoints/125194882901565440_00-00.00_to_00-15.03/125194882901565440_00-00.00_to_00-15.03_poses_chunk0000.safetensors"
o = load_file(f)["reconstruction"].numpy().astype("float32")
x0 = torch.from_numpy(o).float().unsqueeze(0).to(dev)
N_OUT, N_IN, N_PROBES = 8, 14, 48


def recon_cont(x):
    return m.postprocess(m.decoder(m.encoder(m.preprocess(x))))


def contributions(s_out):
    acc = torch.zeros(N_IN, device=dev)
    for _ in range(N_PROBES):
        x = x0.clone().requires_grad_(True)
        r = recon_cont(x)
        v = torch.randn(r.shape[-1], device=dev)
        g, = torch.autograd.grad((r[0, s_out, :] * v).sum(), x)
        acc += (g[0, :N_IN, :] ** 2).sum(-1)
    return torch.sqrt(acc / N_PROBES).cpu().numpy()


M = np.stack([contributions(s) for s in range(N_OUT)])   # (8, N_IN) sensitivity
Mp = M / M.sum(1, keepdims=True) * 100                    # row-normalised %

print("contribution % [output still -> input still]:")
header = "out\\in " + " ".join(f"{i:4d}" for i in range(N_IN))
print(header)
for s in range(N_OUT):
    print(f"  {s:2d}   " + " ".join(f"{Mp[s,i]:4.0f}" for i in range(N_IN)))

# which output still does input still 1 contribute most to?
col1 = M[:, 1]
print(f"\ninput still 1 (t=0.04s) contributes MOST to output still {int(col1.argmax())} "
      f"(t={0.04*col1.argmax():.2f}s); its share there = {Mp[col1.argmax(),1]:.0f}% of that still")
print("input still 1 raw sensitivity per output still:",
      " ".join(f"{c:.2f}" for c in col1))

fig, axes = plt.subplots(2, 4, figsize=(18, 7), sharex=True)
for s, ax in enumerate(axes.flat):
    ax.bar(range(N_IN), Mp[s], color="#3b7dd8")
    ax.bar([1], [Mp[s, 1]], color="#d83b3b")  # highlight input still 1
    ax.set_title(f"output still {s}  (t={s*0.04:.2f}s)")
    ax.set_xlabel("input still"); ax.set_ylabel("% contribution")
    ax.set_xticks(range(0, N_IN, 2))
fig.suptitle("What informs each output still (red = input still 1 @ t=0.04s)", fontsize=14)
fig.tight_layout()
out = "/home/james/pose-tokenizer/decoder_evidence/contrib_histograms.png"
fig.savefig(out, dpi=110); print(f"\nwrote {out}")
