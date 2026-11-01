#!/usr/bin/env python3
"""Save a histogram for EVERY output still in the chunk (which input stills
inform it), 8 histograms per PNG.

Each bar = % of that output still's content explained by the given input still
(continuous-autoencoder forward-difference sensitivity, row-normalised).
"""
import sys; sys.path.insert(0, "/home/james/pose-tokenizer")
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
OUT_DIR = Path("/home/james/pose-tokenizer/decoder_evidence/contrib_hists")
OUT_DIR.mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def recon_cont(x):
    return m.postprocess(m.decoder(m.encoder(m.preprocess(x))))


base = recon_cont(x0)[0].cpu().numpy()
T_out = base.shape[0]
M = np.zeros((T_out, T_in))
rng = np.random.default_rng(0)
for i in range(T_in):
    acc = np.zeros(T_out)
    for _ in range(K):
        x = x0.clone()
        d = torch.from_numpy(rng.standard_normal(o.shape[1]).astype("float32")).to(dev)
        d = d / d.norm()
        x[0, i] = x[0, i] + EPS * d
        acc += np.linalg.norm(recon_cont(x)[0].cpu().numpy() - base, axis=1)
    M[:, i] = acc / K / EPS
Mp = M / (M.sum(1, keepdims=True) + 1e-9) * 100

per_png = 8
n_png = (T_out + per_png - 1) // per_png
for p in range(n_png):
    fig, axes = plt.subplots(2, 4, figsize=(18, 7))
    for j, ax in enumerate(axes.flat):
        s = p * per_png + j
        if s >= T_out:
            ax.axis("off"); continue
        ax.bar(range(T_in), Mp[s], color="#3b7dd8", width=1.0)
        ax.bar([1], [Mp[s, 1]], color="#d83b3b", width=1.0)  # highlight input still 1
        ax.set_title(f"output still {s} (t={s*0.04:.2f}s)")
        ax.set_xlabel("input still"); ax.set_ylabel("%")
        ax.set_xlim(-0.5, T_in - 0.5)
    fig.suptitle(f"What informs each output still — part {p} (stills {p*per_png}..{min((p+1)*per_png,T_out)-1})  "
                 f"[red = input still 1]", fontsize=13)
    fig.tight_layout()
    path = OUT_DIR / f"part_{p:02d}.png"
    fig.savefig(path, dpi=100); plt.close(fig)
    print("wrote", path)

print(f"\n{n_png} PNGs ({per_png} histograms each) in {OUT_DIR}  | T_out={T_out} stills")
