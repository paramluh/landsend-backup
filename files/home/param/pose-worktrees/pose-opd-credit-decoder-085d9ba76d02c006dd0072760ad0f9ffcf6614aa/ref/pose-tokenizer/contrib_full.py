#!/usr/bin/env python3
"""Full input-still -> output-still contribution map for one chunk.

Method: continuous autoencoder (encoder->decoder, NO quantizer). For each INPUT
still i we apply several small random perturbations and measure the resulting
change at every OUTPUT still s (forward finite-difference). The averaged
magnitude estimates the Jacobian sensitivity ||d out[s] / d in[i]||. Each row
(output still) is normalised to 100%, so a cell = % of that output still's
content explained by input still i.

Outputs a heatmap over the WHOLE chunk + clean steady-state bar histograms.
"""
import sys; sys.path.insert(0, "/home/james/pose-tokenizer")
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


@torch.no_grad()
def recon_cont(x):
    return m.postprocess(m.decoder(m.encoder(m.preprocess(x))))


base = recon_cont(x0)[0].cpu().numpy()          # (T_out, 165)
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
        r = recon_cont(x)[0].cpu().numpy()
        acc += np.linalg.norm(r - base, axis=1)
    M[:, i] = acc / K / EPS                      # ~ ||d out[s]/d in[i]||

Mp = M / (M.sum(1, keepdims=True) + 1e-9) * 100  # row-normalised %

print(f"chunk: T_in={T_in} stills (input), T_out={T_out} stills (output)")
print("input still 1 contributes most to output still",
      int(M[:, 1].argmax()), f"(t={0.04*M[:,1].argmax():.2f}s)")

fig = plt.figure(figsize=(15, 11))
# --- heatmap over the whole chunk ---
ax1 = fig.add_subplot(2, 1, 1)
im = ax1.imshow(Mp, aspect="auto", origin="lower", cmap="viridis",
                extent=[0, T_in, 0, T_out])
# Autoencoder is same-rate (25fps in and out): output still s aligns with input
# still s. The 2x downsample is INTERNAL (to 12.5Hz tokens), not in/out stills.
ax1.plot([0, T_out], [0, T_out], "w--", lw=0.8, alpha=0.6,
         label="output still = input still (causal frontier)")
ax1.set_ylim(0, T_out)
ax1.set_xlabel("INPUT still index  (25fps, 0.04s each)")
ax1.set_ylabel("OUTPUT still index")
ax1.set_title("Contribution map: % of each output still explained by each input still (whole chunk)")
ax1.legend(loc="lower right")
fig.colorbar(im, ax=ax1, label="% contribution (row-normalised)")

# --- clean steady-state bar histograms (away from the boundary) ---
samples = [2, 20, 40, T_out - 4]
for j, s in enumerate(samples):
    ax = fig.add_subplot(2, len(samples), len(samples) + j + 1)
    lo, hi = max(0, 2 * s // 2 - 14), min(T_in, 2 * s // 2 + 2)
    ax.bar(range(T_in), Mp[s], color="#3b7dd8")
    ax.set_xlim(lo, hi)
    ax.set_title(f"output still {s} (t={s*0.04:.2f}s)")
    ax.set_xlabel("input still"); 
    if j == 0: ax.set_ylabel("% contribution")
fig.suptitle("Bottom: per-output-still windows — boundary (s=2) vs steady-state (s=20,40,end)", y=0.46)
fig.tight_layout()
out = "/home/james/pose-tokenizer/decoder_evidence/contrib_full.png"
fig.savefig(out, dpi=110); print("wrote", out)
