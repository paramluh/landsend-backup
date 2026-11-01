#!/usr/bin/env python3
"""Encode->decode 32 aloe chunks (a) batched (B=32) and (b) one-at-a-time, then
check they are mathematically the same, and render a side-by-side for one chunk.
Uses 4 codebooks (production)."""
import sys, glob, os
from pathlib import Path
import numpy as np, torch
from safetensors.torch import load_file
sys.path.insert(0, "/home/james/pose-tokenizer")
from pose_tokenizer import PoseTokenizer
from pose_tokenizer.data.rendering import render_comparison_video
from pose_tokenizer.data.kinematic import NUM_JOINTS

ROOT = "/mnt/som-fs/hands/aloe/v1-poses"
NCB = 4
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = PoseTokenizer.from_pretrained("xabirizar9/pose-tokenizer-2x-12cb-lossshift1", device=dev)
tok.model.set_active_codebooks(NCB)

# gather 32 chunks (+ their global_state)
paths = []
with os.scandir(ROOT) as it:
    for e in it:
        if e.is_dir():
            paths += sorted(glob.glob(e.path + "/*_poses_chunk*.safetensors"))
        if len(paths) >= 32:
            break
paths = paths[:32]
data = [load_file(p) for p in paths]
arrs = [d["reconstruction"].numpy().astype("float32") for d in data]
lens = [a.shape[0] for a in arrs]
print(f"{len(arrs)} chunks, frame-lengths: min={min(lens)} max={max(lens)}")

# ---- (a) BATCHED (pad to max length on the right) ----
Tmax = max(lens)
batch = np.stack([np.pad(a, ((0, Tmax - a.shape[0]), (0, 0))) for a in arrs])  # (32,Tmax,165)
with torch.no_grad():
    codes_b = tok.encode(torch.from_numpy(batch).float())          # list[NCB..] of (32, Tc)
    recon_b = tok.decode([c for c in codes_b[:NCB]]).cpu().numpy() # (32, Tout, 165)

# ---- (b) ONE-AT-A-TIME ----
recon_u = []
with torch.no_grad():
    for a in arrs:
        c = tok.encode(torch.from_numpy(a).float())
        recon_u.append(tok.decode([cc for cc in c[:NCB]]).cpu().numpy())  # (Touti,165)

# also get unbatched CODES + continuous latent for diagnosis
codes_u = []
with torch.no_grad():
    for a in arrs:
        codes_u.append([c.cpu().numpy() for c in tok.encode(torch.from_numpy(a).float())])

m = tok.model
def cont_latent(x):  # encoder output, BEFORE quantization
    with torch.no_grad():
        return m.encoder(m.preprocess(x.to(tok.device))).cpu().numpy()

# ---- compare on the real (non-padded) frames ----
print(f"\n{'chunk':>5} {'len':>4} {'recon_max|diff|':>15} {'#code-frames flipped (of NCB*Tc)':>32} {'cont_latent_max|diff|':>22}")
recon_diffs, code_flips = [], []
for i in range(len(arrs)):
    L = recon_u[i].shape[0]
    rd = float(np.abs(recon_b[i, :L] - recon_u[i]).max())
    # codes: batched codes_b[k][i,:Tci] vs unbatched codes_u[i][k]
    Tci = codes_u[i][0].shape[-1]
    flips = sum(int((codes_b[k][i, :Tci].cpu().numpy() != codes_u[i][k]).sum()) for k in range(len(codes_u[i])))
    total = len(codes_u[i]) * Tci
    # continuous latent diff (batched vs unbatched) for this chunk
    lb = cont_latent(torch.from_numpy(batch[i:i+1]).float())[0]
    lu = cont_latent(torch.from_numpy(arrs[i]).float().unsqueeze(0))[0]
    Lc = min(lb.shape[-1], lu.shape[-1])
    cont_d = float(np.abs(lb[:, :Lc] - lu[:, :Lc]).max())
    recon_diffs.append(rd); code_flips.append(flips)
    if i < 12:
        print(f"{i:>5} {lens[i]:>4} {rd:>15.2e} {f'{flips}/{total}':>32} {cont_d:>22.2e}")
recon_diffs = np.array(recon_diffs)
print(f"\nrecon max|diff|: overall-max={recon_diffs.max():.2e}  mean={recon_diffs.mean():.2e}")
print(f"chunks with ANY code flip: {sum(f>0 for f in code_flips)}/{len(code_flips)}")

# ---- side-by-side render for chunk 0: unbatched (green) vs batched (red) ----
gs = data[0]["global_state"].numpy().astype("float32")
L = min(recon_u[0].shape[0], gs.shape[0])
def split(r): 
    r3 = r[:L].reshape(L, NUM_JOINTS, 3); return r3[..., :2].reshape(L, -1), r3[..., 2]
u_kp, u_cf = split(recon_u[0]); b_kp, b_cf = split(recon_b[0])
out = Path("/home/james/pose-tokenizer/batch_vs_unbatch_compare.mp4")
render_comparison_video(orig_kp=u_kp, recon_kp=b_kp, global_state=gs[:L], confidence=u_cf,
                        out_path=out, fps=25.0, max_seconds=10.0, conf_thresh=0.0, recon_confidence=b_cf)
print(f"\nwrote {out}  (left=UNBATCHED, right=BATCHED)")
