#!/usr/bin/env python3
"""Encode+decode 32 aloe chunks BATCHED (bs=32) vs ONE-AT-A-TIME, to check that
batching doesn't change the output. Prints per-chunk diff + GT recon error, and
renders a side-by-side video (left = no-batch, right = batch-32) for one chunk.
"""
import os, glob, sys
from pathlib import Path
import numpy as np, torch
from safetensors.torch import load_file
sys.path.insert(0, "/home/james/pose-tokenizer")
from pose_tokenizer import PoseTokenizer
from pose_tokenizer.data.rendering import render_comparison_video
from pose_tokenizer.data.kinematic import NUM_JOINTS

ROOT = "/mnt/som-fs/hands/aloe/v1-poses"
NCB = 4
BS = 32
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = PoseTokenizer.from_pretrained("xabirizar9/pose-tokenizer-2x-12cb-lossshift1", device=dev)
tok.model.set_active_codebooks(NCB)

# gather 32 chunks that are exactly 77 frames (so no padding needed)
paths, origs, gss = [], [], []
with os.scandir(ROOT) as it:
    for e in it:
        if not e.is_dir():
            continue
        for f in glob.glob(e.path + "/*_poses_chunk*.safetensors"):
            sd = load_file(f)
            a = sd["reconstruction"].numpy().astype("float32")
            if a.shape[0] == 77:
                paths.append(f); origs.append(a); gss.append(sd["global_state"].numpy().astype("float32"))
                break
        if len(paths) >= BS:
            break
print(f"gathered {len(paths)} chunks (77 frames each)")

# --- one-at-a-time ---
single = []
for a in origs:
    codes = tok.encode(torch.from_numpy(a).float())
    single.append(tok.decode([c for c in codes[:NCB]]).cpu().numpy())

# --- batched (bs=32) ---
X = torch.from_numpy(np.stack(origs)).float().to(dev)          # (32,77,165)
codesB = tok.encode(X)                                          # list[ncb] of (32,39)
reconB = tok.decode([c for c in codesB[:NCB]]).cpu().numpy()    # (32,78,165)

# --- compare ---
diffs = [float(np.abs(reconB[i] - single[i]).max()) for i in range(len(paths))]
errS = [float(np.abs(single[i][:77].reshape(77, NUM_JOINTS, 3)[..., :2]
               - origs[i].reshape(77, NUM_JOINTS, 3)[..., :2]).mean()) for i in range(len(paths))]
errB = [float(np.abs(reconB[i][:77].reshape(77, NUM_JOINTS, 3)[..., :2]
               - origs[i].reshape(77, NUM_JOINTS, 3)[..., :2]).mean()) for i in range(len(paths))]
print(f"max|batched - single| : max={max(diffs):.3e}  mean={np.mean(diffs):.3e}")
print(f"GT offset-MAE  single : {np.mean(errS):.5f}")
print(f"GT offset-MAE  batch32: {np.mean(errB):.5f}")

# --- side-by-side video for chunk 0: left = no-batch, right = batch-32 ---
def deshift(r, s=1):
    return np.concatenate([r[s:], r[-1:].repeat(s, 0)], 0) if s > 0 else r
i = 0
L = 77
s3 = deshift(single[i]).reshape(-1, NUM_JOINTS, 3)[:L]
b3 = deshift(reconB[i]).reshape(-1, NUM_JOINTS, 3)[:L]
render_comparison_video(
    orig_kp=s3[..., :2].reshape(L, NUM_JOINTS * 2),       # LEFT = single (no batch)
    recon_kp=b3[..., :2].reshape(L, NUM_JOINTS * 2),      # RIGHT = batch-32
    global_state=gss[i][:L], confidence=s3[..., 2],
    out_path=Path("/home/james/pose-tokenizer/batch_vs_single.mp4"),
    fps=25.0, conf_thresh=0.0, recon_confidence=b3[..., 2],
)
print("wrote batch_vs_single.mp4  (left=no-batch green, right=batch32 red)")
