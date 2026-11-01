#!/usr/bin/env python3
"""Encode -> decode ONE aloe pose chunk and render before|after side-by-side.

Uses the high-level PoseTokenizer.encode/.decode and the repo's
render_comparison_video (original = green, reconstructed = red).
"""
import argparse
import sys
from pathlib import Path
import numpy as np, torch
from safetensors.torch import load_file

sys.path.insert(0, "/home/james/pose-tokenizer")
from pose_tokenizer import PoseTokenizer
from pose_tokenizer.data.rendering import render_comparison_video
from pose_tokenizer.data.kinematic import NUM_JOINTS

DEFAULT_CHUNK = ("/mnt/som-fs/hands/aloe/v1-poses/"
                 "Nickolas_20260507_144432_seg05_00334_to_01917__speech_local_01001p55_to_01006p41__abs_01335p63_to_01340p49/"
                 "Nickolas_20260507_144432_seg05_00334_to_01917__speech_local_01001p55_to_01006p41__abs_01335p63_to_01340p49_poses_chunk0000.safetensors")

ap = argparse.ArgumentParser()
ap.add_argument("--chunk", default=DEFAULT_CHUNK)
ap.add_argument("--repo", default="xabirizar9/pose-tokenizer-2x-12cb-lossshift1")
ap.add_argument("--ncb", type=int, default=4, help="number of pose codebooks to decode with")
ap.add_argument("--shift", type=int, default=1, help="causal de-shift (advance recon)")
ap.add_argument("--out", default="/home/james/pose-tokenizer/aloe_recon_compare_4cb.mp4")
args = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = PoseTokenizer.from_pretrained(args.repo, device=dev)
ncb = len(tok.model.quantizer.quantizers)
print(f"tokenizer: {args.repo}  codebooks={ncb}  downsample={tok.model.downsampling_factor}x")

sd = load_file(args.chunk)
orig = sd["reconstruction"].numpy().astype("float32")        # (T,165) ground-truth keypoints
gs = sd["global_state"].numpy().astype("float32")            # (T,3) root + shoulder width
T = orig.shape[0]
print(f"chunk: {Path(args.chunk).name}  frames={T}")

# encode -> decode using only the first --ncb codebooks (production = 4)
codes = tok.encode(torch.from_numpy(orig).float())
n_use = min(args.ncb, len(codes))
tok.model.set_active_codebooks(n_use)
recon = tok.decode([c for c in codes[:n_use]]).cpu().numpy().astype("float32")  # recon[t] ~ pose[t-shift]
print(f"using {n_use}/{ncb} codebooks  n token-frames={codes[0].shape[-1]}  recon frames={recon.shape[0]}")

# de-shift the reconstruction (advance by `shift`) to realign with the timeline
if args.shift > 0 and recon.shape[0] > args.shift:
    recon = np.concatenate([recon[args.shift:], recon[-1:].repeat(args.shift, 0)], axis=0)

L = min(T, recon.shape[0])
o3 = orig[:L].reshape(L, NUM_JOINTS, 3)
r3 = recon[:L].reshape(L, NUM_JOINTS, 3)
orig_kp = o3[..., :2].reshape(L, NUM_JOINTS * 2)
recon_kp = r3[..., :2].reshape(L, NUM_JOINTS * 2)
orig_conf = o3[..., 2]
recon_conf = r3[..., 2]

# reconstruction error (offset space)
off_mse = float(((o3[..., :2] - r3[..., :2]) ** 2).mean())
off_mae = float(np.abs(o3[..., :2] - r3[..., :2]).mean())
print(f"reconstruction  offset-MSE={off_mse:.4e}  offset-MAE={off_mae:.4f}")

render_comparison_video(
    orig_kp=orig_kp, recon_kp=recon_kp, global_state=gs[:L],
    confidence=orig_conf, out_path=Path(args.out),
    fps=25.0, max_seconds=10.0, conf_thresh=0.0, recon_confidence=recon_conf,
)
print(f"wrote {args.out}  (left=original/green, right=reconstructed/red)")
