#!/usr/bin/env python3
"""Quantify the batched-vs-unbatched difference in KEYPOINT space and save the
10 most extreme deformed frames (overlay: unbatched=green, batched=red).
4 codebooks. Deformation = max per-joint L2 displacement (shoulder-width units)."""
import sys, glob, os
from pathlib import Path
import numpy as np, torch, cv2
from safetensors.torch import load_file
sys.path.insert(0, "/home/james/pose-tokenizer")
from pose_tokenizer import PoseTokenizer
from pose_tokenizer.data.kinematic import (
    NUM_JOINTS, PARENT_INDEX, TOPO_ORDER,
    FACE_CONNECTIONS, BODY_CONNECTIONS, LEFT_HAND_CONNECTIONS, RIGHT_HAND_CONNECTIONS)

ROOT = "/mnt/som-fs/hands/aloe/v1-poses"
NCB, NCHUNK = 4, 64
OUT = Path("/home/james/pose-tokenizer/extreme_deformations"); OUT.mkdir(exist_ok=True)
W = H = 512; PAD = 0.08
_CONN = FACE_CONNECTIONS + BODY_CONNECTIONS + LEFT_HAND_CONNECTIONS + RIGHT_HAND_CONNECTIONS
_J = sorted({i for p in _CONN for i in p})
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = PoseTokenizer.from_pretrained("xabirizar9/pose-tokenizer-2x-12cb-lossshift1", device=dev)
tok.model.set_active_codebooks(NCB)


def pos(off):
    out = np.empty_like(off)
    for j in TOPO_ORDER:
        p = PARENT_INDEX[j]
        out[..., j, :] = (0 if p == -1 else out[..., p, :]) + off[..., j, :]
    return out


paths = []
with os.scandir(ROOT) as it:
    for e in it:
        if e.is_dir():
            paths += sorted(glob.glob(e.path + "/*_poses_chunk*.safetensors"))
        if len(paths) >= NCHUNK:
            break
paths = paths[:NCHUNK]
arrs = [load_file(p)["reconstruction"].numpy().astype("float32") for p in paths]
Tmax = max(a.shape[0] for a in arrs)
batch = np.stack([np.pad(a, ((0, Tmax - a.shape[0]), (0, 0))) for a in arrs])

with torch.no_grad():
    cb = tok.encode(torch.from_numpy(batch).float())
    recon_b = tok.decode([c for c in cb[:NCB]]).cpu().numpy()
    recon_u = [tok.decode([c for c in tok.encode(torch.from_numpy(a).float())[:NCB]]).cpu().numpy()
               for a in arrs]

# per-frame max joint deformation (shoulder-width units)
records = []  # (deform, chunk_i, frame_t, pos_u, pos_b)
for i in range(len(arrs)):
    L = recon_u[i].shape[0]
    u3 = recon_u[i][:L].reshape(L, NUM_JOINTS, 3); b3 = recon_b[i, :L].reshape(L, NUM_JOINTS, 3)
    pu = pos(u3[:, :, :2]); pb = pos(b3[:, :, :2])
    d = np.sqrt(((pu - pb) ** 2).sum(-1))     # (L, J) per-joint displacement
    for t in range(L):
        records.append((float(d[t].max()), i, t, pu[t], pb[t]))

records.sort(key=lambda r: -r[0])
alld = np.array([r[0] for r in records])
print(f"keypoint deformation (max joint disp, shoulder-width units) over {len(records)} frames:")
print(f"  overall max={alld.max():.3f}  mean={alld.mean():.4f}  median={np.median(alld):.4f}  p99={np.percentile(alld,99):.3f}")


def render_overlay(rec, fname, label):
    deform, i, t, pu, pb = rec
    both = np.concatenate([pu, pb], 0)
    y0, y1, x0, x1 = both[:, 0].min(), both[:, 0].max(), both[:, 1].min(), both[:, 1].max()
    def to_px(P):
        yp = ((P[:, 0]-y0)/max(y1-y0, 1e-6)*(1-2*PAD)+PAD)*H
        xp = ((P[:, 1]-x0)/max(x1-x0, 1e-6)*(1-2*PAD)+PAD)*W
        return yp, xp
    canvas = np.zeros((H, W, 3), np.uint8)
    for P, col in [(pu, (0, 220, 0)), (pb, (60, 60, 255))]:   # green=unbatched, red=batched
        yp, xp = to_px(P)
        for a, b in _CONN:
            cv2.line(canvas, (int(xp[a]), int(yp[a])), (int(xp[b]), int(yp[b])), col, 2, cv2.LINE_AA)
        for j in _J:
            cv2.circle(canvas, (int(xp[j]), int(yp[j])), 3, col, -1, cv2.LINE_AA)
    cv2.putText(canvas, f"{label} chunk{i} frame{t} deform={deform:.3f} (green=unbatched red=batched)",
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(OUT / fname), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    return deform


print("\ntop 10 extreme frames:")
for rank, rec in enumerate(records[:10]):
    d = render_overlay(rec, f"extreme_{rank:02d}_chunk{rec[1]}_frame{rec[2]}.png", f"#{rank}")
    print(f"  #{rank}: chunk {rec[1]} frame {rec[2]}  max-joint-deform={d:.3f} sw")

# records sorted DESC; rank for percentile p = (1-p)*N from the top
N = len(records)
print("\npercentile frames:")
for pct in (95, 90):
    idx = min(N - 1, int((1 - pct/100.0) * N))
    rec = records[idx]
    render_overlay(rec, f"pctl_{pct}_chunk{rec[1]}_frame{rec[2]}.png", f"p{pct}")
    print(f"  p{pct}: deform={rec[0]:.4f} sw  (chunk {rec[1]} frame {rec[2]})  -> pctl_{pct}_*.png")
print(f"\nsaved overlays (10 extreme + p95 + p90) to {OUT}/")
