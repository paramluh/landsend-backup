#!/usr/bin/env python3
"""Encode ALL aloe v1-poses chunks to 4-codebook pose tokens.

Input : /mnt/som-fs/hands/aloe/v1-poses/<clip>/<clip>_poses_chunkNNNN.safetensors
        each has reconstruction (T,165) + global_state (T,3)
Output: /mnt/som-fs/hands/aloe/v1-poses-4cb/<clip>/<same filename>
        each has tokens (4, n_token_frames) int16 + global_state (T,3) float32

Batch size 256, first 4 codebooks. Resumable (skips existing outputs).
"""
import argparse, os, glob, time, sys
from pathlib import Path
import numpy as np, torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, "/home/james/pose-tokenizer")
from pose_tokenizer import PoseTokenizer

ROOT = Path("/mnt/som-fs/hands/aloe/v1-poses")
# aloe/ is not writable by this user; closest writable location, names mirror originals.
OUT_ROOT = Path("/mnt/som-fs/hands/aloe-v1-poses-4cb")
NCB = 4

ap = argparse.ArgumentParser()
ap.add_argument("--batch-size", type=int, default=256)
ap.add_argument("--repo", default="xabirizar9/pose-tokenizer-2x-12cb-lossshift1")
ap.add_argument("--limit", type=int, default=0, help="process only first N chunks (0=all)")
ap.add_argument("--overwrite", action="store_true")
args = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = PoseTokenizer.from_pretrained(args.repo, device=dev)
print(f"loaded {args.repo}  codebooks={len(tok.model.quantizer.quantizers)}  using first {NCB}", flush=True)

# 1) enumerate all (out_dir, chunk_path)
t0 = time.time()
jobs = []
with os.scandir(ROOT) as it:
    for e in it:
        if not e.is_dir():
            continue
        for f in glob.glob(os.path.join(e.path, "*_poses_chunk*.safetensors")):
            jobs.append((e.name, f))
        if args.limit and len(jobs) >= args.limit:
            break
if args.limit:
    jobs = jobs[:args.limit]
print(f"enumerated {len(jobs):,} chunks in {time.time()-t0:.1f}s", flush=True)


def out_path(clip_name, chunk_path):
    return OUT_ROOT / clip_name / os.path.basename(chunk_path)


# 2) batch encode
t0 = time.time()
done = skipped = 0
B = args.batch_size
for s in range(0, len(jobs), B):
    batch_jobs = jobs[s:s + B]
    # filter already-done (resumable)
    todo = []
    for clip, cp in batch_jobs:
        op = out_path(clip, cp)
        if op.exists() and not args.overwrite:
            skipped += 1
        else:
            todo.append((clip, cp, op))
    if not todo:
        continue
    recons, gss, lens = [], [], []
    for clip, cp, op in todo:
        d = load_file(cp)
        recons.append(d["reconstruction"].numpy().astype("float32"))
        gss.append(d["global_state"].numpy().astype("float32"))
        lens.append(recons[-1].shape[0])
    Tmax = max(lens)
    arr = np.stack([np.pad(r, ((0, Tmax - r.shape[0]), (0, 0))) for r in recons])
    with torch.no_grad():
        codes = tok.encode(torch.from_numpy(arr).float())          # list of 12, each (b, Tc)
    codes4 = torch.stack([codes[k] for k in range(NCB)], dim=1)     # (b, 4, Tc)  int
    codes4 = codes4.to(torch.int16).cpu().numpy()
    for j, (clip, cp, op) in enumerate(todo):
        n_tok = (lens[j] + 1) // 2                                  # real token-frames for this chunk
        toks = codes4[j, :, :n_tok]                                 # (4, n_tok)
        op.parent.mkdir(parents=True, exist_ok=True)
        save_file({"tokens": torch.from_numpy(toks),
                   "global_state": torch.from_numpy(gss[j])}, str(op))
        done += 1
    if (s // B) % 10 == 0:
        el = time.time() - t0
        rate = done / el if el > 0 else 0
        eta = (len(jobs) - skipped - done) / rate if rate > 0 else 0
        print(f"  [{s+len(batch_jobs):,}/{len(jobs):,}] done={done:,} skipped={skipped:,} "
              f"rate={rate:.0f}/s eta={eta/60:.1f}min", flush=True)

print(f"\nDONE: encoded {done:,} chunks ({skipped:,} skipped) in {(time.time()-t0)/60:.1f} min -> {OUT_ROOT}", flush=True)
