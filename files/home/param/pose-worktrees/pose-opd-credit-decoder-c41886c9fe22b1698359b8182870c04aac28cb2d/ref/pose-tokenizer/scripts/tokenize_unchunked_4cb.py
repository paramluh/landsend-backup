#!/usr/bin/env python3
"""Tokenize the unchunked-uncut pose reconstructions with the 2x-12cb pose
tokenizer, keeping the first 4 codebooks (4cb), one safetensors per clip.

Input : aloe-encoded/pose-unchunked-uncut/{stem}.safetensors
            reconstruction (T,165) float32 / global_state (T,3) float32
Output: aloe-encoded/pose-4cb-unchunked-uncut/{stem}.safetensors
            tokens (4, ceil(T/2)) int16 / global_state (T,3) float32

Matches the existing aloe-encoded/pose-4cb-unchunked encoding exactly
(verified bit-identical): encode(reconstruction) -> 12 codebooks, take the
first 4. Batched right-padded encoding is bit-exact to per-clip encoding when
cudnn is disabled (the model is causal), so we batch for speed and slice each
clip's tokens back to ceil(T/2).
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from pose_tokenizer import PoseTokenizer

REPO = "xabirizar9/pose-tokenizer-2x-12cb-lossshift1"
SRC = Path("/mnt/som-fs/hands/aloe-encoded/pose-unchunked-uncut")
OUT = Path("/mnt/som-fs/hands/aloe-encoded/pose-4cb-unchunked-uncut")
N_CB = 4
FEAT = 165

torch.backends.cudnn.enabled = False  # deterministic + batch-safe (causal model)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    dev = args.device if torch.cuda.is_available() else "cpu"
    tok = PoseTokenizer.from_pretrained(REPO, device=dev)
    ds = tok.model.downsampling_factor
    print(f"model: n_codebooks={tok.config.n_codebooks} ds={ds} "
          f"input_features={tok.config.input_features} -> keep {N_CB}cb", flush=True)

    files = sorted(SRC.glob("*.safetensors"))
    if args.limit:
        files = files[: args.limit]
    if not args.overwrite:
        files = [f for f in files if not (out / f.name).exists()]
    print(f"to tokenize: {len(files)} clips, batch_size={args.batch_size}", flush=True)

    # read lengths, sort so each batch pads uniformly (big speedup)
    t0 = time.time()
    lens = {}
    for f in files:
        with safe_open(str(f), framework="pt") as h:
            lens[f] = h.get_slice("reconstruction").get_shape()[0]
    files.sort(key=lambda f: lens[f])
    print(f"indexed lengths in {time.time()-t0:.0f}s", flush=True)

    done = 0
    t0 = time.time()
    for s in range(0, len(files), args.batch_size):
        chunk = files[s:s + args.batch_size]
        recons, gss, ntoks, names = [], [], [], []
        for f in chunk:
            with safe_open(str(f), framework="pt") as h:
                recons.append(h.get_tensor("reconstruction").float())
                gss.append(h.get_tensor("global_state").float())
            ntoks.append(math.ceil(recons[-1].shape[0] / ds))
            names.append(f.name)
        Tmax = max(r.shape[0] for r in recons)
        B = len(recons)
        batch = torch.zeros(B, Tmax, FEAT)
        for i, r in enumerate(recons):
            batch[i, : r.shape[0]] = r
        with torch.inference_mode():
            codes = tok.encode(batch.to(dev))            # list[12] of (B, Tmax_tok)
        toks = torch.stack(codes[:N_CB], 0).to(torch.int16).cpu()  # (4, B, Tmax_tok)
        for i in range(B):
            t = toks[:, i, : ntoks[i]].contiguous()      # (4, ceil(T/2))
            save_file({"tokens": t, "global_state": gss[i].contiguous()},
                      str(out / names[i]))
        done += B
        el = time.time() - t0
        print(f"  {done}/{len(files)} | {el:.0f}s | {done/max(el,1e-9):.0f} clips/s "
              f"| Tmax={Tmax}", flush=True)
    print(f"DONE: {done} clips in {time.time()-t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
