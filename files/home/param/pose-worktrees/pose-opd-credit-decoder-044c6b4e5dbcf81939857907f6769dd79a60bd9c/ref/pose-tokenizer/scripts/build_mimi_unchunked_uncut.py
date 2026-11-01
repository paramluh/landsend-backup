#!/usr/bin/env python3
"""Encode full speaking-clip audio with Mimi (4 codebooks), unchunked & uncut.

Each clip's whole wav is encoded ON ITS OWN at its natural Mimi length (12.5 Hz).
No chunking, no truncation, no padding/trimming to the pose length -- the audio
is encoded exactly as-is. (Lengths will be ~equal to the uncut pose tokens, off
by 0/-1 from rounding; reconciling any mismatch is left to the caller.)

NOTE: Mimi's encoder is NOT padding-invariant, so batching variable-length clips
with right-padding changes the tokens of the shorter clips. We therefore encode
each clip individually (model forward batch = 1) and only parallelise the wav
loading (I/O) with a thread pool.

Input : aloe/v1/<seg>/speaking-clips/<stem>.wav
        (membership = stems present in pose-4cb-unchunked-uncut)
Output: aloe-encoded/mimi-audio-unchunked-uncut/<stem>.safetensors
            tokens (4, T) int16
"""
from __future__ import annotations

import argparse
import glob
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import librosa
import torch
from safetensors.torch import save_file
from transformers import AutoFeatureExtractor, MimiModel

MODEL = "kyutai/mimi"
SR = 24000
N_CB = 4
V1 = "/mnt/som-fs/hands/aloe/v1"
POSE = Path("/mnt/som-fs/hands/aloe-encoded/pose-4cb-unchunked-uncut")
OUT = Path("/mnt/som-fs/hands/aloe-encoded/mimi-audio-unchunked-uncut")

torch.backends.cudnn.enabled = False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--io-workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    dev = args.device if torch.cuda.is_available() else "cpu"

    mfe = AutoFeatureExtractor.from_pretrained(MODEL)
    mimi = MimiModel.from_pretrained(MODEL).to(dev).eval()
    assert int(mfe.sampling_rate) == SR

    wavs = {os.path.basename(w)[:-4]: w for w in glob.glob(f"{V1}/*/speaking-clips/*.wav")}
    stems = [t.name[:-len(".safetensors")] for t in sorted(POSE.glob("*.safetensors"))]
    if args.limit:
        stems = stems[: args.limit]
    if not args.overwrite:
        stems = [s for s in stems if not (out / f"{s}.safetensors").exists()]
    work = [(s, wavs[s]) for s in stems if s in wavs]
    miss = len(stems) - len(work)
    print(f"to encode: {len(work)} (missing wav: {miss}) | per-clip encode", flush=True)

    def load(item):
        stem, w = item
        return stem, librosa.load(w, sr=SR, mono=True)[0]

    WIN = max(args.io_workers * 4, 32)   # bounded prefetch window
    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.io_workers) as ex:
        for ws in range(0, len(work), WIN):
            window = work[ws:ws + WIN]
            for stem, audio in ex.map(load, window):
                inp = mfe(raw_audio=audio, sampling_rate=SR, return_tensors="pt")
                with torch.inference_mode():
                    codes = mimi.encode(inp["input_values"].to(dev)).audio_codes  # (1,K,T)
                tk = codes[0, :N_CB].to(torch.int16).cpu().contiguous()           # (4,T)
                save_file({"tokens": tk}, str(out / f"{stem}.safetensors"))
                done += 1
                if done % 500 == 0:
                    el = time.time() - t0
                    print(f"  {done}/{len(work)} | {el:.0f}s | {done/max(el,1e-9):.1f} clips/s", flush=True)
    print(f"DONE: {done} clips in {time.time()-t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()
