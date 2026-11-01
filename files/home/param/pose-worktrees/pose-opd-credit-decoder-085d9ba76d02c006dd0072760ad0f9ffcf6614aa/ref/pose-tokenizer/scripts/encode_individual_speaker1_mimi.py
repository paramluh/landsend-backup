#!/usr/bin/env python3
"""Encode all wavs under individual_speaker1 with Mimi (4 codebooks).

Same conventions as build_mimi_unchunked_uncut.py: kyutai/mimi, mono 24 kHz
(librosa), per-file encode (batch=1), tokens (4, T) int16 in safetensors.
Long segments are encoded in frame-aligned windows (1920-sample boundaries)
to bound memory; frame-aligned windowing is code-identical to one-shot.

Input : /mnt/som-fs/hands/individual_speaker1/**/*.wav
Output: /mnt/som-fs/hands/individual_speaker1-encoded/<relative_path>.safetensors
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import librosa
import torch
from safetensors.torch import save_file
from transformers import AutoFeatureExtractor, MimiModel

MODEL = "kyutai/mimi"
SR = 24000
N_CB = 4
FRAME_SAMPLES = 1920          # one 12.5 Hz Mimi frame @ 24 kHz
WINDOW_SEC = 300              # encode in 5-min frame-aligned windows
IN_ROOT = Path("/mnt/som-fs/hands/individual_speaker1")
OUT_ROOT = Path("/mnt/som-fs/hands/individual_speaker1-encoded")

torch.backends.cudnn.enabled = False


def encode_wav(mimi, mfe, dev, audio) -> torch.Tensor:
    win = (WINDOW_SEC * SR // FRAME_SAMPLES) * FRAME_SAMPLES
    parts = []
    for s in range(0, len(audio), win):
        chunk = audio[s:s + win]
        inp = mfe(raw_audio=chunk, sampling_rate=SR, return_tensors="pt")
        with torch.inference_mode():
            codes = mimi.encode(inp["input_values"].to(dev)).audio_codes  # (1,K,T)
        parts.append(codes[0, :N_CB].to(torch.int16).cpu())
    return torch.cat(parts, dim=1).contiguous()  # (4, T)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dev = args.device if torch.cuda.is_available() else "cpu"
    mfe = AutoFeatureExtractor.from_pretrained(MODEL)
    mimi = MimiModel.from_pretrained(MODEL).to(dev).eval()
    assert int(mfe.sampling_rate) == SR

    wavs = sorted(IN_ROOT.rglob("*.wav"))
    if args.limit:
        wavs = wavs[: args.limit]
    work = []
    for w in wavs:
        out = OUT_ROOT / w.relative_to(IN_ROOT).with_suffix(".safetensors")
        if args.overwrite or not out.exists():
            work.append((w, out))
    print(f"to encode: {len(work)}/{len(wavs)}", flush=True)

    t0 = time.time()
    for i, (w, out) in enumerate(work, 1):
        audio, _ = librosa.load(str(w), sr=SR, mono=True)
        tk = encode_wav(mimi, mfe, dev, audio)
        out.parent.mkdir(parents=True, exist_ok=True)
        save_file({"tokens": tk}, str(out))
        el = time.time() - t0
        print(f"  {i}/{len(work)} | {el:.0f}s | {len(audio)/SR:.0f}s audio -> {tuple(tk.shape)} | {w.relative_to(IN_ROOT)}", flush=True)
    print(f"DONE: {len(work)} files in {time.time()-t0:.0f}s -> {OUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
