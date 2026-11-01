#!/usr/bin/env python3
"""One-time preparation of the Rosie Stamp fine-tune data (InternalCan/rosie-stamp-conversations-v1).

Stages (run all by default, or pick with --stages):
  download  hf_hub_download the 8 audio tars into <root>/audio (resumable; already-complete files are kept).
            The HF token is read from the HF_TOKEN environment variable only (never written anywhere).
  extract   extract only the *.original.flac lanes into <root>/flac/<session_id>/<seg>/{Rosie_Stamp,user}.original.flac
            (size-checked against index/audio_members.parquet; existing same-size files kept).
  index     build <root>/prepared/index.json (see data_rosie.py for the schema): per conversation the split, lanes, T,
            Rosie's tokenised word lane (Qwen3-32B pieces), chunk list, and voice-clip candidate windows; plus
            <root>/prepared/codes/<conv>.{rosie,user}.npy (original Mimi codes, int16 [32,T]) for fast mmap loading.
  wav       <root>/prepared/wav/<conv>.user.f32.npy = the partner FLAC decoded once exactly as audio_s4q.ffmpeg_decode_24k
            (ffmpeg -f f32le -ac 1 -ar 24000, raw level); RosieData slices it (bit-identical to audio_s4q.ffmpeg_row_audio at 0 dB).

  /workspace/aloe-live/venv/bin/python prep_rosie.py [--root /workspace/data/rosie] [--stages download,extract,index,wav]
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "code"))

REPO = "InternalCan/rosie-stamp-conversations-v1"
N_TARS = 8


def stage_download(root: Path):
    from huggingface_hub import hf_hub_download
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        sys.exit("HF_TOKEN env var required for download")
    import pyarrow.parquet as pq
    mem = pq.read_table(root / "index" / "audio_members.parquet").to_pandas()
    tars = sorted(mem.tar.unique())
    assert len(tars) == N_TARS, tars
    for t in tars:
        dst = root / t
        # expected size >= last member offset+size
        sub = mem[mem.tar == t]
        min_size = int((sub.offset + sub["size"]).max())
        if dst.exists() and dst.stat().st_size >= min_size:
            print(f"[download] {t} present ({dst.stat().st_size/1e9:.2f} GB)", flush=True)
            continue
        t0 = time.time()
        for attempt in range(5):
            try:
                p = hf_hub_download(REPO, t, repo_type="dataset", local_dir=str(root), token=tok)
                break
            except Exception as e:  # network hiccup -> retry (hf_hub resumes .incomplete files)
                print(f"[download] {t} attempt {attempt} failed: {e!r}", flush=True)
                time.sleep(10)
        else:
            sys.exit(f"download of {t} failed")
        sz = Path(p).stat().st_size
        assert sz >= min_size, (t, sz, min_size)
        print(f"[download] {t} {sz/1e9:.2f} GB in {time.time()-t0:.0f}s", flush=True)


def stage_extract(root: Path):
    import pyarrow.parquet as pq
    mem = pq.read_table(root / "index" / "audio_members.parquet").to_pandas()
    mem = mem[mem.member.str.endswith(".original.flac")]
    out = root / "flac"
    n_new = n_have = 0
    for t, sub in mem.groupby("tar"):
        with open(root / t, "rb") as f:
            for r in sub.itertuples():
                # member = Rosie_Stamp/<sid>/<seg>/<name>.original.flac ; read via the recorded data offset (no tar parsing needed)
                parts = r.member.split("/")
                assert parts[0] == "Rosie_Stamp" and len(parts) == 4, r.member
                dst = out / parts[1] / parts[2] / parts[3]
                if dst.exists() and dst.stat().st_size == int(r.size):
                    n_have += 1
                    continue
                f.seek(int(r.offset))
                data = f.read(int(r.size))
                assert len(data) == int(r.size) and data[:4] == b"fLaC", (r.member, len(data))
                dst.parent.mkdir(parents=True, exist_ok=True)
                tmp = dst.with_name(dst.name + ".part")
                tmp.write_bytes(data)
                os.replace(tmp, dst)
                n_new += 1
    print(f"[extract] original FLACs: {n_new} new, {n_have} present -> {out}", flush=True)
    assert n_new + n_have == len(mem)


def stage_index(root: Path, args):
    import data_rosie
    data_rosie.build_index(str(root), n_workers=args.workers)


def stage_wav(root: Path, args):
    import data_rosie
    data_rosie.build_wav(str(root), n_workers=args.workers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/workspace/data/rosie")
    ap.add_argument("--stages", default="download,extract,index,wav")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    root = Path(args.root)
    for s in args.stages.split(","):
        {"download": lambda: stage_download(root), "extract": lambda: stage_extract(root),
         "index": lambda: stage_index(root, args), "wav": lambda: stage_wav(root, args)}[s]()


if __name__ == "__main__":
    main()
