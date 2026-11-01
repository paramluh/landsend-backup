"""~60 s of each top actor's OWN voice: their own-lane speech segments (runs of their store word spans, gaps < 0.8 s closed, partner silent
in and around the segment, 3-10 s each, at most 2 per conversation, conversations in a fixed hashed order) from the first-chunk TRAIN rows,
decoded from the row's SAM-isolated Mimi codes (the sam-sep store codes the FT trains on) with transformers MimiModel
/data/home/amartyavarma/models/mimi (fp32) with 1 s of left context trimmed off, then joined with 0.4 s of silence.
usage: python -I decode_long.py SHARD NSHARD [TOP] -> actors/audio_long/r<rank>.mp3 + actors/audio_long/long_<SHARD>.json"""
import csv
import hashlib
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, "/data/home/amartyavarma/qwen-1ch/stage3/data/rows_build/scripts")
import s3rows as F  # noqa: E402

STORES = "/data/home/amartyavarma/qwen-1ch-data/s2h-m1-samsep/stores"
MAN = "/data/home/amartyavarma/data/aloe-handover-full/DATA/manifest.csv"
D = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/actors"
SR, HOP, CTX, PAD, GAP = 24000, 1920, 13, 3, 0.4
TARGET_S, MIN_F, MAX_F, CLOSE = 60.0, 38, 125, 10


def mask(words, n):
    z = np.zeros(n, bool)
    for w in words:
        z[int(w["start_frame"]):max(int(w["end_frame"]), int(w["start_frame"]) + 1)] = True
    return z


def segments(own, oth, n):
    """runs of own speech (gaps < CLOSE frames closed) with the partner silent from PAD before to PAD after; longest first."""
    o = own.copy()
    idx = np.flatnonzero(own)
    if len(idx) == 0:
        return []
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= CLOSE:
            o[a:b] = True
    d = np.diff(np.concatenate([[0], o.astype(int), [0]]))
    out = []
    for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        s0, e0 = max(0, s - PAD), min(n, e + PAD)
        if e0 - s0 < MIN_F:
            continue
        e0 = min(e0, s0 + MAX_F)
        if oth[max(0, s0 - PAD):min(n, e0 + PAD)].any():
            continue
        out.append((int(s0), int(e0)))
    return sorted(out, key=lambda x: x[0] - x[1])


def main():
    shard, nshard = int(sys.argv[1]), int(sys.argv[2])
    top = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    os.makedirs(f"{D}/audio_long", exist_ok=True)
    st = json.load(open(f"{D}/actor_stats.json"))
    ranks = {a["actor_id"]: i for i, a in enumerate(st["actors"][:top], 1) if (i - 1) % nshard == shard}
    csv.field_size_limit(1 << 30)
    man = {m["recording_key"]: m for m in csv.DictReader(open(MAN))}
    todo = {aid: [] for aid in ranks}
    for k in range(4):
        for ln in open(f"{STORES}/m1n{k}/rows.jsonl"):
            r = json.loads(ln)
            if int(r["chunk_index"]) != 0 or r["split"] != "train":
                continue
            m = man[r["src_conversation"]]
            for s in F.SIDES:
                aid = m[f"speaker_{r[f'{s}_speaker_dir'][-2:]}_actor_id"]
                if aid in todo:
                    todo[aid].append((hashlib.md5(r["row_key"].encode()).hexdigest(), k, int(r["row"]), s, r["row_key"]))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    from transformers import MimiModel
    mimi = MimiModel.from_pretrained("/data/home/amartyavarma/models/mimi").to(dev).eval()
    stores = {k: F.Rows(F.load_node(f"{STORES}/m1n{k}")) for k in range(4)}
    res = []
    for aid, rank in sorted(ranks.items(), key=lambda x: x[1]):
        pieces, meta, got = [], [], 0.0
        for _, k, i, s, key in sorted(todo[aid]):
            rw = stores[k][i]
            assert rw["row_key"] == key
            n = int(rw["n_frames"])
            o = "user" if s == "assistant" else "assistant"
            segs = segments(mask(rw[f"words_{s}"], n), mask(rw[f"words_{o}"], n), n)[:2]
            if not segs:
                continue
            c = np.asarray(rw[f"codes_{s}"]).astype(np.int64)
            for s0, e0 in segs:
                lo = max(0, s0 - CTX)
                with torch.no_grad():
                    w = mimi.decode(torch.as_tensor(c[:, lo:e0], device=dev)[None]).audio_values[0, 0].float().cpu().numpy()
                pieces.append(np.clip(w[(s0 - lo) * HOP:(e0 - lo) * HOP], -1, 1))
                pieces.append(np.zeros(int(GAP * SR), np.float32))
                meta.append({"row_key": key, "side": s, "start_s": round(s0 / 12.5, 2), "dur_s": round((e0 - s0) / 12.5, 2)})
                got += (e0 - s0) / 12.5
            if got >= TARGET_S:
                break
        wav = np.concatenate(pieces[:-1]) if pieces else np.zeros(SR, np.float32)
        fn = f"r{rank:02d}.mp3"
        sf.write(f"{D}/audio_long/{fn}", wav, SR, format="MP3")
        convs = len({m["row_key"] for m in meta})
        res.append({"rank": rank, "actor_id": aid, "file": fn, "speech_s": round(got, 1), "total_s": round(len(wav) / SR, 1),
                    "n_segments": len(meta), "n_convs": convs, "lanes": sorted({m["side"] for m in meta}), "segments": meta})
        json.dump(res, open(f"{D}/audio_long/long_{shard}.json", "w"), indent=0)
        print(f"rank {rank} {got:.1f}s speech {len(meta)} segs {convs} convs", flush=True)


if __name__ == "__main__":
    main()
