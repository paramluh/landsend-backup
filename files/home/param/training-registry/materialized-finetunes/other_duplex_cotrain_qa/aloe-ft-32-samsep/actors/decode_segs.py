"""Stage A of the per-actor voice clips: candidate segments of each top actor's OWN speech, decoded to wav.
Segment = a run of the actor's own word spans on their lane (store word timings; gaps < 0.48 s closed, 2 frames of padding), 2.4-10 s,
cut back to a word gap when longer, kept only if the PARTNER lane has no word within 1 s of it. Up to 2 per conversation, up to 30
conversations per actor (fixed hashed order), first-chunk TRAIN rows. Decoded from the row's SAM-isolated Mimi codes (the sam-sep store
codes the FT trains on; codes_assistant = SPEAKER_00 isolated, codes_user = SPEAKER_01 isolated) with transformers MimiModel
/data/home/amartyavarma/models/mimi (fp32) with 1 s of left context trimmed off.
Actor id = manifest speaker_<lane>_actor_id for the row's lane (assistant_speaker_dir / user_speaker_dir).
usage: python -I decode_segs.py SHARD NSHARD [TOP] -> actors/segs/r<rank>/<i>.wav + actors/segs/segs_<SHARD>.json"""
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
IDF = os.environ.get("IDF", "actor_id")
TAG = os.environ.get("ACT_TAG", "")
SR, HOP, CTX, PAD, GUARD = 24000, 1920, 13, 2, 12
MIN_F, MAX_F, CLOSE, PER_CONV, MAX_CONV = 30, 125, 6, 2, 30


def mask(words, n):
    z = np.zeros(n, bool)
    for w in words:
        z[int(w["start_frame"]):max(int(w["end_frame"]), int(w["start_frame"]) + 1)] = True
    return z


def segments(own, oth, n):
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
        if e0 - s0 > MAX_F:                      # cut back to the last own-word gap inside the cap
            e1 = s0 + MAX_F
            while e1 > s0 + MIN_F and own[e1 - 1]:
                e1 -= 1
            if e1 <= s0 + MIN_F:
                continue
            e0 = e1
        if e0 - s0 < MIN_F:
            continue
        if oth[max(0, s0 - GUARD):min(n, e0 + GUARD)].any():
            continue
        out.append((int(s0), int(e0)))
    return sorted(out, key=lambda x: x[0] - x[1])


def main():
    shard, nshard = int(sys.argv[1]), int(sys.argv[2])
    top = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    st = json.load(open(f"{D}/actor_stats{TAG}.json"))
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
            assert r["assistant_speaker_dir"] == "SPEAKER_00" and r["user_speaker_dir"] == "SPEAKER_01"
            for s in F.SIDES:
                lane = r[f"{s}_speaker_dir"]
                aid = m[f"speaker_{lane[-2:]}_{IDF}"]
                if aid in todo:
                    partner = m[f"speaker_{'01' if lane == 'SPEAKER_00' else '00'}_{IDF}"]
                    todo[aid].append((hashlib.md5(r["row_key"].encode()).hexdigest(), k, int(r["row"]), s, lane, r["row_key"], partner))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    from transformers import MimiModel
    mimi = MimiModel.from_pretrained("/data/home/amartyavarma/models/mimi").to(dev).eval()
    stores = {k: F.Rows(F.load_node(f"{STORES}/m1n{k}")) for k in range(4)}
    res = []
    for aid, rank in sorted(ranks.items(), key=lambda x: x[1]):
        os.makedirs(f"{D}/segs{TAG}/r{rank:02d}", exist_ok=True)
        nconv, j = 0, 0
        for _, k, i, s, lane, key, partner in sorted(todo[aid]):
            rw = stores[k][i]
            assert rw["row_key"] == key
            n = int(rw["n_frames"])
            o = "user" if s == "assistant" else "assistant"
            segs = segments(mask(rw[f"words_{s}"], n), mask(rw[f"words_{o}"], n), n)[:PER_CONV]
            if not segs:
                continue
            c = np.asarray(rw[f"codes_{s}"]).astype(np.int64)
            for s0, e0 in segs:
                lo = max(0, s0 - CTX)
                with torch.no_grad():
                    w = mimi.decode(torch.as_tensor(c[:, lo:e0], device=dev)[None]).audio_values[0, 0].float().cpu().numpy()
                w = np.clip(w[(s0 - lo) * HOP:(e0 - lo) * HOP], -1, 1)
                fn = f"segs{TAG}/r{rank:02d}/{j:03d}.wav"
                sf.write(f"{D}/{fn}", w, SR, subtype="PCM_16")
                res.append({"rank": rank, "actor_id": aid, "file": fn, "row_key": key, "side": s, "lane": lane, "partner_id": partner,
                            "start_s": round(s0 / 12.5, 2), "dur_s": round((e0 - s0) / 12.5, 2)})
                j += 1
            nconv += 1
            if nconv >= MAX_CONV:
                break
        json.dump(res, open(f"{D}/segs{TAG}/segs_{shard}.json", "w"), indent=0)
        print(f"rank {rank} {j} segs {nconv} convs", flush=True)


if __name__ == "__main__":
    main()
