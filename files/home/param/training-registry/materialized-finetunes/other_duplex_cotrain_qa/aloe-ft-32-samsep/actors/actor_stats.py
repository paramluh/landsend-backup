"""Per-actor data in the aloe-ft-32-samsep fine-tuning set (first-chunk rows of the sam-sep s2h-m1 stores, split train / eval).
Actor id = handover manifest speaker_00_actor_id / speaker_01_actor_id (aloe-handover-full/DATA/manifest.csv), joined on the row's
src_conversation (= manifest recording_key). Lane SPEAKER_00 = assistant, SPEAKER_01 = user (row assistant_speaker_dir / user_speaker_dir).
Per actor: lane hours (row frames / 12.5 on their lane), speech hours (union of their lane's word spans in the row, store word timings),
rows, conversations, assistant / user lane split, persona names seen ("Your name is X"), and sample candidates (10 s windows of their
lane with the most own speech and the least partner speech). -> actors/actor_stats.json"""
import collections
import csv
import json
import re
import sys

import os

import numpy as np

sys.path.insert(0, "/data/home/amartyavarma/qwen-1ch/stage3/data/rows_build/scripts")
import s3rows as F  # noqa: E402

STORES = "/data/home/amartyavarma/qwen-1ch-data/s2h-m1-samsep/stores"
MAN = "/data/home/amartyavarma/data/aloe-handover-full/DATA/manifest.csv"
IDF = os.environ.get("IDF", "actor_id")                       # actor_id | account_uid (the voice: see id_check.py)
OTHER = "account_uid" if IDF == "actor_id" else "actor_id"
OUT = f"/data/home/amartyavarma/ppm/aloe-ft-32-samsep/actors/actor_stats{os.environ.get('ACT_TAG', '')}.json"
FPS, WIN = 12.5, 125

csv.field_size_limit(1 << 30)
man = {}
for m in csv.DictReader(open(MAN)):
    man[m["recording_key"]] = m


def mask(words, n):
    z = np.zeros(n, bool)
    for w in words:
        z[int(w["start_frame"]):max(int(w["end_frame"]), int(w["start_frame"]) + 1)] = True
    return z


A = collections.defaultdict(lambda: {"train": collections.Counter(), "eval": collections.Counter(), "convs": set(), "names": collections.Counter(),
                                     "cands": []})
nrows = collections.Counter()
missing = 0
for k in range(4):
    rows = F.Rows(F.load_node(f"{STORES}/m1n{k}"))
    if k == 0:
        print("row keys:", sorted(rows[0].keys()), flush=True)
    meta = [json.loads(ln) for ln in open(f"{STORES}/m1n{k}/rows.jsonl")]
    assert len(meta) == len(rows)
    for i in range(len(rows)):
        r = meta[i]
        if int(r["chunk_index"]) != 0 or r["split"] not in ("train", "eval"):
            continue
        assert int(r["row"]) == i
        rw = rows[i]
        assert rw["row_key"] == r["row_key"]
        sp = r["split"]
        nrows[sp] += 1
        m = man.get(r["src_conversation"])
        if m is None:
            missing += 1
            continue
        n = int(r["n_frames"])
        masks = {s: mask(rw[f"words_{s}"], n) for s in F.SIDES}
        for s in F.SIDES:
            lane = r[f"{s}_speaker_dir"]
            aid = m[f"speaker_{lane[-2:]}_{IDF}"]
            o = "user" if s == "assistant" else "assistant"
            a = A[aid]
            c = a[sp]
            c["rows"] += 1
            c["lane_s"] += n / FPS
            c["speech_s"] += masks[s].sum() / FPS
            c[f"rows_{s}"] += 1
            c[f"lane_s_{s}"] += n / FPS
            if sp == "train":
                a["convs"].add(r["src_conversation"])
                a["names"][m[f"speaker_{lane[-2:]}_{OTHER}"]] += 1
            if sp == "train" and n >= WIN:
                own = np.convolve(masks[s].astype(float), np.ones(WIN), "valid")
                oth = np.convolve(masks[o].astype(float), np.ones(WIN), "valid")
                sc = own - 2 * oth
                j = int(np.argmax(sc))
                a["cands"].append({"store": k, "row": i, "row_key": r["row_key"], "side": s, "lane": lane, "f0": j, "own": int(own[j]),
                                   "oth": int(oth[j]), "score": float(sc[j])})
    print("store", k, dict(nrows), flush=True)

out = {"field": f"manifest speaker_00_{IDF} / speaker_01_{IDF} joined on src_conversation", "other_field": OTHER, "rows": dict(nrows), "missing_manifest": missing,
       "actors": []}
for aid, a in A.items():
    t, e = a["train"], a["eval"]
    cands = sorted(a["cands"], key=lambda x: -x["score"])
    pick, seen = [], set()
    for c in cands:
        conv = c["row_key"].split("/")[2]
        if conv in seen:
            continue
        seen.add(conv)
        pick.append(c)
        if len(pick) == 4:
            break
    out["actors"].append({"actor_id": aid, "train": dict(t), "eval": dict(e), "convs_train": len(a["convs"]), "names": a["names"].most_common(5),
                          "samples": pick})
out["actors"].sort(key=lambda x: -x["train"].get("speech_s", 0))
json.dump(out, open(OUT, "w"), indent=0)
tr = sum(x["train"].get("speech_s", 0) for x in out["actors"])
print("actors", len(out["actors"]), "train speech h", round(tr / 3600, 1), "missing", missing, flush=True)
