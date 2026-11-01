"""Own-speech hours per speaker account over ALL handover none-crossbleed conversations that have sam-sep codes (every chunk, full call):
own speech = union of the word spans whose _separated_speaker is the account's lane (words.json), lane -> speaker_NN_account_uid (manifest).
Also: conversations, as A (user lane S01) / as B (assistant lane S00), distinct partners, the account's conversations in crossbleed buckets
(flag), and its convs purged upstream (none bucket but no sam-sep pair). -> shortlist/hours_all.json"""
import collections, csv, json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pyarrow.parquet as pq
DATA = "/data/home/amartyavarma/data/aloe-handover-full/DATA"
OUT = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/actors/shortlist/hours_all.json"
csv.field_size_limit(1 << 30)
man = list(csv.DictReader(open(f"{DATA}/manifest.csv")))
pairs = pq.read_table("/data/home/amartyavarma/data/aloe-sam-sep/hf/index/pairs.parquet").to_pylist()
print("pairs cols", list(pairs[0].keys())[:8], flush=True)
sk = {(p.get("session") or p.get("session_id"), p.get("seg")) for p in pairs}
def one(m):
    try:
        W = json.load(open(f"{DATA}/{m['relative_dir']}/w00.words.json"))["words"]
    except Exception:
        return m, None
    out = {}
    for L in ("00", "01"):
        iv = sorted((w["start"], w["end"]) for w in W if w.get("_separated_speaker") == f"SPEAKER_{L}")
        tot, ce = 0.0, -1.0
        for s, e in iv:
            if e <= ce:
                continue
            tot += e - max(s, ce); ce = e
        out[L] = tot
    return m, out
with ThreadPoolExecutor(48) as ex:
    res = list(ex.map(one, man))
A = collections.defaultdict(lambda: {"own_s": 0.0, "convs": 0, "as_A": 0, "as_B": 0, "partners": set(), "call_s": 0.0, "xbleed_convs": 0, "purged_convs": 0, "keys": []})
miss = 0
for m, o in res:
    bucket = m["relative_dir"].split("/")[1]
    ok = (m["session_id"], m["seg"]) in sk
    for L in ("00", "01"):
        acc = m[f"speaker_{L}_account_uid"]; a = A[acc]
        if bucket != "none":
            a["xbleed_convs"] += 1
            continue
        if not ok:
            a["purged_convs"] += 1
            continue
        if o is None:
            miss += 1
            continue
        a["own_s"] += o[L]; a["convs"] += 1; a["call_s"] += float(m["clip_duration_sec"])
        a["as_B" if L == "00" else "as_A"] += 1
        a["partners"].add(m[f"speaker_{'01' if L == '00' else '00'}_account_uid"]); a["keys"].append(m["recording_key"])
rows = sorted(({"account": k, **{x: (len(v) if x == "partners" else v) for x, v in a.items()}} for k, a in A.items()), key=lambda r: -r["own_s"])
json.dump({"rows": rows, "missing_words": miss, "n_pairs": len(sk)}, open(OUT, "w"))
tot = sum(r["own_s"] for r in rows) / 3600
print("accounts", len(rows), "own h", round(tot, 1), "missing", miss)
for r in rows[:15]:
    print(r["account"][:8], round(r["own_s"] / 3600, 2), r["convs"], r["as_A"], r["as_B"], r["partners"], r["xbleed_convs"], r["purged_convs"])
