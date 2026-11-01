"""aloe-v2pure-200 samples ROWS (10-18): 10 held-out clips of job 23051's eval split (= the 8B v2pure's 113 held-out recordings) as
trainer-parity micros (data_aloe_pose.Stores.conv, the run's own code + env: the clip's own segment, B's system block, B lane / Mimi / pose,
channel A empty) + the clip's pose normalisation from the v2pure cache (frame0_root / frame0_width / orig_w / orig_h) for rendering.
SELECTION (no ASR; deterministic, independent of any model output):
  eval items with 188 <= n <= 375 frames (15-30 s); cache lufs_status 'ok' (not peak-limited) and audio_crossbleed_status 'not_detected'
  (no partner bleed); 2.4-3.6 words / s; no gap > 1.5 s between consecutive words; first word starts <= 0.5 s (the 2 s prefill has speech);
  NO voice clip in B's system block (P = 5: the prefill is only the clip's first 2 s of real speech + pose + words, as the 8B's prefill
  example); per recording the qualifying clip with the highest word coverage; recordings ordered by blake2b8('v2pure-samples|' + rid), first 10.
CHECKS: store row <-> cache item (row_key c<k> = the clip's ordinal by (start, speaker, segment_id)): n, B Mimi codes and B pose codes equal.
   python build_rows.py <out dir>      (source aloe-v2pure-ft/aloe_v2pure.env first; compute node, CPU)"""
import hashlib
import json
import os
import pickle
import sys

import numpy as np
import pyarrow.parquet as pq

OUT = sys.argv[1]
assert os.environ.get("QQA_A_EMPTY") == "1" and os.environ.get("QQA_POSE_B") == "1", "source aloe-v2pure-ft/aloe_v2pure.env first"
sys.path.insert(0, "/data/home/amartyavarma/ppm/aloe-v2pure-ft/code")
import data_aloe_pose as A  # noqa: E402

D = A.D
CACHE = "/data/home/amartyavarma/aloe-pose/sam-cache-active-v2pure/eval"
NSEL = 10
os.makedirs(OUT, exist_ok=True)
cfg = A.aloe_cfg()
sp = A.splits_v2(A.STORES)
ev = sp["eval"]
assert len(ev) == 3481, len(ev)
ci = pq.read_table(f"{CACHE}/items.parquet").to_pandas()
coff = np.load(f"{CACHE}/codes_off.npy")
codes = np.load(f"{CACHE}/codes.npy", mmap_mode="r")
pose = np.load(f"{CACHE}/pose.npy", mmap_mode="r")
assert len(coff) == len(ci) + 1, (len(coff), len(ci))
print("cache codes", codes.shape, codes.dtype, "pose", pose.shape, flush=True)
# cache item of each store row: k = ordinal of the clip in its recording by (start, speaker, segment_id)
byrec = {}
for i, r in ci.iterrows():
    byrec.setdefault(r["rid"], []).append(i)
for rid in byrec:
    byrec[rid].sort(key=lambda i: (ci.at[i, "start"], ci.at[i, "speaker"], ci.at[i, "segment_id"]))


def cache_of(row_key):
    _, _, rid, c = row_key.split("/")
    return byrec[rid][int(c[1:])]


def h8(s):
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")


cand = {}
for k, r, sw, n, key in ev:
    if not 188 <= n <= 375 or A.want_voice(key, cfg):
        continue
    i = cache_of(key)
    c = ci.loc[i]
    assert int(c["n"]) == n, (key, c["key"], c["n"], n)
    if c["lufs_status"] != "ok" or c["audio_crossbleed_status"] != "not_detected":
        continue
    w = json.loads(c["words"])
    dur = n / 12.5
    wps = len(w) / dur
    st = [max(0.0, float(x[0])) for x in w]
    en = [min(dur, float(x[1])) for x in w]
    gaps = np.diff(st) if len(st) > 1 else np.array([dur])
    if not 2.4 <= wps <= 3.6 or gaps.max() > 1.5 or st[0] > 0.5:
        continue
    cov = sum(max(0.0, b - a) for a, b in zip(st, en)) / dur
    rid = key.split("/")[2]
    it = dict(store=int(k), node_row=int(r), swap=int(sw), n=int(n), row_key=key, rid=rid, cache_i=int(i), cache_key=c["key"],
              wps=round(wps, 2), cov=round(cov, 3), max_gap=round(float(gaps.max()), 2), text=c["text"])
    if rid not in cand or cov > cand[rid]["cov"]:
        cand[rid] = it
recs = sorted(cand, key=lambda rid: h8("v2pure-samples|" + rid))
print("qualifying recordings", len(recs), flush=True)
sel = [cand[rid] for rid in recs[:NSEL]]
st_ = A.Stores(A.STORES)
rows = []
for j, it in enumerate(sel):
    m = st_.conv((it["store"], it["node_row"], it["swap"]), cfg, 0)
    P, n = int(m["P"]), it["n"]
    assert P == 5, ("expected an empty system block (no voice clip)", it["row_key"], P)
    pv = json.loads(m["prov"])
    assert pv["chan_A"] == "empty" and not pv["clips"], pv
    assert (m["ids"][0] == D.PAD_ID).all() and (m["depth"][0] < 0).all() and (m["present"][0] == 0).all(), "channel A not empty"
    assert m["ids"].shape[1] == P + n + D.CD, (m["ids"].shape, P, n)
    a0, a1 = coff[it["cache_i"]], coff[it["cache_i"] + 1]
    assert a1 - a0 == n, (a1 - a0, n)
    cut = lambda z: np.asarray(z[a0:a1] if z.shape[1] == D.DEPTH else z[:, a0:a1].T).astype(np.int64)  # noqa: E731  -> [n, 32]
    cc, pc = cut(codes), cut(pose)
    off = np.arange(D.DEPTH, dtype=np.int64) * D.CBS
    mb = m["depth"][1, P + D.CD:P + D.CD + n] - off
    pb = m["pdepth"][P + D.CD:P + D.CD + n] - off
    assert np.array_equal(mb, cc), ("B Mimi != cache codes", it["row_key"])
    assert np.array_equal(pb, pc), ("B pose != cache pose", it["row_key"])
    c = ci.loc[it["cache_i"]]
    norm = {"frame0_root": [float(x) for x in c["frame0_root"]], "frame0_width": float(c["frame0_width"]), "orig_w": float(c["orig_w"]),
            "orig_h": float(c["orig_h"])}
    row = f"B-{j:03d}"
    pickle.dump(m, open(f"{OUT}/{row}.pkl.tmp", "wb"))
    os.replace(f"{OUT}/{row}.pkl.tmp", f"{OUT}/{row}.pkl")
    rows.append(dict(it, row=row, P=P, T=int(len(m["pid"])), dur_s=round(n / 12.5, 2), item=m["item"], norm={"s1": norm}))
    print(row, {k: v for k, v in rows[-1].items() if k not in ("text", "norm")}, flush=True)
json.dump(rows, open(f"{OUT}/rows.json", "w"), indent=1)
print("done", len(rows), "rows; total gen s", round(sum(r["dur_s"] - 2.0 for r in rows), 1))
