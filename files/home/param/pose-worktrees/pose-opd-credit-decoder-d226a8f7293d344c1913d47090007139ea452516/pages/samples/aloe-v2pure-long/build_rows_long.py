"""aloe-v2pure LONG samples ROWS (10-18): 60 s windows from job 23051's HELD-OUT split (= work/build_rows.py of aloe-v2pure-200, adapted).
The held-out cache has NO same-speaker chunks that abut in time (every gap between consecutive segments of a speaker is >= 1 s; scan_contig.py),
but 181 held-out segments are themselves >= 60 s long (the trainer's rows are whole segments up to 7,500 frames, no crop), so each window
is the FIRST 750 frames (60.0 s) of ONE contiguous held-out segment: real audio / pose / words for the whole 60 s, no concatenation.
Row = the trainer-parity micro (data_aloe_pose.Stores.conv of that segment: B's empty system block, channel A empty) cut to P + 750 + CD
positions (ids / depth / present / pdepth / pid; the training-target arrays are dropped), + the clip's pose normalisation from the cache.
SELECTION (no ASR; deterministic, independent of any model output): eval items with n >= 750; NO voice clip in B's system block (P = 5);
cache lufs_status 'ok' and audio_crossbleed_status 'not_detected'; over the first 60 s: 2.2-3.8 words / s, no gap > MAXGAP s between
consecutive word starts, first word starts <= 0.5 s; per recording the qualifying segment with the highest word coverage in its first 60 s;
recordings ordered by blake2b8('v2pure-long|' + rid), first 10.
CHECKS: store row <-> cache item: n, B Mimi codes and B pose codes equal over the whole segment.
10-18 update (user: ~2 MINUTE clips): window length NW and row-name prefix are arguments (120 s = 1500 frames, rows M-xxx); the script
prints how many recordings qualify at several max-gap thresholds before selecting.
   python build_rows_long.py <out dir> [MAXGAP] [NW] [PREFIX]   (source aloe-v2pure-ft/aloe_v2pure.env first; compute node, CPU)"""
import hashlib
import json
import os
import pickle
import sys

import numpy as np
import pyarrow.parquet as pq

OUT = sys.argv[1]
MAXGAP = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5
assert os.environ.get("QQA_A_EMPTY") == "1" and os.environ.get("QQA_POSE_B") == "1", "source aloe-v2pure-ft/aloe_v2pure.env first"
sys.path.insert(0, "/data/home/amartyavarma/ppm/aloe-v2pure-ft/code")
import data_aloe_pose as A  # noqa: E402

D = A.D
CACHE = "/data/home/amartyavarma/aloe-pose/sam-cache-active-v2pure/eval"
NSEL = 10
NW = int(sys.argv[3]) if len(sys.argv) > 3 else 750          # window frames at 12.5 Hz (750 = 60 s, 1500 = 120 s)
PREFIX = sys.argv[4] if len(sys.argv) > 4 else "L"
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


cand, nlong, allc = {}, 0, []
for k, r, sw, n, key in ev:
    if n < NW:
        continue
    nlong += 1
    if A.want_voice(key, cfg):
        continue
    i = cache_of(key)
    c = ci.loc[i]
    assert int(c["n"]) == n, (key, c["key"], c["n"], n)
    if c["lufs_status"] != "ok" or c["audio_crossbleed_status"] != "not_detected":
        continue
    dur = NW / 12.5
    w = [x for x in json.loads(c["words"]) if float(x[0]) < dur]
    if len(w) < 2:
        continue
    st = [max(0.0, float(x[0])) for x in w]
    en = [min(dur, float(x[1])) for x in w]
    wps = len(w) / dur
    gaps = np.diff(st)
    if 2.2 <= wps <= 3.8 and st[0] <= 0.5:
        allc.append((key.split("/")[2], float(gaps.max())))
    if not 2.2 <= wps <= 3.8 or gaps.max() > MAXGAP or st[0] > 0.5:
        continue
    cov = sum(max(0.0, b - a) for a, b in zip(st, en)) / dur
    rid = key.split("/")[2]
    it = dict(store=int(k), node_row=int(r), swap=int(sw), n_seg=int(n), n=NW, row_key=key, rid=rid, cache_i=int(i), cache_key=c["key"],
              wps=round(wps, 2), cov=round(cov, 3), max_gap=round(float(gaps.max()), 2), seg_start_s=float(c["start"]),
              start_frame_25fps=int(c["start_frame_25fps"]), text=" ".join(str(x[2]) if len(x) > 2 else "" for x in w))
    if rid not in cand or cov > cand[rid]["cov"]:
        cand[rid] = it
for g in (1.5, 2.0, 2.5, 3.0, 4.0, 99):
    print(f"  max gap <= {g} s: {len({rid for rid, mg in allc if mg <= g})} recordings", flush=True)
recs = sorted(cand, key=lambda rid: h8("v2pure-long|" + rid))
print(f"held-out segments >= {NW} frames: {nlong}; qualifying recordings (max gap {MAXGAP} s): {len(recs)}", flush=True)
assert len(recs) >= int(os.environ.get("MIN_RECS", "8")), "too few qualifying recordings: relax MAXGAP"
sel = [cand[rid] for rid in recs[:NSEL]]
st_ = A.Stores(A.STORES)
rows = []
for j, it in enumerate(sel):
    m = st_.conv((it["store"], it["node_row"], it["swap"]), cfg, 0)
    P, ns = int(m["P"]), it["n_seg"]
    assert P == 5, ("expected an empty system block (no voice clip)", it["row_key"], P)
    pv = json.loads(m["prov"])
    assert pv["chan_A"] == "empty" and not pv["clips"], pv
    assert (m["ids"][0] == D.PAD_ID).all() and (m["depth"][0] < 0).all() and (m["present"][0] == 0).all(), "channel A not empty"
    assert m["ids"].shape[1] == P + ns + D.CD, (m["ids"].shape, P, ns)
    a0, a1 = coff[it["cache_i"]], coff[it["cache_i"] + 1]
    assert a1 - a0 == ns, (a1 - a0, ns)
    cut = lambda z: np.asarray(z[a0:a1] if z.shape[1] == D.DEPTH else z[:, a0:a1].T).astype(np.int64)  # noqa: E731  -> [n, 32]
    cc, pc = cut(codes), cut(pose)
    off = np.arange(D.DEPTH, dtype=np.int64) * D.CBS
    assert np.array_equal(m["depth"][1, P + D.CD:P + D.CD + ns] - off, cc), ("B Mimi != cache codes", it["row_key"])
    assert np.array_equal(m["pdepth"][P + D.CD:P + D.CD + ns] - off, pc), ("B pose != cache pose", it["row_key"])
    T = P + NW + D.CD                                                    # the 60 s window = the segment's first 750 frames
    w = {"kind": m["kind"], "ids": m["ids"][:, :T].copy(), "depth": m["depth"][:, :T].copy(), "present": m["present"][:, :T].copy(),
         "pid": m["pid"][:T].copy(), "pdepth": m["pdepth"][:T].copy(), "P": P, "swap": m["swap"], "prov": m["prov"], "row_key": m["row_key"],
         "item": m["item"], "window": {"seg_frames": ns, "frames": NW, "cut": f"first {NW} frames of the segment (positions [0, P + {NW + D.CD}))"}}
    assert (w["pdepth"][P + D.CD:P + D.CD + NW] >= 0).all() and (w["pdepth"][:P + D.CD] < 0).all()
    assert (w["depth"][1, P + D.CD:P + D.CD + NW] >= 0).all()
    c = ci.loc[it["cache_i"]]
    norm = {"frame0_root": [float(x) for x in c["frame0_root"]], "frame0_width": float(c["frame0_width"]), "orig_w": float(c["orig_w"]),
            "orig_h": float(c["orig_h"])}
    row = f"{PREFIX}-{j:03d}"
    pickle.dump(w, open(f"{OUT}/{row}.pkl.tmp", "wb"))
    os.replace(f"{OUT}/{row}.pkl.tmp", f"{OUT}/{row}.pkl")
    rows.append(dict(it, row=row, P=P, T=T, dur_s=NW / 12.5, item=m["item"], norm={"s1": norm}))
    print(row, {k: v for k, v in rows[-1].items() if k not in ("text", "norm")}, flush=True)
json.dump(rows, open(f"{OUT}/rows.json", "w"), indent=1)
print("done", len(rows), f"rows of {NW / 12.5:.1f} s (2 s prefill + {NW / 12.5 - 2:.1f} s generated each)")
