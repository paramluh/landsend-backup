"""aloe-8b-vs-32b ROWS (10-18): the 8 held-out clips of the 8B SAM v2pure samples page (sam-aloe-ft-v2pure/samples/work/clips.json, idx =
v2pure eval cache index) as trainer-parity micros of job 23051 (data_aloe_pose.Stores.conv, the run's own code + env: the clip's own segment,
B lane / Mimi / pose, channel A empty) = aloe-v2pure-200/work/build_rows.py's construction, with ONE choice: B's system block EMPTY (no voice
clip; P = 5, the trained form of 2/3 of the rows) so the 32B's only audio/pose context is the 2 s prefill, as the 8B's (which has no voice
clip either). Store row <-> cache item via the run's row_key ordinal rule (c<k> = k-th clip of the recording by (start, speaker, segment_id));
checks: n, B Mimi codes and B pose codes equal the cache (= the 8B's inputs/{idx}.npz), key equal.
   python build_rows.py <out dir>      (source aloe-v2pure-ft/aloe_v2pure.env first; compute node, CPU)"""
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
S8 = "/data/home/amartyavarma/sam-aloe-ft-v2pure/samples"
os.makedirs(OUT, exist_ok=True)
cfg = A.aloe_cfg()
assert "voice_mod" in cfg, sorted(cfg)
cfg = dict(cfg, voice_mod=1 << 62)                     # no voice clip in B's system block (asserted P == 5 below)
ev = A.splits_v2(A.STORES)["eval"]
ci = pq.read_table(f"{CACHE}/items.parquet").to_pandas()
coff = np.load(f"{CACHE}/codes_off.npy")
codes = np.load(f"{CACHE}/codes.npy", mmap_mode="r")
pose = np.load(f"{CACHE}/pose.npy", mmap_mode="r")
byrec = {}
for i, r in ci.iterrows():
    byrec.setdefault(r["rid"], []).append(i)
for rid in byrec:
    byrec[rid].sort(key=lambda i: (ci.at[i, "start"], ci.at[i, "speaker"], ci.at[i, "segment_id"]))
item_of = {}
for k, r, sw, n, key in ev:
    _, _, rid, c = key.split("/")
    item_of[int(byrec[rid][int(c[1:])])] = (k, r, sw, n, key)
clips = json.load(open(f"{S8}/work/clips.json"))
assert len(clips) == 8
st = A.Stores(A.STORES)
rows = []
for j, c in enumerate(clips):
    i = int(c["idx"])
    k, r, sw, n, key = item_of[i]
    x = ci.loc[i]
    assert x["key"] == c["key"] and int(x["n"]) == n == int(c["F"]), (c, x["key"], n)
    m = st.conv((k, r, sw), cfg, 0)
    P = int(m["P"])
    assert P == 5, ("expected an empty system block", key, P)
    pv = json.loads(m["prov"])
    assert pv["chan_A"] == "empty" and not pv["clips"] and pv["voice_drawn"] == 0, pv
    assert (m["ids"][0] == D.PAD_ID).all() and (m["depth"][0] < 0).all() and (m["present"][0] == 0).all(), "channel A not empty"
    assert m["ids"].shape[1] == P + n + D.CD
    a0, a1 = coff[i], coff[i + 1]
    assert a1 - a0 == n
    cut = lambda z: np.asarray(z[a0:a1] if z.shape[1] == D.DEPTH else z[:, a0:a1].T).astype(np.int64)  # noqa: E731
    cc, pc = cut(codes), cut(pose)
    off = np.arange(D.DEPTH, dtype=np.int64) * D.CBS
    assert np.array_equal(m["depth"][1, P + D.CD:P + D.CD + n] - off, cc), ("B Mimi != cache", key)
    assert np.array_equal(m["pdepth"][P + D.CD:P + D.CD + n] - off, pc), ("B pose != cache", key)
    z8 = np.load(f"{S8}/inputs/{i}.npz")
    assert np.array_equal(z8["pose"], pc.T) and np.array_equal(z8["audio"], cc.T), ("!= the 8B page's staged inputs", i)
    m8 = json.load(open(f"{S8}/inputs/{i}.json"))
    norm = {"frame0_root": [float(v) for v in x["frame0_root"]], "frame0_width": float(x["frame0_width"]), "orig_w": float(x["orig_w"]),
            "orig_h": float(x["orig_h"])}
    row = f"B-{j:03d}"
    pickle.dump(m, open(f"{OUT}/{row}.pkl.tmp", "wb"))
    os.replace(f"{OUT}/{row}.pkl.tmp", f"{OUT}/{row}.pkl")
    rows.append(dict(row=row, idx=i, key=c["key"], row_key=key, store=int(k), node_row=int(r), swap=int(sw), n=int(n), P=P,
                     T=int(len(m["pid"])), dur_s=round(n / 12.5, 2), item=m["item"], text=m8["text"], global_state0=m8["global_state0"],
                     orig_hw=m8["orig_hw"], norm={"s1": norm}))
    print(row, i, c["key"], key, n, flush=True)
json.dump(rows, open(f"{OUT}/rows.json", "w"), indent=1)
print("done", len(rows))
