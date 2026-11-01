"""Independent check of the aloe-pose-ft-sam stores (does not use build_sam_stores.py or its cache):
  1. every non-codes file of every store byte-equal to the original (filecmp, shallow=False); STORES.json / PLAN.json too;
  2. codes arrays: same npy header (shape [R, 32, 7500], int16) as the original; every row: frames < n_frames in 0..2047, frames >= n_frames == -1;
  3. 64 random rows per store (seeded): decoded straight from the HF parquet blob (SPEAKER_00 -> assistant, SPEAKER_01 -> user), the row's
     window equals the store row exactly; the original row differs (not a copy).
usage: python -I verify_sam_stores.py <store k>    (writes <OUT>/verify/p<k>.json; exit 1 on any failure)"""
import filecmp
import io
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq

SRC, OUT, SAM = "/data/home/amartyavarma/ppm/aloe-pose-ft/stores", "/data/home/amartyavarma/ppm/aloe-pose-ft-sam", "/data/home/amartyavarma/data/aloe-sam-sep/hf"
SH = "rows-00000-of-00001"
k = int(sys.argv[1])
src, dst = f"{SRC}/p{k}", f"{OUT}/stores/p{k}"
res = {"store": k, "fail": []}
names = sorted(os.listdir(src))
for f in names:
    if f.startswith(f"{SH}.codes_"):
        continue
    if not filecmp.cmp(f"{src}/{f}", f"{dst}/{f}", shallow=False):
        res["fail"].append(f"file differs: {f}")
res["files_byte_equal"] = len([f for f in names if not f.startswith(f"{SH}.codes_")])
if k == 0:
    for f in ("STORES.json", "PLAN.json"):
        if not filecmp.cmp(f"{SRC}/{f}", f"{OUT}/stores/{f}", shallow=False):
            res["fail"].append(f"top file differs: {f}")
rows = pq.read_table(f"{src}/rows.parquet", columns=["row_key", "src_conversation", "frame_start", "n_frames"]).to_pylist()
n = np.array([r["n_frames"] for r in rows])
for s in ("user", "assistant"):
    o = np.load(f"{src}/{SH}.codes_{s}.npy", mmap_mode="r"); x = np.load(f"{dst}/{SH}.codes_{s}.npy", mmap_mode="r")
    if (o.shape, o.dtype, o.offset) != (x.shape, x.dtype, x.offset) or os.path.getsize(f"{src}/{SH}.codes_{s}.npy") != os.path.getsize(f"{dst}/{SH}.codes_{s}.npy"):
        res["fail"].append(f"codes_{s} header/size differs")
    bad = 0
    for i in range(len(rows)):
        a = np.asarray(x[i])
        if not ((a[:, n[i]:] == -1).all() and a[:, :n[i]].min() >= 0 and a[:, :n[i]].max() < 2048):
            bad += 1
    res[f"codes_{s}_rows_layout_bad"] = bad
    if bad:
        res["fail"].append(f"codes_{s}: {bad} rows with bad layout")
idx = {f"{r['session']}/{r['seg']}": r for r in pq.read_table(f"{SAM}/index/pairs.parquet", columns=["session", "seg", "token_shard", "token_row", "T"]).to_pylist()}
rng = np.random.default_rng(1000 + k)
pick = sorted(rng.choice(len(rows), 64, replace=False).tolist())
xs = {s: np.load(f"{dst}/{SH}.codes_{s}.npy", mmap_mode="r") for s in ("user", "assistant")}
os_ = {s: np.load(f"{src}/{SH}.codes_{s}.npy", mmap_mode="r") for s in ("user", "assistant")}
same_as_orig = 0
for i in pick:
    r = rows[i]; p = idx[r["src_conversation"]]
    pf = pq.ParquetFile(f"{SAM}/{p['token_shard']}")
    rg, off = 0, int(p["token_row"])
    while off >= pf.metadata.row_group(rg).num_rows:
        off -= pf.metadata.row_group(rg).num_rows; rg += 1
    t = pf.read_row_group(rg, columns=["session", "seg", "SPEAKER_00_isolated_npy", "SPEAKER_01_isolated_npy"]).slice(off, 1).to_pylist()[0]
    assert f"{t['session']}/{t['seg']}" == r["src_conversation"], (t["session"], t["seg"], r["src_conversation"])
    fs, nn = r["frame_start"], r["n_frames"]
    for s, col in (("assistant", "SPEAKER_00_isolated_npy"), ("user", "SPEAKER_01_isolated_npy")):
        w = np.load(io.BytesIO(t[col]))[:, fs:fs + nn]
        if not np.array_equal(np.asarray(xs[s][i])[:, :nn], w):
            res["fail"].append(f"row {i} {r['row_key']} {s}: != HF parquet window")
        same_as_orig += int(np.array_equal(np.asarray(os_[s][i])[:, :nn], w))
res["spot_rows"] = len(pick); res["spot_rows_equal_to_original"] = same_as_orig
os.makedirs(f"{OUT}/verify", exist_ok=True)
json.dump(res, open(f"{OUT}/verify/p{k}.json", "w"), indent=1)
print(json.dumps(res), flush=True)
sys.exit(1 if res["fail"] else 0)
