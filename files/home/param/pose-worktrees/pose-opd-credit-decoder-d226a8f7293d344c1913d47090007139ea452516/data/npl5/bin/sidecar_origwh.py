"""npl5 SIDECAR (user change #2, 10-27: keep each segment's pose-struct image size for a later aspect-ratio feature): for one slot's raw
share, read ONLY segment_id, clip_id and the nested pose.orig_h / pose.orig_w subfields of every part-*.parquet (never the whole pose column)
-> W/scan/origwh/<slot>.parquet: file (job/release/part, = the scan's / plan's 'file'), row (index in that parquet, = the scan's 'row'),
segment_id, clip_id, orig_h, orig_w; + <slot>.json. A sidecar only: the scan (W/scan/<slot>.parquet) and the plans are untouched (byte-identical
to on-host fresh_pose output). Kept in the scan/origwh/ SUBDIR so fresh_pose's prior-scan glob (<scan dir>/*.parquet) never picks it up.
usage (qwen-s2 venv, python -I):  sidecar_origwh.py <slot raw root holding jobs/> <slot> [--workers 2]"""
import argparse
import json
import multiprocessing as mp
import os
import time

import pyarrow as pa
import pyarrow.parquet as pq

W = os.environ.get("NPL5_W", "/data/home/amartyavarma/ppm/pose5/npl5")
ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("slot")
ap.add_argument("--workers", type=int, default=2)
a = ap.parse_args()
J = f"{a.root}/jobs"
O = f"{W}/scan/origwh"
os.makedirs(O, exist_ok=True)


def one(rel):
    # ParquetFile.read decodes only the selected leaf columns (2.4 ms/file); pq.read_table(columns=[nested]) took 28 ms/file for the same
    # values (checked equal on 60 files, test/nested_read_test2.py)
    t = pq.ParquetFile(f"{J}/{rel}").read(columns=["segment_id", "clip_id", "pose.orig_h", "pose.orig_w"])
    p = t.column("pose").combine_chunks()
    return rel, t.num_rows, t.column("segment_id").to_pylist(), t.column("clip_id").to_pylist(), p.field("orig_h").to_pylist(), p.field("orig_w").to_pylist()


t0 = time.time()
files = sorted(f"{j}/{r}/{p}" for j in os.listdir(J) for r in os.listdir(f"{J}/{j}") for p in os.listdir(f"{J}/{j}/{r}")
               if p.startswith("part-") and p.endswith(".parquet"))
cols = {k: [] for k in ("file", "row", "segment_id", "clip_id", "orig_h", "orig_w")}
with mp.get_context("fork").Pool(a.workers) as pool:
    for rel, n, s, c, h, w in pool.imap(one, files, chunksize=16):
        cols["file"] += [rel] * n; cols["row"] += list(range(n)); cols["segment_id"] += s; cols["clip_id"] += c
        cols["orig_h"] += h; cols["orig_w"] += w
tb = pa.table({"file": pa.array(cols["file"], pa.string()), "row": pa.array(cols["row"], pa.int32()), "segment_id": pa.array(cols["segment_id"], pa.string()),
               "clip_id": pa.array(cols["clip_id"], pa.string()), "orig_h": pa.array(cols["orig_h"], pa.float64()), "orig_w": pa.array(cols["orig_w"], pa.float64())})
pq.write_table(tb, f"{O}/{a.slot}.parquet.tmp"); os.replace(f"{O}/{a.slot}.parquet.tmp", f"{O}/{a.slot}.parquet")
nh = sum(x is None for x in cols["orig_h"])
info = {"slot": a.slot, "files": len(files), "segments": tb.num_rows, "null_orig_h": nh, "seconds": round(time.time() - t0, 1),
        "orig_hw_top": {f"{h:g}x{w:g}": k for (h, w), k in sorted(__import__("collections").Counter(zip(cols["orig_h"], cols["orig_w"])).items(),
                                                                  key=lambda x: -x[1])[:8]}}
json.dump(info, open(f"{O}/{a.slot}.json", "w"), indent=1)
print(json.dumps(info), flush=True)
