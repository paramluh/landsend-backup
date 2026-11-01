"""Copy the picked Daisy rows (daisy/picks.json, this host only) out of the node-local PPM store /scratch/amartyavarma/s3-rows: read-only,
mmap, only those rows' codes + word timings -> daisy/rows/<node>_<shard>_<row>.npz (+ .json). usage: extract.py HOST"""
import json, os, sys
import numpy as np
sys.path.insert(0, "/data/home/amartyavarma/qwen-1ch/stage3/data/rows_build/scripts")
import s3rows as F  # noqa: E402
D = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/actors/daisy"
host = sys.argv[1]
picks = [p for p in json.load(open(f"{D}/picks.json")) if p["node"] == host]
node = F.load_node("/scratch/amartyavarma/s3-rows")
rows = F.Rows(node)
off = {}
o = 0
for k, s in enumerate(node["shards"]):
    off[s] = o
    o += int((node["shard_of"] == k).sum())
for p in picks:
    i = off[p["shard"]] + p["row_in_shard"]
    r = rows[i]
    if r.get("row_key") not in (None, p["row_key"]):
        print("MISMATCH", r.get("row_key"), p["row_key"], flush=True)
        continue
    n = int(r["n_frames"])
    tag = f"{host}_{p['shard']}_{p['row_in_shard']}"
    np.savez(f"{D}/rows/{tag}.npz", codes_user=np.asarray(r["codes_user"])[:, :n].copy(), codes_assistant=np.asarray(r["codes_assistant"])[:, :n].copy())
    json.dump({**p, "n_frames": n, "row_key_store": r.get("row_key"), "words_user": [dict(w) for w in r["words_user"]],
               "words_assistant": [dict(w) for w in r["words_assistant"]]}, open(f"{D}/rows/{tag}.json", "w"))
    print("ok", tag, n, flush=True)
