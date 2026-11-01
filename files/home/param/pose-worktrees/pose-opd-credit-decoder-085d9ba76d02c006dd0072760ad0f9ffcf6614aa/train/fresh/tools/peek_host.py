import os, sys, json, glob, collections
import pyarrow.parquet as pq
root = "/scratch/amartyavarma/new-pose-llm-data/new-pose-llm-data/jobs"
jobs = sorted(os.listdir(root))
print("jobs", len(jobs))
cnt = collections.Counter(); nrel = collections.Counter()
ex = None; ex_unsealed = None
for j in jobs:
    rels = os.listdir(f"{root}/{j}")
    nrel[len(rels)] += 1
    for r in rels:
        fs = set(os.listdir(f"{root}/{j}/{r}"))
        cnt["complete" if "complete.json" in fs else "nocomplete"] += 1
        cnt["release" if "release.json" in fs else "norelease"] += 1
        cnt["parquet_any" if any(f.endswith(".parquet") for f in fs) else "noparquet"] += 1
        if "complete.json" in fs and ex is None: ex = f"{root}/{j}/{r}"
        if "complete.json" not in fs and "release.json" in fs and ex_unsealed is None: ex_unsealed = f"{root}/{j}/{r}"
print(dict(cnt), "releases per job", dict(nrel))
for d in (ex, ex_unsealed):
    if d is None: continue
    print("==", d, sorted(os.listdir(d)))
    for f in ("complete.json", "release.json"):
        if os.path.exists(f"{d}/{f}"):
            t = json.load(open(f"{d}/{f}"))
            s = json.dumps(t)
            print(f, "keys", list(t.keys()) if isinstance(t, dict) else type(t), s[:1500])
    ps = sorted(glob.glob(f"{d}/part-*.parquet"))
    if ps:
        f = pq.ParquetFile(ps[0]); print("rows", f.metadata.num_rows); print(f.schema_arrow)
        r = f.read_row_group(0).slice(0, 1).to_pylist()[0]
        for k, v in r.items():
            if k in ("pose", "audio"):
                for kk, vv in v.items():
                    sv = repr(vv); print(f"  {k}.{kk}: {sv[:200]}")
            else:
                print(f"  {k}: {repr(v)[:600]}")
