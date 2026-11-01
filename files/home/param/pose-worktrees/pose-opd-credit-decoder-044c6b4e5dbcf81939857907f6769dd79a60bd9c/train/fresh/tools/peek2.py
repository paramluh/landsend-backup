import os, sys, json, glob, collections
import pyarrow.parquet as pq
root = "/scratch/amartyavarma/new-pose-llm-data/new-pose-llm-data/jobs"
jobs = sorted(os.listdir(root))
st = collections.Counter(); ex = {}
for j in jobs[:3000]:
    for r in os.listdir(f"{root}/{j}"):
        d = f"{root}/{j}/{r}"; fs = set(os.listdir(d))
        if "release.json" in fs:
            t = json.load(open(f"{d}/release.json"))
            key = (t.get("state"), "complete.json" in fs, t.get("selection") if not isinstance(t.get("selection"), dict) else json.dumps(t.get("selection"))[:80])
            st[(t.get("state"), "complete.json" in fs)] += 1
            ex.setdefault((t.get("state"), "complete.json" in fs), d)
        else:
            st[("NO_RELEASE", "complete.json" in fs, tuple(sorted(fs))[:6])] += 1
            ex.setdefault(("NO_RELEASE", "complete.json" in fs), d)
for k, v in st.most_common(): print(v, k)
for k, d in ex.items():
    print("==", k, d, sorted(os.listdir(d)))
    if os.path.exists(f"{d}/release.json"):
        t = json.load(open(f"{d}/release.json"))
        print("  files", json.dumps(t.get("files"))[:600]); print("  rows", t.get("rows"), "selection", json.dumps(t.get("selection"))[:300])
        print("  exceptions", json.dumps(t.get("exceptions"))[:300]); print("  retryable", json.dumps(t.get("retryable_failures"))[:300])
d = ex.get(("completed", True)) or ex.get(("completed_with_exceptions", True))
ps = sorted(glob.glob(f"{d}/part-*.parquet"))
r = pq.ParquetFile(ps[0]).read_row_group(0).slice(0, 1).to_pylist()[0]
for k, v in r.items():
    if k in ("pose", "audio"):
        for kk, vv in v.items():
            if kk.startswith("tokens_cb") and kk not in ("tokens_cb0", "tokens_cb31"): continue
            print(f"  {k}.{kk}: {repr(vv)[:300]}")
    else:
        print(f"  {k}: {repr(v)[:1200]}")
