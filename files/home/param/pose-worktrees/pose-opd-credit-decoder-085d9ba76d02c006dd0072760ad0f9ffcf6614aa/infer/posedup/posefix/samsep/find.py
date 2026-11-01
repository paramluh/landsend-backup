import json, os, glob
import pyarrow.parquet as pq
D = "/data/home/amartyavarma/data/aloe-sam-sep/hf/index"
pairs = pq.read_table(f"{D}/pairs.parquet").to_pandas()
print("pairs columns", list(pairs.columns), len(pairs))
print(pairs.head(3).to_string()[:1500])
G = "/data/home/amartyavarma/ppm/aloe-posedup-live/genweb/openings"
ids = {}
for p in sorted(os.listdir(G)):
    f = f"{G}/{p}/meta.json"
    if os.path.isfile(f):
        ids[p] = json.load(open(f))["recording_id"]
cols = [c for c in pairs.columns if pairs[c].dtype == object]
for p, rid in ids.items():
    hit = {c: int(pairs[c].astype(str).str.contains(rid[:16]).sum()) for c in cols}
    print(p, rid, {k: v for k, v in hit.items() if v})
# also: how many keys look like 40-hex vs uuid
sess = pairs[cols[0]].astype(str)
print("example keys", sess.head(5).tolist())
print("40hex-like", int(sess.str.match(r"^[0-9a-f]{40}").sum()), "uuid-like", int(sess.str.match(r"^[0-9a-f]{8}-").sum()))
