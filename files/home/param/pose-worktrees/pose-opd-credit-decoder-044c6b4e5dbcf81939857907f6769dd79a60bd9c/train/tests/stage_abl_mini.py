"""stage a MINI pose store + text store (metadata + only the shards the 8 local ranks' eval rows / packs need) from this training node's
/scratch to /data/home/amartyavarma/ppm/teststore (10-12 dependence-probe test on a spare node; deleted after the test)."""
import os, sys, shutil, json
sys.path.insert(0, "/data/home/amartyavarma/ppm/code"); sys.path.insert(0, "/data/home/amartyavarma/ppm/code/lib")
import numpy as np
import pose_s3q as PS
import txtdup_s3q as TXD
OUT = "/data/home/amartyavarma/ppm/teststore"
E_ = 2
pr = "/scratch/amartyavarma/pose-s3-samdata/rows"
node = PS.load_node(pr)
need = set()
for l in range(8):
    p = PS.plan(node, 20261011, 8, l, 1, 8, l, E_)
    for r in p["eval_rows"]:
        need.add(node["shards"][int(node["shard_of"][r])])
po = f"{OUT}/pose-mini"; os.makedirs(po, exist_ok=True)
for f in os.listdir(pr):
    if f in ("NODE.json", "rows.parquet") or f.endswith((".DONE", ".index.npz")) or any(f.startswith(s + ".") for s in need):
        shutil.copy2(f"{pr}/{f}", f"{po}/{f}")
print("pose shards", sorted(need))
tr = "/scratch/amartyavarma/ppm/txtdup-full"
st = TXD.Store(tr)
ks = set()
for p_ in range(8 * E_):
    for i in st.pack("eval", p_):
        ks.add(int(np.searchsorted(st.soff, i, side="right") - 1))
to = f"{OUT}/txt-mini"; os.makedirs(to, exist_ok=True)
for f in ("MANIFEST.json", "packs.npz"):
    shutil.copy2(f"{tr}/{f}", f"{to}/{f}")
for k in sorted(ks):
    shutil.copy2(f"{tr}/shard-{k:05d}.arrow", f"{to}/shard-{k:05d}.arrow")
print("text shards", sorted(ks), "node", node["node"])
