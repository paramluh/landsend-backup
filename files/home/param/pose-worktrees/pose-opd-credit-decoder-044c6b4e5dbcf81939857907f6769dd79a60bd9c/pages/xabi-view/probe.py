"""xabi-view probe (10-27): inspect store row / cache item / trainer micro formats (read-only). Compute node, CPU."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/data/home/amartyavarma/ppm/aloe-v2pure-ft/code")
import data_aloe_pose as A  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

D = A.D
S = A.Stores()
node, st, sib, ps, pz, b_ok = S.open(0)
row = st[0]
print("ROW KEYS", sorted(row.keys()))
for k, v in row.items():
    if isinstance(v, np.ndarray):
        print(" ", k, v.dtype, v.shape)
    elif isinstance(v, (list, tuple)):
        print(" ", k, type(v).__name__, len(v), repr(v[:3])[:300])
    else:
        print(" ", k, repr(v)[:200])
t = pq.read_table("/data/home/amartyavarma/aloe-pose/sam-cache-active-v2pure/train-0467/items.parquet")
print("ITEMS COLS", t.schema)
r = t.slice(2, 1).to_pylist()[0]
print({k: (repr(v)[:200]) for k, v in r.items()})
cfg = A.aloe_cfg()
print("CFG", json.dumps({k: v for k, v in cfg.items() if not isinstance(v, (list, dict))})[:2000])
for r_ in range(0, 12):
    m = S.conv((0, r_, 1 if b_ok[r_, 0] else 0), cfg, 0)
    print("MICRO", r_, m["row_key"], "P", m["P"], "T", len(m["pid"]), "keys", sorted(m.keys()))
    if r_ < 3 or m["P"] > 5:
        print("  ids shape", m["ids"].shape, "depth", m["depth"].shape, "pdepth", m["pdepth"].shape)
        P = int(m["P"])
        print("  sys ids", m["ids"][1][:P].tolist()[:200])
        print("  depth1 prefix any>=0", int((m["depth"][1][:P, 0] >= 0).sum()))
        print("  conv ids first 40", m["ids"][1][P:P + 40].tolist())
        print("  depth1 conv first 4 cb0", m["depth"][1][P:P + 6, :4].tolist())
        print("  pdepth conv", m["pdepth"][P:P + 6, :4].tolist())
        print("  ids0", m["ids"][0][:8].tolist(), "present", m["present"][:, :3].tolist() if m["present"].ndim > 1 else m["present"][:5])
        print("  prov", m["prov"][:1500])
        if r_ >= 3:
            break
print("CONST PAD", D.PAD_ID, "NW", D.NW_ID, "EOB", D.EOB_ID, "CTRL", D.CTRL_LO, "CD", D.CD, "CBS", D.CBS)
