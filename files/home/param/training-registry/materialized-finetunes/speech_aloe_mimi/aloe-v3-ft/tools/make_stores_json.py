"""plan/STORES.json: the 4 built v3 stores' digests (s3rows.load_node digest), rows, chunk-0 rows, file sizes; the trainer and staging assert
staged copies against it."""
import json, os, sys
sys.path.insert(0, "/data/home/amartyavarma/ppm/aloe-v3-ft/code/lib")
os.environ.setdefault("S3_MIN_FRAMES", "125")
import s3rows as F
import numpy as np
S = "/data/home/amartyavarma/ppm/aloe-v3-ft/stores"
out = {"format": "aloe-v3-stores-v1", "root": S, "stores": []}
for k in range(4):
    r = f"{S}/v3n{k}"
    n = F.load_node(r)
    z = np.load(f"{r}/rows-00000-of-00001.index.npz")
    out["stores"].append({"name": f"v3n{k}", "digest": n["digest"], "rows": int(len(n["length"])), "chunk0_rows": int((z["chunk_index"] == 0).sum()),
                          "row_frames": int(n["meta"]["row_frames"]), "files": {f: os.path.getsize(f"{r}/{f}") for f in sorted(os.listdir(r))}})
json.dump(out, open("/data/home/amartyavarma/ppm/aloe-v3-ft/plan/STORES.json", "w"), indent=1)
print(json.dumps([(s["name"], s["digest"], s["rows"], s["chunk0_rows"]) for s in out["stores"]]))
