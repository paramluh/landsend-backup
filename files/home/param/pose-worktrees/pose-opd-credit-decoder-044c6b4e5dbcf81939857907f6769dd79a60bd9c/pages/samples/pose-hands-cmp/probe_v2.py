import pyarrow.parquet as pq, numpy as np, json
m = pq.read_schema("/data/home/amartyavarma/aloe-pose/meta-v2/manifest.parquet"); print(m.names)
t = pq.read_table("/data/home/amartyavarma/aloe-pose/meta-v2/manifest.parquet").slice(0, 2).to_pylist()
for r in t: print({k: str(v)[:120] for k, v in r.items()})
E = "/data/home/amartyavarma/aloe-pose/sam-cache-active-v2pure/eval"
it = pq.read_table(f"{E}/items.parquet"); print(it.schema.names); print({k: str(v)[:150] for k, v in it.slice(0,1).to_pylist()[0].items()})
P = np.load(f"{E}/pose.npy", mmap_mode="r"); C = np.load(f"{E}/codes.npy", mmap_mode="r"); off = np.load(f"{E}/codes_off.npy")
print(P.shape, P.dtype, C.shape, off[:5], len(off))
print(json.load(open(f"{E}/DONE.json")))
