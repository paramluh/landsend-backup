import pyarrow.parquet as pq
t = pq.read_table("/data/home/amartyavarma/aloe-pose/sam-cache-active-v2pure/eval/items.parquet")
print(t.schema)
d = t.slice(0, 2).to_pylist()
for x in d: print({k: (str(v)[:120]) for k, v in x.items()})
