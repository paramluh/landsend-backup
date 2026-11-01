import json, sys
import pyarrow as pa, pyarrow.ipc as ipc
p = "/data/home/amartyavarma/qwen-1ch-data/s2h-m1-samsep/stores/m1n0/rows-00000-of-00001.persona.arrow"
t = ipc.open_file(pa.memory_map(p, "r")).read_all()
print(t.schema)
print(t.num_rows)
d = t.slice(0, 1).to_pylist()[0]
for k, v in d.items():
    s = str(v)
    print("==", k, ":", s[:1500])
