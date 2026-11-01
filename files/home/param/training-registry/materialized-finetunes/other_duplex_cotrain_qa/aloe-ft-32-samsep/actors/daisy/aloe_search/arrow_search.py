"""Case-insensitive 'dais' in the Aloe store persona sidecars (zstd arrow; s2h-m1 + samsep, all rows/chunks/splits) and the handover manifest."""
import glob, json, re
import pyarrow as pa, pyarrow.ipc as ipc
pat = re.compile(r"dais", re.I)
for f in sorted(glob.glob("/data/home/amartyavarma/qwen-1ch-data/s2h-m1*/stores/*/*.persona.arrow")):
    t = ipc.open_file(pa.memory_map(f, "r")).read_all().to_pylist()
    n = 0
    for r in t:
        for s in ("user", "assistant"):
            sp = r.get(f"{s}_system_prompt") or ""
            for m in pat.finditer(sp):
                n += 1
                if n <= 5:
                    print(f.split("/stores/")[0][-12:], r["row_key"], s, "|", sp[max(0, m.start() - 80):m.end() + 60].replace("\n", " "))
    print(f, "hits", n, "rows", len(t))
