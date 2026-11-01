"""Pick ~10 Daisy chunks (distinct podcasts, most hours first, hosts of job 29005 only) and locate them in the node-local PPM stores via the
/data persona sidecars (sidecars/<node>/<shard>.persona.arrow rows are in store order) -> daisy/picks.json (node, shard, row, row_key, ...)."""
import collections, glob, json, os
import pyarrow as pa, pyarrow.ipc as ipc
D = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/actors/daisy"
SC = "/data/home/amartyavarma/qwen-1ch-data/persona/sidecars"
HOSTS = {f"slinky-{i}" for i in (0, 1, 2, 3, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 19)}
R = json.load(open(f"{D}/daisy.json"))["rows"]
by = collections.defaultdict(list)
for r in R:
    by[r["ek"].split("/")[1]].append(r)
pods = sorted(by.items(), key=lambda kv: -sum(r["sec"] for r in kv[1]))
want = []
for pod, rs in pods:
    rs = [r for r in rs if r["node"] in HOSTS]
    if rs:
        want.append(max(rs, key=lambda r: r["sec"]))
    if len(want) == 12:
        break
need = collections.defaultdict(dict)
for w in want:
    need[w["node"]][(w["ek"], w["chunk"])] = w
picks = []
for node, ws in need.items():
    for f in sorted(glob.glob(f"{SC}/{node}/*.persona.arrow")):
        t = ipc.open_file(pa.memory_map(f, "r")).read_all().select(["row_key"]).column(0).to_pylist()
        for i, k in enumerate(t):
            for (ek, ch), w in ws.items():
                if k.startswith(ek) and k.endswith(f"{ch:02d}") and len(k) <= len(ek) + 4:
                    picks.append({**{x: w[x] for x in ("ek", "side", "chunk", "sec", "partner_name", "who", "node")}, "row_key": k,
                                  "shard": os.path.basename(f)[:-len(".persona.arrow")], "row_in_shard": i})
print(json.dumps([(p["node"], p["shard"], p["row_in_shard"], p["row_key"][-60:]) for p in picks], indent=0))
json.dump(picks, open(f"{D}/picks.json", "w"), indent=0)
print("picked", len(picks), "of", len(want))
