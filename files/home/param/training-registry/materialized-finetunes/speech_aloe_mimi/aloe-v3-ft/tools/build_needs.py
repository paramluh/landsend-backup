"""plan/NEEDS.json: per training node i (ranks 8i..8i+7, world 64) every channel-A audio file its ranks read: train item idx -> rank idx % 64,
eval item j -> rank j % 64, the fallback dummy item (plan item 0) on every node (= build_m1vox_index's NEEDS rule) + per-node totals."""
import collections, json, os, sys
os.environ.setdefault("S3_MIN_FRAMES", "125"); os.environ.setdefault("PPM_CD", "2")
os.environ["ALOE_STORE_ROOT"] = "/data/home/amartyavarma/ppm/aloe-v3-ft/stores"
sys.path.insert(0, "/data/home/amartyavarma/ppm/aloe-v3-ft/code")
import data_aloe_v3 as A
WORLD, LW = 64, 8
PL = A.make_plan(WORLD, roots=A.SRC_STORES)
st = A.Stores(A.SRC_STORES)
need = collections.defaultdict(dict)
def add(node, item):
    k, r, sw = item
    aside = "assistant" if int(sw) else "user"
    e = st.vox(int(k))[int(r)]["A"][aside]
    if e["kind"] == "gcs":
        need[node][e["flac"]] = {"kind": "gcs", "rel": e["flac"], "uri": e["uri"], "bytes": e["flac_bytes"], "generation": e["generation"]}
    else:
        need[node][e["flac"]] = {"kind": "mimidec", "rel": e["flac"], "store": int(k), "row": int(r), "side": aside, "n_frames": e["n_frames"]}
for i, it in enumerate(PL["items"]):
    add((i % WORLD) // LW, it)
for j, it in enumerate(PL["eval"]):
    add((j % WORLD) // LW, it)
for node in range(WORLD // LW):
    add(node, PL["items"][0])
nd = {str(n): sorted(v.values(), key=lambda t: t["rel"]) for n, v in sorted(need.items())}
json.dump(nd, open("/data/home/amartyavarma/ppm/aloe-v3-ft/plan/NEEDS.json", "w"))
print("steps", PL["steps"], "items", len(PL["items"]), "eval", len(PL["eval"]), "per node (gcs files, GB, mimidec):",
      {n: (sum(t["kind"] == "gcs" for t in v), round(sum(t.get("bytes", 0) for t in v) / 1e9, 1), sum(t["kind"] == "mimidec" for t in v)) for n, v in nd.items()})
