"""aloe-vox-ft: one built store -> its STORES.json entry (node digest, rows, pose digest, vox_index sha256, item lists) written to
plan/stores/p<K>.<host>.json; `merge` combines p0..p7 into plan/STORES.json (asserting every host's copy of a store agrees).
usage: python -I store_meta.py one --store K [--root /scratch/amartyavarma/aloe-vox-ft/stores] ; python -I store_meta.py merge"""
import glob, hashlib, json, os, socket, sys
sys.path.insert(0, "/data/home/amartyavarma/ppm/aloe-vox-ft/code/lib"); sys.path.insert(0, "/data/home/amartyavarma/ppm/aloe-vox-ft/code")
OUT = "/data/home/amartyavarma/ppm/aloe-vox-ft/plan"
if sys.argv[1] == "one":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--store", type=int, required=True)
    ap.add_argument("--root", default="/scratch/amartyavarma/aloe-vox-ft/stores"); a = ap.parse_args()
    os.environ["ALOE_STORE_ROOT"] = a.root
    import s3rows as R3
    import data_aloe_vox as A
    r = f"{a.root}/p{a.store}"
    assert os.path.isfile(f"{r}/BUILD_DONE") and os.path.isfile(f"{r}/VOX_DONE"), r
    n = R3.load_node(r); pj = json.load(open(f"{r}/POSE.json"))
    tr, ev = A.local_items(a.store)
    vi = json.load(open(f"{r}/vox_index.json"))["rows"]
    d = {"node_digest": n["digest"], "rows": int(len(n["length"])), "pose_digest": pj["pose_digest"],
         "row_convs": ["/".join(e["row_key"].split("/")[2:4]) for e in vi],
         "vox_sha256": hashlib.sha256(open(f"{r}/vox_index.json", "rb").read()).hexdigest(), "items_train": tr, "items_eval": ev,
         "host": socket.gethostname().split(".")[0]}
    os.makedirs(f"{OUT}/stores", exist_ok=True)
    json.dump(d, open(f"{OUT}/stores/p{a.store}.{d['host']}.json", "w"))
    print(f"p{a.store} on {d['host']}: rows {d['rows']} items train {len(tr)} eval {len(ev)} digest {d['node_digest']}")
else:
    out = {}
    for k in range(8):
        fs = sorted(glob.glob(f"{OUT}/stores/p{k}.*.json")); assert fs, ("store not built anywhere", k)
        ds = [json.load(open(f)) for f in fs]
        keys = ("node_digest", "rows", "pose_digest", "vox_sha256", "items_train", "items_eval", "row_convs")
        for d in ds[1:]:
            assert all(d[x] == ds[0][x] for x in keys), ("store copies differ", k, fs)
        out[f"p{k}"] = {x: ds[0][x] for x in keys}; out[f"p{k}"]["hosts"] = [d["host"] for d in ds]
    tot = sum(len(v["items_train"]) for v in out.values()); te = sum(len(v["items_eval"]) for v in out.values())
    assert tot == 27517 and te == 531, (tot, te)
    json.dump(out, open(f"{OUT}/STORES.json.tmp", "w")); os.replace(f"{OUT}/STORES.json.tmp", f"{OUT}/STORES.json")
    print("STORES.json:", {k: (v["rows"], len(v["items_train"]), len(v["items_eval"]), v["hosts"]) for k, v in out.items()}, "train", tot, "eval", te)
