"""after the 8 store builds: STORES.json (node digest per store as data_aloe_pose asserts it, file sizes, pose digest) + the totals."""
import json, os, sys, glob
sys.path.insert(0, "/data/home/amartyavarma/ppm/aloe-pose-ft/code")
import data_aloe_pose as A
out, tot = {}, {"rows": 0, "train_items": 0, "eval_items": 0, "bytes": 0}
for k, root in enumerate(A.SRC_STORES):
    assert os.path.exists(f"{root}/BUILD_DONE"), root
    node = A.D.R3.load_node(root)
    pj = json.load(open(f"{root}/POSE.json"))
    files = {os.path.basename(f): os.path.getsize(f) for f in sorted(glob.glob(f"{root}/*")) if os.path.isfile(f)}
    out[f"p{k}"] = {"node_digest": node["digest"], "pose_digest": pj["pose_digest"], "rows": pj["rows"], "files": files,
                    "b_items_train": pj["b_items_train"], "b_items_eval": pj["b_items_eval"]}
    tot["rows"] += pj["rows"]; tot["train_items"] += pj["b_items_train"]; tot["eval_items"] += pj["b_items_eval"]; tot["bytes"] += sum(files.values())
json.dump(out, open(A.DIGESTS_FILE, "w"), indent=1)
pl, nr = A.splits(A.SRC_STORES)
print("STORES.json written; totals", tot, "splits", {k: len(v) for k, v in pl.items()}, "rows", nr)
