"""v2pure cache summary + cross-check vs posev3 cache (informational: items in both, and for those, pose/text/frames equality; Mimi changed frac).
usage: verify_v2pure.py --workers W  -> sam-cache-active-v2pure/VERIFY.json"""
import argparse, json, os, sys
import numpy as np
V2P = "/data/home/amartyavarma/aloe-pose/sam-cache-active-v2pure"; P3 = "/data/home/amartyavarma/aloe-pose/sam-cache-active-posev3"

def load(root, name):
    keys = open(f"{root}/{name}/audio_id.txt").read().split("\n"); off = np.load(f"{root}/{name}/codes_off.npy")
    toff = np.load(f"{root}/{name}/text_off.npy")
    return keys, off, toff

def one(name):
    import pyarrow.parquet as pq
    k2, o2, t2 = load(V2P, name); c2 = np.load(f"{V2P}/{name}/codes.npy", mmap_mode="r"); p2 = np.load(f"{V2P}/{name}/pose.npy", mmap_mode="r")
    x2 = np.load(f"{V2P}/{name}/text.npy", mmap_mode="r")
    r = {"shard": name, "items": len(k2), "frames": int(o2[-1]), "codes_shape_eq_pose": c2.shape == p2.shape and c2.shape[1] == 32,
         "cmin": int(min(c2.min(), p2.min())), "cmax": int(max(c2.max(), p2.max())), "keys": k2}
    return r

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=24); a = ap.parse_args()
    import multiprocessing as mp
    names = sorted(d for d in os.listdir(V2P) if (d.startswith("train-") or d == "eval") and not d.endswith(".tmp"))
    with mp.get_context("fork").Pool(a.workers) as pool:
        res = pool.map(one, names, chunksize=4)
    # posev3 key -> (shard, idx)
    p3 = {}
    for n in sorted(d for d in os.listdir(P3) if d.startswith("train-") or d == "eval"):
        for i, k in enumerate(open(f"{P3}/{n}/audio_id.txt").read().split("\n")):
            p3[k] = (n, i)
    v2k = {}
    for r in res:
        for i, k in enumerate(r["keys"]):
            v2k[k] = (r["shard"], i)
    both = sorted(set(v2k) & set(p3)); only_v2 = len(set(v2k) - set(p3)); only_p3 = len(set(p3) - set(v2k))
    ev2 = {k for k, (s, _) in v2k.items() if s == "eval"}; ev3 = {k for k, (s, _) in p3.items() if s == "eval"}
    # sample 3000 shared items: pose/text/frames equality vs posev3; Mimi changed frac
    rng = np.random.default_rng(0); samp = [both[i] for i in rng.choice(len(both), min(3000, len(both)), replace=False)]
    cache = {}
    def arr(root, n):
        if (root, n) not in cache:
            cache[(root, n)] = (np.load(f"{root}/{n}/codes.npy", mmap_mode="r"), np.load(f"{root}/{n}/pose.npy", mmap_mode="r"),
                                np.load(f"{root}/{n}/text.npy", mmap_mode="r"), np.load(f"{root}/{n}/codes_off.npy"), np.load(f"{root}/{n}/text_off.npy"))
        return cache[(root, n)]
    pose_eq = text_eq = 0; cf = []
    for k in samp:
        (s2, i2), (s3, i3) = v2k[k], p3[k]
        c2, q2, x2, o2, t2 = arr(V2P, s2); c3, q3, x3, o3, t3 = arr(P3, s3)
        A, B = np.asarray(q2[o2[i2]:o2[i2 + 1]]), np.asarray(q3[o3[i3]:o3[i3 + 1]])
        pose_eq += int(A.shape == B.shape and np.array_equal(A, B))
        text_eq += int(np.array_equal(np.asarray(x2[t2[i2]:t2[i2 + 1]]), np.asarray(x3[t3[i3]:t3[i3 + 1]])))
        if A.shape == B.shape:
            cf.append(float((np.asarray(c2[o2[i2]:o2[i2 + 1]]) != np.asarray(c3[o3[i3]:o3[i3 + 1]])).mean()))
    v = {"shards": len(res), "items_train": sum(r["items"] for r in res if r["shard"] != "eval"),
         "items_eval": sum(r["items"] for r in res if r["shard"] == "eval"),
         "frames_train": sum(r["frames"] for r in res if r["shard"] != "eval"),
         "all_codes_shape_eq_pose": all(r["codes_shape_eq_pose"] for r in res), "code_min": min(r["cmin"] for r in res), "code_max": max(r["cmax"] for r in res),
         "vs_posev3": {"items_in_both": len(both), "only_v2pure": only_v2, "only_posev3": only_p3, "eval_items_v2pure": len(ev2),
                       "eval_items_posev3": len(ev3), "eval_items_in_both": len(ev2 & ev3), "sampled_shared": len(samp),
                       "pose_equal_frac": round(pose_eq / max(len(samp), 1), 4), "text_equal_frac": round(text_eq / max(len(samp), 1), 4),
                       "mimi_changed_frac_mean": round(float(np.mean(cf)), 4) if cf else None}}
    v["ok"] = bool(v["all_codes_shape_eq_pose"] and v["code_min"] >= 0 and v["code_max"] <= 2047 and v["items_train"] > 0 and v["items_eval"] > 0)
    json.dump(v, open(f"{V2P}/VERIFY.json", "w"), indent=1); print(json.dumps(v, indent=1)); sys.exit(0 if v["ok"] else 3)

if __name__ == "__main__":
    main()
