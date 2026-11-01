"""pose motion stats over the GENERATED frames (pose frame >= 125 = after the 10 s prefill), per arm, pooled over runs and both speakers:
cb0 distinct values, cb0 frame-to-frame repeat %, light (cb1..31) mean distinct codes per codebook and mean change rate; GT on the same frames.
usage: python pose_stats.py <conv ids e.g. 0,2,4> <arm>=<runs dir> [...]"""
import json, sys
import numpy as np
NPRE = 125
ids = [int(x) for x in sys.argv[1].split(",")]
arms = [a.split("=", 1) for a in sys.argv[2:]]


def stats(xs):
    c0 = [x[0] for x in xs]
    dist0 = np.mean([len(np.unique(c)) for c in c0])
    rep0 = 100 * np.mean(np.concatenate([c[1:] == c[:-1] for c in c0]))
    ld = np.mean([np.mean([len(np.unique(x[k])) for k in range(1, 32)]) for x in xs])
    lch = 100 * np.mean([np.mean(x[1:, 1:] != x[1:, :-1]) for x in xs])
    return {"cb0_distinct": round(float(dist0), 1), "cb0_repeat_pct": round(float(rep0), 1), "light_distinct": round(float(ld), 1), "light_change_pct": round(float(lch), 1)}


out = {}
gt = []
for name, rd in arms:
    xs = []
    for i in ids:
        for s in ("s1", "s2"):
            p = np.load(f"{rd}/B-{i:03d}/pose_{s}.npy").astype(np.int64)[:, NPRE:]
            xs.append(p)
            if not gt or len(gt) < 2 * len(ids):
                g = np.load(f"{rd}/B-{i:03d}/gt_pose_{s}.npy").astype(np.int64)[:, NPRE:p.shape[1] + NPRE]
                gt.append(g) if len(gt) < 2 * len(ids) else None
    out[name] = stats(xs)
out["real"] = stats(gt)
print(json.dumps(out))
