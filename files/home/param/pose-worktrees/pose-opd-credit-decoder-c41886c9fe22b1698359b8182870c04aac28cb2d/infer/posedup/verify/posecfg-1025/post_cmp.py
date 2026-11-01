"""post-deploy comparison of the public-URL test sessions (session.npz / session.json / result.json per tag)."""
import json, sys
import numpy as np
O = "/data/home/amartyavarma/ppm/aloe-posedup-live/verify/posecfg-1025"


def load(t):
    return np.load(f"{O}/{t}/session.npz"), json.load(open(f"{O}/{t}/session.json")), json.load(open(f"{O}/{t}/result.json"))


def cmp(a, b):
    r = {}
    for k in ("tok", "pose", "mimi", "b"):
        x, y = a[k], b[k]
        n = min(len(x), len(y))
        d = np.flatnonzero((x[:n] != y[:n]).reshape(n, -1).any(1))
        r[k] = {"n": [len(x), len(y)], "identical": bool(len(x) == len(y) and not len(d)), "first_diff": int(d[0]) if len(d) else None, "n_diff": int(len(d))}
    return r


ref, refj, refr = load("ref-old-1")
out = {}
for t in sys.argv[1:]:
    a, j, r = load(t)
    pcf = a["pcf"] if "pcf" in a.files else None
    out[t] = {"vs_ref_old": cmp(a, ref), "latency": j["latency_ms"], "frames": r["frames"], "rtf": r["rtf"], "lag_max": r["lag_max"],
              "pcf_frames": int(pcf.sum()) if pcf is not None else None, "first_pcf_frame": int(np.flatnonzero(pcf)[0]) if pcf is not None and pcf.any() else None,
              "param_log": j.get("param_log"), "pose": r["pose"], "words": r["words"][:160]}
print(json.dumps(out, indent=1))
json.dump(out, open(f"{O}/post_cmp.json", "w"), indent=1)
