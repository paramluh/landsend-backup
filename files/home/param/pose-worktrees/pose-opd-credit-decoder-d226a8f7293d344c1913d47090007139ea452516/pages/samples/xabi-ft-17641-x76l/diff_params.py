"""X76L vs X76 resolved-params diff: per row, request.json + engine_params + video settings; ids / timings / workers ignored.
Only pose light top-k and speech light top-k may differ."""
import json, sys
C = "/data/home/amartyavarma/video/studio-posemono/pose-ft-samples/s20/control"
a = json.load(open(f"{C}/xs17641_fs_params.json")); b = json.load(open(f"{C}/xs17641_l250_params.json"))
IGN = {"_submitted", "total_seconds", "worker"}
def flat(o, p=""):
    out = {}
    for k, v in o.items():
        if k in IGN or k in ("request_id", "video_done", "video_fail"):
            continue
        if isinstance(v, dict):
            out.update(flat(v, f"{p}{k}."))
        else:
            out[f"{p}{k}"] = v
    return out
allk = set()
for row in a:
    fa, fb = flat(a[row]), flat(b[row])
    d = {k: (fa.get(k), fb.get(k)) for k in set(fa) | set(fb) if fa.get(k) != fb.get(k)}
    allk |= set(d)
    print(row, sorted(d.items()))
ok = {"request.light_topk", "request.pose_light_topk", "engine_params.light_topk", "engine_params.pose_light_topk"}
print("DIFF KEYS", sorted(allk), "->", "PASS (only the two light top-k values differ)" if allk <= ok else "FAIL")
