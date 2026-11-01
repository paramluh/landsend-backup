"""gesturemetric: for live sessions with no prefill metadata, measure the forced pose prefill empirically = longest prefix of session pose
equal to the person's GT clip tokens (avatar-clip prefill) at any start offset; also report for sessions WITH metadata as a control."""
import glob, json, os, sys
import numpy as np
from safetensors.numpy import load_file
L = "/data/home/amartyavarma/ppm/aloe-posedup-live"
PROMPTS = "/data/home/amartyavarma/video/demo-data/prompts"
ds = sorted(glob.glob(L + "/sessions/*")) + sorted(glob.glob(L + "/stage-lb/*/sessions/*"))
out = {}
for d in ds:
    sj = json.load(open(d + "/session.json")) if os.path.exists(d + "/session.json") else {}
    if not sj.get("pose_frames"):
        continue
    meta = sj.get("pose_prefill_frames")
    person = sj.get("person") or "01_ab63d745"
    if person not in os.listdir(PROMPTS):
        person = "01_ab63d745"
    pose = np.load(d + "/session.npz")["pose"].reshape(-1, 32)
    best = (0, None)
    for src, f in (("clip", f"{PROMPTS}/{person}/gt/pose_tokens.safetensors"), ("opening", f"{L}/genweb/openings/{person}/pose_tokens.safetensors")):
        if not os.path.exists(f):
            continue
        gt = load_file(f)["codes"]
        gt = gt.T if gt.shape[0] == 32 else gt
        for off in range(0, len(gt) - 1):
            n = min(60, len(gt) - off, len(pose))
            m = (pose[:n] == gt[off:off + n]).all(1)
            k = int(np.argmin(m)) if not m.all() else len(m)
            if k > best[0]:
                best = (k, f"{src}@{off}")
    # also: cb0 change rate over first 26 vs rest (forced real prefill looks like real motion either way; equality is the test)
    out[os.path.basename(d)] = {"meta": meta, "person": person, "gt_prefix_match": best[0], "gt_offset": best[1]}
    print(os.path.basename(d), meta, best, flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "gen", "_prefill_check.json"), "w"), indent=1)
