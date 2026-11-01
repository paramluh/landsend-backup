"""TF17850 (GT-forced, 12 s) k150 no-CFG / k50 CFG2 / k50 CFG4 vs 17850 BEST (xs17850_j4, free-running 20 s) and the GT: per row pose cb0 uniques,
pose light-codebook diversity, pose motion = mean |frame diff| of the decoded pose latents (vdata) over the generated part; GT motion over the
same frames from the package's gt/pose_latents_electron_v3 (frames 2..12 s)."""
import json, os, statistics as st, sys, numpy as np
from safetensors.torch import load_file
sys.path.insert(0, "/data/home/amartyavarma/ppm/ppm-pose-server/code")
import srv_common as SC
E = "/data/home/amartyavarma/video/studio-posemono/pose-ft-samples/s20/control"; R = "/data/home/amartyavarma/ppm/ppm-pose-server/requests"
TAGS = ("xs17850_j4", "xs17850_tf_k150", "xs17850_tf_k50c2", "xs17850_tf_k50c4")


def motion(x, frac0, frac1=1.0):
    x = x.reshape(x.shape[0], -1)
    g = x[int(round(x.shape[0] * frac0)):int(round(x.shape[0] * frac1))]
    return float(np.abs(np.diff(g, axis=0)).mean()) if len(g) > 1 else None


res, gt = {}, {}
for t in TAGS:
    rows = {}
    for row, (rid, k) in json.load(open(f"{E}/{t}.json")).items():
        sd = f"{R}/{rid}/s{k}"
        if not os.path.exists(f"{sd}/meta.json"):
            continue
        q = json.load(open(f"{R}/{rid}/request.json"))
        z = np.load(f"{sd}/codes.npz"); P = int(z["prefill_frames"]); pl = z["pose"][1:, P:]
        lp = f"{sd}/vdata/pose_latents_electron_v3.safetensors"
        mot = motion(load_file(lp)["latents"].float().numpy(), P / z["pose"].shape[1]) if os.path.exists(lp) else None
        rows[row] = dict(cb0u=len(np.unique(z["pose"][0, P:])) / pl.shape[1] * 100, pose_light_u=float(np.mean([len(np.unique(c)) for c in pl])) / pl.shape[1] * 100,
                         motion=mot, vdone=os.path.exists(f"{sd}/VIDEO_DONE"))
        if row not in gt:
            ci = SC.clip(q["clip"])
            g = load_file(f"{ci['prompt_dir']}/gt/pose_latents_electron_v3.safetensors")["latents"].float().numpy()
            gt[row] = motion(g, 25 / 150)
    res[t] = rows
print("per-100-frame uniques (cb0, light mean) and motion (mean |d latent|); GT motion over frames 25..149 of the package:")
print("GT", "motion", round(st.mean([v for v in gt.values() if v is not None]), 4))
for t in TAGS:
    v = list(res[t].values())
    print(t, len(v), "videos", sum(x["vdone"] for x in v), {k: round(st.mean([x[k] for x in v if x.get(k) is not None]), 4) for k in ("cb0u", "pose_light_u", "motion") if any(x.get(k) is not None for x in v)})
for row in res[TAGS[1]]:
    print(row, "GT", None if gt.get(row) is None else round(gt[row], 4), " ".join(f"{t.split('_')[-1]} {res[t][row]['motion']:.4f}" if res[t].get(row, {}).get("motion") is not None else f"{t.split('_')[-1]} -" for t in TAGS))
