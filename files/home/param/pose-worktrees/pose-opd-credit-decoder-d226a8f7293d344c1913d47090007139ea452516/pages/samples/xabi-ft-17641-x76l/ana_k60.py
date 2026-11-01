"""K60 (17,850 free rollouts, k60/60, audio k60, text T1 top-k 60 min-p 0.05) vs 17,850 BEST (xs17850_j4) and no-CFG k50/k150 (xs17850_n150), all
free-running 2 + 18 s, and the GT: per row pose cb0 uniques, pose light-codebook diversity (per 100 frames), generated words, speech silence, pose
motion = mean |frame diff| of the decoded pose latents (vdata) over 2..12 s (= the GT window, package gt/pose_latents_electron_v3 frames 25..149)
and over 2..20 s."""
import json, os, statistics as st, sys, numpy as np
from safetensors.torch import load_file
sys.path.insert(0, "/data/home/amartyavarma/ppm/ppm-pose-server/code")
import srv_common as SC
E = "/data/home/amartyavarma/video/studio-posemono/pose-ft-samples/s20/control"; R = "/data/home/amartyavarma/ppm/ppm-pose-server/requests"
TAGS = ("xs17850_k60l180p180", "xs17850_k60l180", "xs17850_k60", "xs17850_j4", "xs17850_n150")


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
        q = json.load(open(f"{R}/{rid}/request.json")); me = json.load(open(f"{sd}/meta.json"))
        z = np.load(f"{sd}/codes.npz"); P = int(z["prefill_frames"]); N = z["pose"].shape[1]; pl = z["pose"][1:, P:]
        lp = f"{sd}/vdata/pose_latents_electron_v3.safetensors"
        x = load_file(lp)["latents"].float().numpy() if os.path.exists(lp) else None
        rows[row] = dict(cb0u=len(np.unique(z["pose"][0, P:])) / pl.shape[1] * 100, pose_light_u=float(np.mean([len(np.unique(c)) for c in pl])) / pl.shape[1] * 100,
                         words=me["gen_words"], silent=me["audio"]["silent_frac_100ms"],
                         mot12=None if x is None else motion(x, P / N, 150 / N), mot20=None if x is None else motion(x, P / N),
                         vdone=os.path.exists(f"{sd}/VIDEO_DONE"))
        if row not in gt:
            ci = SC.clip(q["clip"])
            g = load_file(f"{ci['prompt_dir']}/gt/pose_latents_electron_v3.safetensors")["latents"].float().numpy()
            gt[row] = motion(g, 25 / 150)
    res[t] = rows
print("per-100-frame uniques (cb0, light mean), words, silent frac, motion (mean |d latent|) 2..12 s and 2..20 s; GT motion over frames 25..149:")
print("GT", "mot12", round(st.mean([v for v in gt.values() if v is not None]), 4))
for t in TAGS:
    v = list(res[t].values())
    print(t, len(v), "videos", sum(x["vdone"] for x in v), {k: round(st.mean([x[k] for x in v if x.get(k) is not None]), 4)
                                                          for k in ("cb0u", "pose_light_u", "words", "silent", "mot12", "mot20") if any(x.get(k) is not None for x in v)})
for row in res[TAGS[1]]:
    print(row, "GT", None if gt.get(row) is None else round(gt[row], 4), " ".join(f"{t.split('_')[-1]} {res[t][row]['mot12']:.4f}" if res[t].get(row, {}).get("mot12") is not None else f"{t.split('_')[-1]} -" for t in TAGS))
