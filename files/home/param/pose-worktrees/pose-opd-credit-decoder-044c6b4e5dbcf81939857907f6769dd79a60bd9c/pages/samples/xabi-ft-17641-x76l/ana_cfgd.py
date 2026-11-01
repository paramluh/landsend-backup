"""X76L vs X76 per row: words, speech stats, pose cb0 uniques, light-codebook diversity (pose / audio cb1-31, generated frames),
pose motion = mean |frame diff| of the decoded pose latents (vdata, generated part)."""
import json, os, statistics as st, numpy as np
from safetensors.torch import load_file
E = "/data/home/amartyavarma/video/studio-posemono/pose-ft-samples/s20/control"; R = "/data/home/amartyavarma/ppm/ppm-pose-server/requests"
res = {}
for t in ("xs17641_fs", "xs17641_d1", "xs17641_d2", "xs17641_d3"):
    m = json.load(open(f"{E}/{t}.json")); rows = {}
    for row, (rid, k) in m.items():
        sd = f"{R}/{rid}/s{k}"
        if not os.path.exists(f"{sd}/meta.json"):
            continue
        me = json.load(open(f"{sd}/meta.json")); z = np.load(f"{sd}/codes.npz"); P = int(z["prefill_frames"])
        pl = z["pose"][1:, P:]; al = z["audio"][1:, P:]
        mot = None
        lp = f"{sd}/vdata/pose_latents_electron_v3.safetensors"
        if os.path.exists(lp):
            L = load_file(lp); x = L["latents"].float().numpy(); x = x.reshape(x.shape[0], -1)
            n0 = int(round(x.shape[0] * P / z["pose"].shape[1]))
            g = x[n0:]
            mot = float(np.abs(np.diff(g, axis=0)).mean()) if len(g) > 1 else None
        rows[row] = dict(words=me["gen_words"], cb0u=me["pose_gen_cb0_unique"], silent=me["audio"]["silent_frac_100ms"], dbfs=me["audio"]["rms_dbfs"],
                         pose_light_u=float(np.mean([len(np.unique(c)) for c in pl])), audio_light_u=float(np.mean([len(np.unique(c)) for c in al])),
                         motion=mot, text=me["gen_text"], vdone=os.path.exists(f"{sd}/VIDEO_DONE"))
    res[t] = rows
for t in ("xs17641_fs", "xs17641_d1", "xs17641_d2", "xs17641_d3"):
    v = list(res[t].values())
    keys = [k for k in v[0] if isinstance(v[0][k], (int, float))]
    print(t, len(v), {k: round(st.mean([x[k] for x in v if x.get(k) is not None]), 3) for k in keys})
