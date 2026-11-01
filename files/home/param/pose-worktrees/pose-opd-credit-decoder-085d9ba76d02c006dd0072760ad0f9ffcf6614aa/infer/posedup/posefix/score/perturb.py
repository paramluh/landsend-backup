"""posefix 10-26 (user hypothesis: codes encode diffs?): decode GT pose codes with ONE token perturbed and measure how far / how long the
decoded skeleton changes vs the clean decode (posev3 tokenizer, causal TCN decoder). Perturbations at token t: (a) all 32 codes replaced by
another frame's codes (t + 40), (b) only light codebooks 1-31 random, (c) only cb0 random. Effect = mean |joint displacement| / sw per 25 fps
frame, frames 2t-4 .. 2t+24; reported per token offset."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import score as S

rng = np.random.default_rng(0)
out = {}
for name, path, person in (("p01_gt", "/data/home/amartyavarma/video/demo-data/prompts/01_ab63d745/gt/pose_tokens.safetensors", "01_ab63d745"),):
    codes, _ = S.load_codes(path)
    codes, _, _ = S.prep_codes(codes)
    f0 = S.frame0_of(person)
    kp_ref, kp0 = S.to_keypoints(S.decode(codes), f0)
    k0 = kp0.numpy(); sw = max(abs(float(k0[5, 0] - k0[6, 0])), 1e-3)
    ref = kp_ref.numpy()[:, S.P.SEL, :2]
    res = {}
    for kind in ("swap_all", "rand_light", "rand_cb0"):
        prof = []
        for t in (30, 60, 90):
            c = codes.copy()
            if kind == "swap_all":
                c[t] = codes[(t + 40) % len(codes)]
            elif kind == "rand_light":
                c[t, 1:] = rng.integers(0, 2048, 31)
            else:
                c[t, 0] = rng.integers(0, 2048)
            kp, _ = S.to_keypoints(S.decode(c), f0)
            d = np.linalg.norm(kp.numpy()[:, S.P.SEL, :2] - ref, axis=-1).mean(-1) / sw      # [T]
            prof.append([float(d[2 * t + k]) if 0 <= 2 * t + k < len(d) else np.nan for k in range(-4, 26)])
        p = np.nanmean(np.asarray(prof), 0)
        res[kind] = {"frames_rel_to_2t": list(range(-4, 26)), "mean_disp_sw": [round(float(x), 4) for x in p],
                     "last_frame_above_0.01sw": int(max([k for k, x in zip(range(-4, 26), p) if x > 0.01], default=-99))}
        print(name, kind, json.dumps(res[kind]), flush=True)
    out[name] = res
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "perturb.json"), "w"), indent=1)
