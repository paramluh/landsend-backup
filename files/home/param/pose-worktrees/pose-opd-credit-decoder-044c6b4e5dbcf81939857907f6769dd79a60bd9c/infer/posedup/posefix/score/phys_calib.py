#!/usr/bin/env python3
"""posefix 10-26: calibrate phys.py on REAL pose: held-out Aloe tracks (run_baselines.aloe_tracks(), first 3000 tokens = 4 min each,
Person 01's frame) + the 20 demo actors' GT clips (own frame 0). Per-item bands (bone length p0.5 / p99.5, jump / speed / accel / finger /
wrist p99.5, elbow fold p0.5 with long bones), then tau per test = pooled real frame-score q0.998 -> out/phys_calib.json; prints the real
fail % per test and set. Keypoints cached in work/phys_real_k55.npz (float32)."""
import json
import os
import sys

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
import numpy as np  # noqa: E402
import phys as PH  # noqa: E402

CACHE = os.path.join(H, "work", "phys_real_k55.npz")


def real_sets():
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        return list(z["sets"])
    import score as S
    import run_baselines as RB
    sets = []

    def k55_of(codes, person):
        ck, _, _ = S.prep_codes(S._fix_codes(codes))
        kp, kp0 = S.to_keypoints(S.decode(ck, "cpu"), S.frame0_of(person))
        k0 = kp0.numpy()
        return kp.numpy()[:, S.P.SEL].astype(np.float32), max(abs(float(k0[5, 0] - k0[6, 0])), 1e-3)
    for conv, side, path in RB.aloe_tracks()[:24]:
        c, _ = S.load_codes(path)
        k, sw = k55_of(c[:3000], RB.PERSON)
        sets.append(("aloe", f"{conv}:{side}", k, sw))
        print("aloe", conv, side, k.shape, flush=True)
    for d in sorted(os.listdir(S.PROMPTS)):
        p = os.path.join(S.PROMPTS, d, "gt", "pose_tokens.safetensors")
        if os.path.exists(p):
            c, _ = S.load_codes(p)
            k, sw = k55_of(c, d)
            sets.append(("actors", d, k, sw))
    np.savez(CACHE, sets=np.asarray(sets, dtype=object))
    return sets


def q(x, p, axis=0):
    with np.errstate(all="ignore"):
        return np.nanpercentile(x, p, axis=axis)


def main():
    sets = real_sets()
    F = [PH.features(k.astype(np.float64), sw) for _, _, k, sw in sets]
    cat = {key: np.concatenate([f[key] for f in F]) for key in F[0]}
    c = {"len_lo": q(cat["L"], 0.5).tolist(), "len_hi": q(cat["L"], 99.5).tolist(), "jump_hi": q(cat["J"], 99.5).tolist(),
         "vel_hi": q(cat["V"], 99.5).tolist(), "acc_hi": q(cat["A"], 99.5).tolist(),
         "zig_hi": np.maximum(q(cat["zig"], 99.5), 5.0).tolist(), "turn_hi": np.maximum(q(cat["turn"], 99.5), 10.0).tolist(),
         "ua_med": q(cat["ua"], 50).tolist(), "fa_med": q(cat["fa"], 50).tolist(), "wb_hi": q(cat["wb"], 99.5).tolist()}
    longb = (cat["ua"] >= 0.5 * np.asarray(c["ua_med"])) & (cat["fa"] >= 0.5 * np.asarray(c["fa_med"]))
    c["elb_lo"] = np.maximum(q(np.where(longb, cat["elb"], np.nan), 0.5), 1.0).tolist()
    S_ = [PH.scores(f, c) for f in F]
    allS = np.concatenate(S_)
    c["tau"] = {t: max(float(np.quantile(allS[:, i], 0.998)), 1.0) for i, t in enumerate(PH.TESTS)}
    c["about"] = ("phys_calib.py: real = held-out Aloe tracks (24 x 4 min, P01 frame) + 20 actor GT clips; bands per item, tau = pooled "
                  "frame-score q0.998 (>= 1)")
    os.makedirs(os.path.join(H, "out"), exist_ok=True)
    json.dump(c, open(PH.CALIB, "w"), indent=1)
    PH._CAL.clear()
    rep = {}
    for nm in ("aloe", "actors"):
        fl = np.concatenate([s > np.asarray([c["tau"][t] for t in PH.TESTS])[None] for s, (g, _, _, _) in zip(S_, sets) if g == nm])
        rep[nm] = PH.summarize(fl)
    fl = allS > np.asarray([c["tau"][t] for t in PH.TESTS])[None]
    rep["pooled"] = PH.summarize(fl)
    rep["tau"] = c["tau"]
    json.dump(rep, open(PH.CALIB.replace("phys_calib", "phys_real"), "w"), indent=1)
    print(json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
