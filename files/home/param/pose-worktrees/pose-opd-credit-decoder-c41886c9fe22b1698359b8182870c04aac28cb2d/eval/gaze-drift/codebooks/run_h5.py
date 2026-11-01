"""H5: per-codebook head sensitivity, code distribution stats (GT / prefill / generated), offline token fixes.
python run_h5.py [stage]   stage in {stats, sens, gen, fix, all}; results merged into results.json."""
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, "/data/home/amartyavarma/ppm/gaze-drift/codebooks")
import cb_lib as L  # noqa: E402

torch.set_num_threads(8)
OUT = "/data/home/amartyavarma/ppm/gaze-drift/codebooks"
RJ = f"{OUT}/results.json"
STAGE = sys.argv[1] if len(sys.argv) > 1 else "all"
T0 = time.time()
gt, gen = L.load_all()
RNG = np.random.default_rng(0)


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


def save(key, val):
    import fcntl
    with open(RJ + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        R = json.load(open(RJ)) if os.path.isfile(RJ) else {}
        R[key] = val
        tmp = f"{RJ}.{os.getpid()}.tmp"
        json.dump(R, open(tmp, "w"), indent=1)
        os.replace(tmp, RJ)


def r(x, n=4):
    return float(np.round(x, n))


# ------------------------------------------------------------------ 3. code distribution stats
def stats():
    out = {"note": "matched window = token frames 25..150 for GT and generated (equal N = 11 x 125); full gen = 25..250; "
                   "prefill = token frames 0..25 (equals GT exactly); KL in bits, add-0.5 smoothing over 2048 codes; "
                   "chg = fraction of frames whose code differs from the previous frame"}
    srcs = {"GT": [gt[x][:, 25:150] for x in L.ROWS], "prefill": [gt[x][:, :25] for x in L.ROWS],
            "GT_all": [gt[x] for x in L.ROWS]}
    for s in L.SETS:
        srcs[s] = [gen[s][x][:, 25:150] for x in L.ROWS]
        srcs[s + "_full"] = [gen[s][x][:, 25:250] for x in L.ROWS]
        srcs[s + "_late"] = [gen[s][x][:, 150:250] for x in L.ROWS]
    per = {k: [L.code_stats([a[c] for a in v]) for c in range(32)] for k, v in srcs.items()}
    # same-N noise floor: GT frames 25..87 vs 88..150, and gen likewise
    half = lambda arrs, lo, hi: [a[:, lo:hi] for a in arrs]  # noqa: E731
    for k, v in per.items():
        out[k] = {"H": [r(d["H"], 3) for d in v], "chg": [r(d["chg"], 3) for d in v], "distinct": [d["distinct"] for d in v]}
        if k != "GT":
            out[k]["KL_vs_GT"] = [r(L.kl(v[c]["cnt"], per["GT"][c]["cnt"]), 3) for c in range(32)]
    flo = {}
    for name, arrs in [("GT", [gt[x] for x in L.ROWS])] + [(s, [gen[s][x] for x in L.ROWS]) for s in L.SETS]:
        a = [L.code_stats([q[c, 25:87] for q in arrs]) for c in range(32)]
        b = [L.code_stats([q[c, 88:150] for q in arrs]) for c in range(32)]
        flo[name] = [r(L.kl(a[c]["cnt"], b[c]["cnt"]), 3) for c in range(32)]
    out["KL_split_half_floor"] = flo
    save("stats", out)
    log("stats: chg GT", np.round(out["GT"]["chg"], 2))
    for s in L.SETS:
        log(f"stats: chg {s}", np.round(out[s]["chg"], 2))
    g, f = np.array(out["GT"]["chg"]), np.array(out["fs"]["chg"])
    line = (f"EARLY (H5, {time.strftime('%H:%M')}): pose code-change rate per frame, matched window 2-12 s, mean over cb1..31: "
            f"GT {g[1:].mean():.3f} vs gen fs {f[1:].mean():.3f}, d1 {np.mean(out['d1']['chg'][1:]):.3f}, d3 {np.mean(out['d3']['chg'][1:]):.3f}; "
            f"cb0 GT {g[0]:.3f} vs fs {f[0]:.3f}. Unigram KL(gen fs||GT) mean cb1..31 {np.mean(out['fs']['KL_vs_GT'][1:]):.3f} bits "
            f"(split-half floor GT {np.mean(flo['GT'][1:]):.3f}).\n")
    open(f"{OUT}/EARLY.txt", "w").write(line)
    log(line)


# ------------------------------------------------------------------ 2. sensitivity on GT tokens
def group_delta(fa, fb):
    d = {}
    d["head"] = float(np.mean(np.linalg.norm(fa["head"] - fb["head"], axis=-1)))
    d["head_sw"] = float(np.mean(np.linalg.norm(fa["head_sw"] - fb["head_sw"], axis=-1)))
    d["yaw"] = float(np.mean(np.abs(fa["yaw"] - fb["yaw"])))
    d["body"] = float(np.mean(np.linalg.norm(fa["body"] - fb["body"], axis=-1)))
    d["hands"] = float(np.mean(np.linalg.norm(fa["hands"] - fb["hands"], axis=-1)))
    d["head_jit"] = L.jitter(fa["head"]) - L.jitter(fb["head"])
    d["yaw_jit"] = L.jitter(fa["yaw"]) - L.jitter(fb["yaw"])
    return d


def sens():
    keys = ["head", "head_sw", "yaw", "body", "hands", "head_jit", "yaw_jit"]
    acc = {m: {k: np.zeros((len(L.ROWS), 32)) for k in keys} for m in ("random", "hold", "trunc")}
    for ri, row in enumerate(L.ROWS):
        c = gt[row][None]                                            # [1, 32, F]
        C = L.contribs(c)                                            # [32, 1, D, F]
        base = C.sum(0)
        rc = RNG.integers(0, 2048, size=c.shape)
        Rn = L.contribs(rc)
        Hd = torch.cat([C[..., :1], C[..., :-1]], -1)               # code from 1 token frame earlier
        fb = L.feats(L.positions(L.dec_lat(base)[0]))
        es, ss = fb["es"], fb["ss"]
        for mode in ("random", "hold", "trunc"):
            if mode == "trunc":                                      # keep codebooks 0..k-1 only (k = 1..32)
                cs = torch.cumsum(C[:, 0], 0)                        # [32, D, F]
                z = cs
            else:
                Q = Rn if mode == "random" else Hd
                z = base[None, 0] - C[:, 0] + Q[:, 0]                # [32, D, F]
            r3 = L.dec_lat(z)
            for cb in range(32):
                fa = L.feats(L.positions(r3[cb]), es, ss)
                d = group_delta(fa, fb)
                for k in keys:
                    acc[mode][k][ri, cb] = d[k]
        log("sens", row)
    out = {"note": "GT tokens, 11 rows; per codebook c: random = codes of c replaced by uniform random codes; hold = codes of c shifted "
                   "by one token frame (previous code); trunc[k] = decode with codebooks 0..k only (index k = keep k+1 codebooks). "
                   "head = mean |delta| of nose/eyes/ears rel. to shoulder midpoint in ear-distance units; head_sw / body / hands in "
                   "shoulder-width units; yaw = mean |delta yaw proxy|; *_jit = increase of HF jitter (RMS of x - gauss(x, 2 frames))."}
    for m in acc:
        out[m] = {k: [r(v, 5) for v in acc[m][k].mean(0)] for k in keys}
    hr = np.array(out["random"]["head"])
    out["rank_head_random"] = [int(i) for i in np.argsort(-hr)]
    hs = np.array(out["random"]["head_sw"]) / (np.array(out["random"]["hands"]) + 1e-9)
    out["head_vs_hands_ratio_random"] = [r(v, 3) for v in hs]
    save("sens", out)
    log("sens head(random)", np.round(hr, 3))
    log("sens hands(random)", np.round(out["random"]["hands"], 3))
    log("sens head(hold)", np.round(out["hold"]["head"], 3))
    log("trunc head err", np.round(out["trunc"]["head"], 3))


# ------------------------------------------------------------------ 4. head metrics on gen + offline fixes
VWIN = {"prefill": (0, 50), "matched": (50, 300), "late": (300, 500)}


def head_metrics(feat, lo, hi, f_pref):
    h, y = feat["head"][lo:hi], feat["yaw"][lo:hi]
    return {"head_jit": L.jitter(h), "yaw_jit": L.jitter(y), "head_lf": L.lowpass_std(h), "yaw_lf": L.lowpass_std(y),
            "yaw_vel": float(np.mean(np.abs(np.diff(y)))), "head_vel": float(np.mean(np.linalg.norm(np.diff(h, axis=0), axis=-1))),
            "yaw_off": float(abs(np.mean(y[-50:]) - np.mean(f_pref["yaw"][:50])))}


def mode3(a):
    b = a.copy()
    for t in range(1, a.shape[-1] - 1):
        p, q = a[..., t - 1], a[..., t + 1]
        b[..., t] = np.where(p == q, p, a[..., t])
    return b


def hold2(a):
    b = a.copy()
    b[..., 1::2] = a[..., 0:-1:2][..., : b[..., 1::2].shape[-1]]
    return b


def apply_fix(codes, fix, gmode):
    """codes [32, F] -> (latent [1, D, F], codes_after or None, kp_sigma)"""
    c = codes.copy()
    ks = 0.0
    lat_sig, lat_k0 = 0.0, 32
    if fix.startswith("trunc"):
        k = int(fix[5:])
        C = L.contribs(c[None])
        return C[:k].sum(0), c[:k], 0.0
    if fix.startswith("hold2_"):
        k0 = int(fix[6:])
        c[k0:] = hold2(c[k0:])
    elif fix.startswith("mode3_"):
        k0 = int(fix[6:])
        c[k0:] = mode3(c[k0:])
    elif fix.startswith("gtmode"):
        k0 = int(fix[6:])
        c[k0:] = gmode[k0:, None]
    elif fix.startswith("latsm"):                                    # latsm<sigma>_<k0>: gauss over token frames on cb>=k0 latent
        s, k0 = fix[5:].split("_")
        lat_sig, lat_k0 = float(s), int(k0)
    elif fix.startswith("kpsm"):
        ks = float(fix[4:])
    C = L.contribs(c[None])
    z = C.sum(0)
    if lat_sig > 0:
        fine = C[lat_k0:].sum(0).numpy()
        z = C[:lat_k0].sum(0) + torch.from_numpy(gaussian_filter1d(fine, lat_sig, axis=-1, mode="nearest"))
    return z, c, ks


FIXES = ["none", "trunc24", "trunc16", "trunc8", "trunc4", "gtmode16", "gtmode8", "hold2_1", "hold2_8", "hold2_16", "mode3_1",
         "mode3_8", "latsm1_1", "latsm1_8", "latsm2_1", "kpsm1", "kpsm2"]


def fixes(part=None):
    fl = FIXES if part is None else FIXES[part::2]
    base = {row: L.feats(L.positions(L.dec_codes(gt[row][None])[0])) for row in L.ROWS}     # per-clip scales from GT decode
    gm = np.zeros(32, np.int64)
    for c in range(32):
        gm[c] = np.bincount(np.concatenate([gt[x][c] for x in L.ROWS]), minlength=2048).argmax()
    srcs = {"GT": gt}
    srcs.update({s: gen[s] for s in L.SETS})
    key = "fixes" if part is None else f"fixes_{part}"
    res = json.load(open(RJ)).get(key, {}).get("res", {}) if os.path.isfile(RJ) else {}
    for fix in fl:
        if fix in res:
            continue
        res[fix] = {}
        for sname, dd in srcs.items():
            per = {w: [] for w in VWIN}
            chg = []
            for row in L.ROWS:
                codes = dd[row]
                fbase = base[row]
                z, ca, ks = apply_fix(codes, fix, gm)
                pos = L.positions(L.dec_lat(z)[0])
                if ks > 0:
                    pos = gaussian_filter1d(pos, ks, axis=0, mode="nearest")
                ft = L.feats(pos, fbase["es"], fbase["ss"])
                T = pos.shape[0]
                for w, (lo, hi) in VWIN.items():
                    if hi <= T:
                        per[w].append(head_metrics(ft, lo, hi, ft))
                if ca is not None and ca.shape[0] > 1:
                    lo_t, hi_t = (25, 150)
                    chg.append(float(np.mean(ca[1:, lo_t + 1:hi_t] != ca[1:, lo_t:hi_t - 1])))
            res[fix][sname] = {w: {k: r(np.mean([d[k] for d in v]), 4) for k in v[0]} for w, v in per.items() if v}
            res[fix][sname]["chg_light_matched"] = r(np.mean(chg), 4) if chg else None
        log("fix", fix, {s: (res[fix][s]["matched"]["head_jit"], res[fix][s]["matched"]["yaw_lf"]) for s in srcs})
        save("fixes" if part is None else f"fixes_{part}", {"note": "video-frame windows prefill 0-50 (0-2 s), matched 50-300 (2-12 s, GT has 300 frames), late 300-500 "
                               "(12-20 s, gen only). head_jit = RMS of nose/eyes/ears (rel. shoulder mid, ear-dist units) minus its "
                               "gauss(2 frames) = HF jitter; yaw_jit same on yaw proxy; *_lf = std of gauss(12 frames = 0.5 s) "
                               "low-pass = slow drift; yaw_off = |mean yaw last 2 s of window - mean yaw prefill|; "
                               "chg_light_matched = code-change rate cb1..31 token frames 25..150 after the fix.",
                       "gt_mode_codes": gm.tolist(), "res": res})


if STAGE in ("stats", "all"):
    stats()
if STAGE in ("sens", "all"):
    sens()
if STAGE in ("fix", "all"):
    fixes()
if STAGE.startswith("fixp"):
    fixes(int(STAGE[4:]))
log("done", STAGE)
