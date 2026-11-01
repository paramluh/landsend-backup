"""Join per-second rendered-eye metrics (path/eye_per_sec.json) with per-second head-pose features from the video-path decode
(geom/kp*.npz, render/vdata/kp*.npz). Writes join/bins.json, join/results.json, join/scatter.png."""
import json

import numpy as np
from scipy.stats import spearmanr

G = "/data/home/amartyavarma/ppm/gaze-drift"
OUT = f"{G}/join"
FPS, PF = 25, 50
CLIP = {"S-05": "10_27286bdd", "S-13": "06_74807fe0", "S-15": "10_27286bdd", "S-20": "19_47473a46", "E-03": "03_b437950e",
        "E-05": "05_e25c6859", "E-09": "09_d9ef3c94", "E-12": "12_72cb45d3", "E-13": "13_956c09af", "E-15": "15_d1e1322c",
        "E-18": "18_1ffa9083"}
EXCL = {"S-05", "S-15"}
UNSEEN = {"S-20", "E-15", "E-18"}
z1, z3 = np.load(f"{G}/geom/kp.npz"), np.load(f"{G}/geom/kp3.npz")
eyes = json.load(open(f"{G}/path/eye_per_sec.json"))["rows"]


def kp_for(key):
    s, r = key.split("/")
    if s == "GT":
        return z1[f"gtpkg__{CLIP[r]}"][:300]
    if s in ("fs", "d1", "d2", "d3"):
        return z1[f"{s}__{r}"]
    if s == "l500":
        return z3[f"gz_l500__{r}"]
    e = s[1:]
    if e in ("E0", "E1", "E2", "E3"):
        return np.load(f"{G}/render/vdata/kp-{r}.npz")[e]
    if e in ("E4", "E9"):
        return np.load(f"{G}/render/vdata/kp-{r}.npz")["E0"][:300]
    if e in ("E5", "E6"):
        return z1[f"gtpkg__{CLIP[r]}"][:300]
    if e in ("E7", "E8"):
        return np.load(f"{G}/render/vdata/kpgt-{r}.npz")[e]
    return None


def frame_feats(kp):
    xy = kp[..., :2].astype(np.float64)
    nose, emid, amid = xy[:, 0], (xy[:, 1] + xy[:, 2]) / 2, (xy[:, 3] + xy[:, 4]) / 2
    sh = (xy[:, 5] + xy[:, 6]) / 2
    ed, ad, sw = (np.linalg.norm(xy[:, 1] - xy[:, 2], axis=-1), np.linalg.norm(xy[:, 3] - xy[:, 4], axis=-1),
                  np.linalg.norm(xy[:, 5] - xy[:, 6], axis=-1))
    F = {"pitch_ne": (nose[:, 1] - emid[:, 1]) / ed,          # + = nose further below eyes = head pitched DOWN
         "pitch_ee": (emid[:, 1] - amid[:, 1]) / ed,          # + = eyes below ears = head pitched DOWN
         "yaw": (nose[:, 0] - amid[:, 0]) / ad,
         "height": (sh[:, 1] - nose[:, 1]) / sw,              # + = nose higher above shoulder midpoint (shoulder widths)
         "eye_height": (sh[:, 1] - emid[:, 1]) / sw}
    rel = (xy[:, :5] - sh[:, None]) / np.median(ed[:PF])
    sp = np.r_[np.nan, np.linalg.norm(np.diff(rel, axis=0), axis=-1).mean(1)]   # head speed rel. shoulders, eye-dists/frame
    F["speed"] = sp
    F["abs_speed"] = np.r_[np.nan, np.linalg.norm(np.diff(xy[:, 0], axis=0), axis=-1) / np.median(ed[:PF])]
    return F


REF = {}   # GT prefill means per clip
for r, c in CLIP.items():
    f = frame_feats(z1[f"gtpkg__{c}"][:300])
    REF[r] = {k: float(np.nanmean(v[:PF])) for k, v in f.items()}

ABSF = ["pitch_ne", "pitch_ee", "yaw", "height", "eye_height", "speed", "abs_speed"]
rows = []
for key, e in eyes.items():
    s, r = key.split("/")
    if r in EXCL:
        continue
    kp = kp_for(key)
    if kp is None:
        continue
    f = frame_feats(kp)
    nb = min(len(e["iy_rel"]), len(kp) // FPS)
    for b in range(nb):
        sl = slice(b * FPS, (b + 1) * FPS)
        d = {"set": s, "row": r, "key": key, "bin": b, "gen": s not in ("GT",) and s not in ("RE5", "RE6", "RE7", "RE8"),
             "iy": e["iy_rel"][b], "dc": e["down_or_closed"][b], "open": e["open_rel"][b], "oncam": e["oncam_geo"][b],
             "ix": e["ix_rel"][b], "tracked": e["tracked"][b]}
        for k in ABSF:
            d[k] = float(np.nanmean(f[k][sl]))
        for k in ["pitch_ne", "pitch_ee", "yaw", "height", "eye_height"]:
            d[k + "_rel"] = d[k] - REF[r][k]
        d["absyaw_rel"] = abs(d["yaw_rel"])
        d["speed_rel"] = d["speed"] / REF[r]["speed"]
        rows.append(d)
json.dump(rows, open(f"{OUT}/bins.json", "w"))

MAIN = {"GT", "fs", "d1", "d2", "d3", "l500"}
FEATS = ["pitch_ne_rel", "pitch_ee_rel", "yaw_rel", "absyaw_rel", "speed", "speed_rel", "abs_speed", "height_rel", "eye_height_rel",
         "pitch_ne", "pitch_ee", "height", "eye_height"]


def ok(d):
    return d["bin"] >= 2 and d["tracked"] is not None and d["tracked"] >= 0.5 and d["iy"] is not None and d["dc"] is not None


def sel(sets=MAIN, drop18=False, rows_=None):
    return [d for d in rows if d["set"] in sets and ok(d) and not (drop18 and d["row"] == "E-18") and (rows_ is None or d["row"] in rows_)]


def within_spear(D, x, y):
    """rank x, y within each series, centre, pool -> Pearson (= pooled within-sample Spearman); plus median per-series rho."""
    by = {}
    for d in D:
        by.setdefault(d["key"], []).append(d)
    X, Y, per = [], [], []
    for k, L in by.items():
        if len(L) < 5:
            continue
        a = np.array([d[x] for d in L]); b = np.array([d[y] for d in L])
        if np.std(a) == 0 or np.std(b) == 0:
            continue
        ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
        X += list((ra - ra.mean()) / len(L)); Y += list((rb - rb.mean()) / len(L))
        per.append(spearmanr(a, b)[0])
    return float(np.corrcoef(X, Y)[0, 1]), float(np.nanmedian(per)), len(per)


def between_spear(D, x, y, a=2, b=12):
    by = {}
    for d in D:
        if a <= d["bin"] < b:
            by.setdefault(d["key"], []).append(d)
    by = {k: L for k, L in by.items() if L}
    mx = [np.mean([d[x] for d in L]) for L in by.values()]
    my = [np.mean([d[y] for d in L]) for L in by.values()]
    return float(spearmanr(mx, my)[0]), len(mx)


def zs(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / v.std()


def ols(D, y, xs, demean=False):
    Y = np.array([d[y] for d in D], float)
    X = np.stack([np.array([d[x] for d in D], float) for x in xs], 1)
    if demean:
        keys = np.array([d["key"] for d in D])
        for k in set(keys):
            m = keys == k
            X[m] -= X[m].mean(0); Y[m] -= Y[m].mean()
    X = (X - X.mean(0)) / X.std(0); Yz = (Y - Y.mean()) / Y.std()
    A = np.c_[np.ones(len(Y)), X]
    beta, *_ = np.linalg.lstsq(A, Yz, rcond=None)
    res = Yz - A @ beta
    s2 = res @ res / (len(Y) - A.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(A.T @ A)))
    r2 = 1 - res @ res / (Yz @ Yz)
    # unique R2 per feature (drop-one)
    uniq = {}
    for i, x in enumerate(xs):
        A2 = np.delete(A, i + 1, 1)
        b2, *_ = np.linalg.lstsq(A2, Yz, rcond=None)
        r2b = 1 - ((Yz - A2 @ b2) ** 2).sum() / (Yz @ Yz)
        uniq[x] = round(float(r2 - r2b), 3)
    return {"n": len(Y), "R2": round(float(r2), 3), "beta": {x: round(float(beta[i + 1]), 3) for i, x in enumerate(xs)},
            "t": {x: round(float(beta[i + 1] / se[i + 1]), 1) for i, x in enumerate(xs)}, "uniqueR2": uniq}


def logit(D, xs, thr=0.5):
    Y = np.array([d["dc"] > thr for d in D], float)
    X = np.stack([zs([d[x] for d in D]) for x in xs], 1)
    A = np.c_[np.ones(len(Y)), X]
    w = np.zeros(A.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-A @ w)); W = p * (1 - p) + 1e-9
        H = A.T @ (A * W[:, None]) + 1e-6 * np.eye(len(w))
        w = w + np.linalg.solve(H, A.T @ (Y - p))
    p = 1 / (1 + np.exp(-A @ w))
    se = np.sqrt(np.diag(np.linalg.inv(A.T @ (A * (p * (1 - p))[:, None]))))
    ll = (Y * np.log(p + 1e-12) + (1 - Y) * np.log(1 - p + 1e-12)).sum()
    p0 = Y.mean(); ll0 = len(Y) * (p0 * np.log(p0) + (1 - p0) * np.log(1 - p0))
    return {"n": len(Y), "pos_frac": round(float(p0), 3), "mcfaddenR2": round(float(1 - ll / ll0), 3),
            "coef_z": {x: round(float(w[i + 1]), 2) for i, x in enumerate(xs)}, "z": {x: round(float(w[i + 1] / se[i + 1]), 1) for i, x in enumerate(xs)}}


R = {}
# 1) per-set signed pitch shift vs GT (rows excl S-05/S-15; GT = 2-12 s)
def win_mean(key, k, a, b):
    v = [d[k] for d in rows if d["key"] == key and a <= d["bin"] < b and d[k] is not None and (d["tracked"] or 0) >= 0.5]
    return float(np.mean(v)) if v else None


SHIFT = {}
for k in ["pitch_ne_rel", "pitch_ee_rel", "height_rel", "speed_rel", "absyaw_rel", "dc", "iy"]:
    SHIFT[k] = {}
    for s in ["GT", "fs", "d1", "d2", "d3", "l500"]:
        for (a, b) in [(2, 12), (12, 20)]:
            for grp, rs in [("all9", [r for r in CLIP if r not in EXCL]), ("unseen", sorted(UNSEEN)), ("unseen_no18", ["S-20", "E-15"])]:
                vals = []
                for r in rs:
                    g = win_mean(f"GT/{r}", k, 2, 12); x = win_mean(f"{s}/{r}", k, a, b)
                    if g is not None and x is not None:
                        vals.append(x - g if k != "speed_rel" else x / g)
                SHIFT[k][f"{s}|{a}-{b}|{grp}"] = round(float(np.mean(vals)), 3) if vals else None
R["shift_vs_GT"] = SHIFT
# raw per-row 2-12 means
R["per_row_2_12"] = {key: {k: round(win_mean(key, k, 2, 12), 3) for k in ["pitch_ne_rel", "pitch_ee_rel", "height_rel", "speed_rel", "yaw_rel", "dc", "iy"]}
                     for key in sorted({d["key"] for d in rows}) if win_mean(key, "iy", 2, 12) is not None}
R["per_row_12_20"] = {key: {k: round(win_mean(key, k, 12, 20), 3) for k in ["pitch_ne_rel", "pitch_ee_rel", "height_rel", "speed_rel", "yaw_rel", "dc", "iy"]}
                      for key in sorted({d["key"] for d in rows}) if win_mean(key, "iy", 12, 20) is not None}

# 2) correlations
COR = {}
for tag, D in [("main", sel()), ("main_no18", sel(drop18=True)), ("main_noE15", sel(rows_=[r for r in CLIP if r not in EXCL | {"E-15"}])),
               ("render", sel(sets={d["set"] for d in rows if d["set"].startswith("R")}))]:
    COR[tag] = {}
    for y in ["dc", "iy"]:
        for x in FEATS:
            a = np.array([d[x] for d in D]); b = np.array([d[y] for d in D])
            pooled = spearmanr(a, b)[0]
            w, wmed, nser = within_spear(D, x, y)
            bt, nb = between_spear(D, x, y)
            COR[tag][f"{y}~{x}"] = {"pooled": round(float(pooled), 3), "within_pooled": round(w, 3), "within_median": round(wmed, 3),
                                    "between_2_12": round(bt, 3), "n_bins": len(D), "n_series": nser}
R["spearman"] = COR

# 3) models
XS = ["pitch_ne_rel", "speed", "height_rel"]
XS2 = ["pitch_ne_rel", "pitch_ee_rel", "speed", "height_rel", "absyaw_rel"]
MOD = {}
for tag, D in [("main", sel()), ("main_no18", sel(drop18=True)), ("main_noE15", sel(rows_=[r for r in CLIP if r not in EXCL | {"E-15"}]))]:
    MOD[tag] = {"ols_dc": ols(D, "dc", XS), "ols_iy": ols(D, "iy", XS), "ols_dc_within": ols(D, "dc", XS, demean=True),
                "ols_iy_within": ols(D, "iy", XS, demean=True), "logit_dc": logit(D, XS), "ols_iy_5": ols(D, "iy", XS2),
                "ols_dc_5": ols(D, "dc", XS2)}
R["models"] = MOD

# 4) case series per second
CASE = {}
for key in ["GT/E-15", "fs/E-15", "d1/E-15", "d2/E-15", "d3/E-15", "l500/E-15", "GT/S-13", "fs/S-13", "d3/S-13", "l500/S-13"]:
    L = sorted([d for d in rows if d["key"] == key], key=lambda d: d["bin"])
    CASE[key] = {k: [round(d[k], 3) if d[k] is not None else None for d in L] for k in ["iy", "dc", "pitch_ne_rel", "pitch_ee_rel", "height_rel", "speed_rel", "yaw_rel"]}
R["cases"] = CASE
json.dump(R, open(f"{OUT}/results.json", "w"), indent=1)

# 5) scatter: iy vs best feature (chosen by |pooled| on main)
best = max(FEATS, key=lambda x: abs(COR["main"][f"iy~{x}"]["pooled"]) if x.endswith("_rel") or x.startswith("speed") else 0)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
D = sel()
fig, ax = plt.subplots(figsize=(7, 5), dpi=110)
col = {"GT": "#222222", "fs": "#1f77b4", "d1": "#2ca02c", "d2": "#ff7f0e", "d3": "#d62728", "l500": "#9467bd"}
for s in ["fs", "d1", "d2", "d3", "l500", "GT"]:
    P = [d for d in D if d["set"] == s]
    ax.scatter([d[best] for d in P], [d["iy"] for d in P], s=10 if s != "GT" else 18, alpha=0.55 if s != "GT" else 0.9,
               c=col[s], label=f"{s} (n={len(P)})", edgecolors="none")
E = [d for d in D if d["row"] == "E-15" and d["set"] != "GT"]
ax.scatter([d[best] for d in E], [d["iy"] for d in E], s=30, facecolors="none", edgecolors="k", lw=0.6, label="E-15 generated")
ax.axhline(0, c="grey", lw=0.5); ax.axvline(0, c="grey", lw=0.5)
c = COR["main"][f"iy~{best}"]
ax.set_xlabel(f"{best} (rel. clip GT prefill mean; + = head pitched down)" if "pitch" in best else best)
ax.set_ylabel("vertical iris in render, per 1-s bin (+ = down)")
ax.set_title(f"iris-y vs {best}: pooled rho {c['pooled']:+.2f}, within-sample {c['within_pooled']:+.2f}\n9 rows (excl. S-05/S-15), bins 2-20 s, tracked>=0.5", fontsize=9)
ax.legend(fontsize=7, loc="best")
fig.tight_layout(); fig.savefig(f"{OUT}/scatter.png")
print("best", best)
print(json.dumps({k: R["spearman"]["main"][k] for k in R["spearman"]["main"]}, indent=0)[:3000])
