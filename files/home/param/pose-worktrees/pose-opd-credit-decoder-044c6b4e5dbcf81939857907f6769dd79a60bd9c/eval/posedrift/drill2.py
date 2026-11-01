import json, sys, math, numpy as np
sys.path.insert(0, "/data/home/amartyavarma/ppm/aloe-posedup-live/posedrift")
from analyze import load
rs, real = load()
dup = [r for r in rs if r["model"] == "duplex-498" and r["gen_s"] >= 20]
mono = [r for r in rs if r["model"] == "monoplex-357" and r["gen_s"] >= 20]
aloe = [r for r in real if r["kind"] == "aloe"]
grp = lambda r: "live" if r["client"] in ("browser", "playground") else "scripted"
M = ("body_conf", "head_roll", "below_floor", "shear", "lean", "pitch")
print("TIME CURVES (10 s bins; mean over calls); cols:", M)
def curve(sub, name, tmax=130):
    print(name, "n_calls", len(sub))
    for t0 in range(0, tmax, 20):
        B = [b for r in sub for b in r["bins"] if t0 <= b["t0_s"] < t0 + 20]
        if len(B) < 3: continue
        print(f"  t{t0:3d}-{t0+20:3d} nb={len(B):3d} " + " ".join(f"{m}={np.mean([b[m] for b in B]):.3f}" for m in M))
curve(aloe, "REAL aloe")
for g in ("live", "scripted"):
    for c in (0, 1):
        curve([r for r in dup if grp(r) == g and r["cfg"] == c], f"DUP {g} cfg{c}")
curve([r for r in mono if r["cfg"] == 0], "MONO cfg0"); curve([r for r in mono if r["cfg"] == 1], "MONO cfg1")
# per-bin OLS with cluster-robust SE (by call)
def ols(sub, y):
    rows, Y, G = [], [], []
    for gi, r in enumerate(sub):
        for b in r["bins"]:
            t = (b["t0_s"] + 5) / 60
            rows.append([1, r["cfg"], t, r["cfg"] * t, math.log2(r["topk"][0]), math.log2(r["topk"][1]), float(grp(r) == "live")]); Y.append(b[y]); G.append(gi)
    X, Y, G = np.asarray(rows, float), np.asarray(Y, float), np.asarray(G)
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]; e = Y - X @ beta
    XtXi = np.linalg.pinv(X.T @ X)
    meat = sum(np.outer(X[G == g].T @ e[G == g], X[G == g].T @ e[G == g]) for g in np.unique(G))
    se = np.sqrt(np.diag(XtXi @ meat @ XtXi))
    names = ["const", "cfg", "t_min", "cfg_x_t", "log2_topk", "log2_light", "live"]
    return " ".join(f"{n}={b:+.4f}({s:.4f})" for n, b, s in zip(names, beta, se)) + f" nbins={len(Y)}"
print("PER-BIN OLS (duplex, cluster SE by call): y ~ cfg + t + cfg*t + log2 topk + log2 light + live")
for y in M: print(" ", y, ols(dup, y))
print("(34,53) cfg1 + b62a/7b5a per-call below_floor & body bins:")
for r in dup:
    if r["topk"] == (34, 53):
        print(" ", r["id"], r["client"], r["cfg"], r["end_s"], "floor", [round(b["below_floor"], 2) for b in r["bins"]], "hand", [round(b["hand_conf"], 2) for b in r["bins"]])
print("mono gen_s", np.percentile([r["gen_s"] for r in mono], [10, 50, 90]))
