"""Is Ashley (TyypZEov) also in the pool under another account? Account centroids from allvoices/emb_*.npz (one segment per call lane, 3 for
Ashley's); rank every other account by centroid cosine to Ashley's; for the top hits also the segment-level cosines; reference = Ashley's own
segments vs her centroid, and every account's nearest-other-account cosine -> allvoices/search.json"""
import glob, json
import numpy as np
D = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/actors/allvoices"
ASH = "TyypZEov3LSf5Gp8N6kSnBTZYXy2"
E, M = [], []
for f in sorted(glob.glob(f"{D}/emb_*.npz")):
    z = np.load(f); E.append(z["E"]); M += json.loads(str(z["meta"]))
E = np.concatenate(E)
by = {}
for i, m in enumerate(M):
    by.setdefault(m["acc"], []).append(i)
def centroid(ix):
    X = E[ix]; c = X.mean(0); c /= np.linalg.norm(c)
    for _ in range(3):
        s = X @ c; keep = s >= np.quantile(s, 0.3) if len(ix) > 3 else np.ones(len(ix), bool)
        c = X[keep].mean(0); c /= np.linalg.norm(c)
    return c
accs = list(by)                       # every account with at least one clean segment
C = np.stack([centroid(by[a]) for a in accs]); ai = {a: i for i, a in enumerate(accs)}
ca = C[ai[ASH]]
own = E[by[ASH]] @ ca
sims = C @ ca
order = [accs[i] for i in np.argsort(-sims) if accs[i] != ASH]
S = C @ C.T; np.fill_diagonal(S, -1)
nn = S.max(1)
top = []
for a in order[:10]:
    seg = E[by[a]] @ ca
    top.append({"acc": a, "centroid_cos": round(float(sims[ai[a]]), 3), "n_seg": len(by[a]), "seg_mean": round(float(seg.mean()), 3),
                "seg_max": round(float(seg.max()), 3), "seg_frac_ge_p10_own": round(float((seg >= np.quantile(own, 0.1)).mean()), 2)})
res = {"n_segments": len(M), "n_accounts": len(accs), "ashley_segments": len(by[ASH]),
       "ashley_own": {"median": round(float(np.median(own)), 3), "p10": round(float(np.quantile(own, 0.1)), 3), "min": round(float(own.min()), 3)},
       "nn_all": {"median": round(float(np.median(nn)), 3), "p99": round(float(np.quantile(nn, 0.99)), 3)}, "top": top}
json.dump(res, open(f"{D}/search.json", "w"), indent=1)
print(json.dumps(res, indent=1))
