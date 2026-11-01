#!/usr/bin/env python3
"""posedrift 10-26: group the per-call scores (out/calls.*.jsonl) by recorded sampling settings vs real (out/real.jsonl) -> out/analysis.json
+ printed summary. Per-call metrics are averaged with equal weight per call; 95 % CIs = bootstrap over calls."""
import glob
import json
import math
import os

import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
OUT = f"{H}/out"
M = ("conf", "body_conf", "hand_conf", "below_floor", "red", "phys_any", "lean", "shear", "pitch", "pitch_nose", "head_roll")
RNG = np.random.default_rng(0)


def load():
    rs = [json.loads(l) for f in sorted(glob.glob(f"{OUT}/calls.*.jsonl")) for l in open(f)]
    rs = [r for r in rs if "error" not in r]
    for r in rs:
        p = r["params"]
        r["cfg"] = int(bool(p.get("pose_cfg_on")) and (p.get("pose_cfg_s") or 1) != 1 and bool(p.get("pose_cfg_cb0") or p.get("pose_cfg_light")))
        r["cfg_s"] = p.get("pose_cfg_s") if r["cfg"] else 1.0
        r["topk"] = (p.get("pose_topk"), p.get("pose_light_topk"))
        r["rep"] = (p.get("pose_rep") or 1.0, p.get("pose_rep_light") or 1.0)
        r["pose_log"] = any(any(str(k).startswith("pose_") for k in (e.get("changes") or e.get("params") or e if isinstance(e, dict) else {}))
                            for e in r.get("param_log") or [])
    real = [json.loads(l) for l in open(f"{OUT}/real.jsonl")]
    return rs, real


def ci(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if len(x) == 0:
        return None
    bs = RNG.choice(x, (2000, len(x))).mean(1)
    return [round(float(x.mean()), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4), int(len(x))]


def diff(a, b):
    """mean(a) - mean(b), bootstrap 95 % CI, Cohen's d."""
    a = np.asarray([v for v in a if v is not None and np.isfinite(v)], float)
    b = np.asarray([v for v in b if v is not None and np.isfinite(v)], float)
    if len(a) < 2 or len(b) < 2:
        return None
    bs = RNG.choice(a, (2000, len(a))).mean(1) - RNG.choice(b, (2000, len(b))).mean(1)
    sd = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)) or 1e-9
    return {"diff": round(float(a.mean() - b.mean()), 4), "ci95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
            "d": round(float((a.mean() - b.mean()) / sd), 2), "n": [len(a), len(b)]}


def window(r, k, t_lo, t_hi):
    v = [b[k] for b in r["bins"] if t_lo <= b["t0_s"] < t_hi]
    return float(np.mean(v)) if v else None


def group_table(rs, key, metrics=M, win=None):
    out = {}
    for g in sorted({key(r) for r in rs}, key=str):
        sub = [r for r in rs if key(r) == g]
        out[str(g)] = {m: ci([window(r, m, *win) if win else r[m] for r in sub]) for m in metrics}
        out[str(g)]["n"] = len(sub)
    return out


def curves(rs, key, metrics=("body_conf", "below_floor", "lean", "shear", "pitch", "head_roll"), tmax=180):
    out = {}
    for g in sorted({key(r) for r in rs}, key=str):
        sub = [r for r in rs if key(r) == g]
        cg = {}
        for t0 in range(0, tmax, 10):
            vals = {m: [b[m] for r in sub for b in r["bins"] if t0 <= b["t0_s"] < t0 + 10] for m in metrics}
            n = len(vals[metrics[0]])
            if n >= 2:
                cg[t0] = {"n": n, **{m: round(float(np.mean(v)), 3) for m, v in vals.items()},
                          "lean_p90": round(float(np.percentile(vals["lean"], 90)), 2)}
        out[str(g)] = cg
    return out


def ols(rs, y, cols):
    X = np.asarray([[1.0] + [f(r) for f in cols.values()] for r in rs], float)
    Y = np.asarray([r[y] if not callable(y) else y(r) for r in rs], float)
    ok = np.isfinite(Y) & np.isfinite(X).all(1)
    X, Y = X[ok], Y[ok]
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    res = Y - X @ beta
    XtXi = np.linalg.pinv(X.T @ X)
    meat = (X * res[:, None] ** 2).T @ X                      # HC0 robust SE
    se = np.sqrt(np.diag(XtXi @ meat @ XtXi))
    return {"n": int(ok.sum()), **{name: [round(float(b), 4), round(float(s), 4)] for name, b, s in zip(["const"] + list(cols), beta, se)}}


def main():
    rs, real = load()
    dup = [r for r in rs if r["model"] == "duplex-498" and r["gen_s"] >= 20]
    mono = [r for r in rs if r["model"] == "monoplex-357" and r["gen_s"] >= 20]
    aloe = [r for r in real if r["kind"] == "aloe"]
    actors = [r for r in real if r["kind"] == "actors"]
    A = {"n": {"dup": len(dup), "mono": len(mono), "real_aloe": len(aloe), "real_actors": len(actors)}}
    # reference: real aloe whole track + first 60 s / 60-120 s (to compare like-for-like windows)
    A["real"] = {"aloe_all": {m: ci([r[m] for r in aloe]) for m in M}, "actors_all": {m: ci([r[m] for r in actors]) for m in M},
                 "aloe_0_60": {m: ci([window(r, m, 0, 60) for r in aloe]) for m in M},
                 "aloe_60_120": {m: ci([window(r, m, 60, 120) for r in aloe]) for m in M},
                 "aloe_slope_per_min_first3min": None,
                 "aloe_lean_bin_p99": round(float(np.percentile([b["lean"] for r in aloe for b in r["bins"]], 99)), 2),
                 "aloe_lean_bin_p50": round(float(np.percentile([b["lean"] for r in aloe for b in r["bins"]], 50)), 2),
                 "aloe_shear_bin_p1_p99": [round(float(np.percentile([b["shear"] for r in aloe for b in r["bins"]], q)), 2) for q in (1, 99)],
                 "aloe_bodyconf_bin_p1": round(float(np.percentile([b["body_conf"] for r in aloe for b in r["bins"]], 1)), 3),
                 "aloe_below_floor_bin_p99": round(float(np.percentile([b["below_floor"] for r in aloe for b in r["bins"]], 99)), 3)}
    A["real"]["aloe_slope_per_min"] = {k: ci([r["slope_per_min"].get(k) for r in aloe]) for k in ("conf", "body_conf", "below_floor", "lean", "shear", "pitch")}
    # 1. CFG on vs off, duplex, whole call and matched 0-30 s window
    A["dup_cfg_whole"] = group_table(dup, lambda r: r["cfg"])
    A["dup_cfg_0_30s"] = group_table(dup, lambda r: r["cfg"], win=(0, 30))
    A["dup_cfg_diff_0_30s"] = {m: diff([window(r, m, 0, 30) for r in dup if r["cfg"]], [window(r, m, 0, 30) for r in dup if not r["cfg"]]) for m in M}
    A["dup_cfg_diff_whole"] = {m: diff([r[m] for r in dup if r["cfg"]], [r[m] for r in dup if not r["cfg"]]) for m in M}
    # 2. stratified by top-k pair (only strata with both arms)
    strata = {}
    for tk in sorted({r["topk"] for r in dup}, key=str):
        on = [r for r in dup if r["topk"] == tk and r["cfg"]]
        off = [r for r in dup if r["topk"] == tk and not r["cfg"]]
        strata[str(tk)] = {"n_on": len(on), "n_off": len(off)}
        if on and off:
            strata[str(tk)].update({m: diff([window(r, m, 0, 30) for r in on], [window(r, m, 0, 30) for r in off])
                                    for m in ("conf", "body_conf", "below_floor", "red", "lean", "shear", "pitch", "head_roll")})
    A["dup_cfg_by_topk_0_30s"] = strata
    A["dup_by_topk_cfg_0_30s"] = group_table(dup, lambda r: (r["topk"], r["cfg"], r["cfg_s"]), metrics=("conf", "body_conf", "below_floor", "lean", "shear"), win=(0, 30))
    A["dup_by_client_cfg_0_30s"] = group_table(dup, lambda r: (r["client"], r["cfg"]), metrics=("conf", "body_conf", "below_floor", "lean", "shear"), win=(0, 30))
    # 3. regression (per call, 0-30 s window)
    cols = {"cfg_on": lambda r: r["cfg"], "cfg_s_minus1": lambda r: (r["cfg_s"] or 1) - 1, "log2_topk": lambda r: math.log2(r["topk"][0] or 2048),
            "log2_light_topk": lambda r: math.log2(r["topk"][1] or 2048), "rep_on": lambda r: float(r["rep"] != (1.0, 1.0)),
            "client_test": lambda r: float(r["client"] == "test"), "genweb_offline": lambda r: float(r["src"] == "genweb")}
    A["dup_ols_0_30s"] = {m: ols(dup, lambda r, m=m: window(r, m, 0, 30) if window(r, m, 0, 30) is not None else float("nan"),
                                  {k: v for k, v in cols.items() if k != "cfg_s_minus1"}) for m in ("conf", "body_conf", "below_floor", "lean", "shear")}
    # 4. drift: long calls (>= 60 s generated), slopes + late-minus-early by CFG; time curves
    long_ = [r for r in dup if r["end_s"] >= 60]
    A["dup_long_n"] = {"on": sum(r["cfg"] for r in long_), "off": sum(1 - r["cfg"] for r in long_)}
    A["dup_long_slope"] = {g: {k: ci([r["slope_per_min"].get(k) for r in long_ if r["cfg"] == g]) for k in ("conf", "body_conf", "below_floor", "lean", "shear", "pitch", "head_roll")} for g in (0, 1)}
    A["dup_long_slope_diff"] = {k: diff([r["slope_per_min"].get(k) for r in long_ if r["cfg"]], [r["slope_per_min"].get(k) for r in long_ if not r["cfg"]]) for k in ("conf", "body_conf", "below_floor", "lean", "shear", "pitch")}
    A["dup_long_late_minus_early"] = {g: {k: ci([(r.get("late_minus_early") or {}).get(k) for r in long_ if r["cfg"] == g]) for k in ("conf", "body_conf", "below_floor", "lean", "shear", "pitch", "head_roll")} for g in (0, 1)}
    A["dup_curves_by_cfg"] = curves(dup, lambda r: r["cfg"])
    A["real_curves"] = curves(aloe, lambda r: "aloe", tmax=300)
    lim = A["real"]["aloe_lean_bin_p99"]
    A["dup_long_bad_lean_last_bin"] = {g: [sum(1 for r in long_ if r["cfg"] == g and r["bins"] and r["bins"][-1]["lean"] > lim), sum(1 for r in long_ if r["cfg"] == g)] for g in (0, 1)}
    A["dup_any_bin_bad_lean"] = {g: [sum(1 for r in dup if r["cfg"] == g and any(b["lean"] > lim for b in r["bins"])), sum(1 for r in dup if r["cfg"] == g)] for g in (0, 1)}
    A["dup_bad_lean_calls"] = [{"id": r["id"], "cfg": r["cfg"], "topk": r["topk"], "client": r["client"], "end_s": r["end_s"],
                                "first_bad_t": next(b["t0_s"] for b in r["bins"] if b["lean"] > lim), "max_lean": max(b["lean"] for b in r["bins"])}
                               for r in dup if any(b["lean"] > lim for b in r["bins"])]
    # 5. monoplex (independent model): CFG on vs off
    A["mono_cfg_0_30s"] = group_table(mono, lambda r: r["cfg"], win=(0, 30))
    A["mono_cfg_diff_0_30s"] = {m: diff([window(r, m, 0, 30) for r in mono if r["cfg"]], [window(r, m, 0, 30) for r in mono if not r["cfg"]]) for m in M}
    A["mono_by_topk_cfg_0_30s"] = group_table(mono, lambda r: (r["topk"], r["cfg"], r["cfg_s"]), metrics=("conf", "body_conf", "below_floor", "lean", "shear"), win=(0, 30))
    json.dump(A, open(f"{OUT}/analysis.json", "w"), indent=1)
    for k in ("n", "dup_cfg_diff_0_30s", "dup_cfg_by_topk_0_30s", "dup_ols_0_30s", "dup_long_n", "dup_long_slope_diff", "dup_long_late_minus_early",
              "dup_long_bad_lean_last_bin", "dup_any_bin_bad_lean", "mono_cfg_diff_0_30s"):
        print(k, json.dumps(A[k]), flush=True)
    print("real", json.dumps({k: A["real"][k] for k in ("aloe_lean_bin_p99", "aloe_lean_bin_p50", "aloe_shear_bin_p1_p99", "aloe_bodyconf_bin_p1", "aloe_below_floor_bin_p99")}))
    print("real_0_60", json.dumps({m: A["real"]["aloe_0_60"][m] for m in ("conf", "body_conf", "below_floor", "red", "lean", "shear", "pitch")}))
    print("dup_0_30", json.dumps(A["dup_cfg_0_30s"]))


if __name__ == "__main__":
    main()
