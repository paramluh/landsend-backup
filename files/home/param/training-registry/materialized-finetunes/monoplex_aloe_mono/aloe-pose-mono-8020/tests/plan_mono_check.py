"""aloe-pose-mono-8020 pre-launch check (CPU, compute node; never the login node).
  python plan_mono_check.py ref        -> (regression) the patched data module with 24198's / 23051's env reproduces their plans exactly
  python plan_mono_check.py mono <procs> -> this run's plan at world 64 (items, rows, steps, layout, rows digest), sample micros built with
                                          the trainer's own code under this run's cfg and 23051's cfg, and the realised prompt cells over
                                          EVERY train / eval item (data_aloe_pose.prompt_stats_mono, the trainer's function).
Writes tests/plan_mono_check.json (mono) / tests/plan_mono_ref.json (ref)."""
import json
import os
import sys
import time
from multiprocessing import Pool

Q = "/data/home/amartyavarma/ppm/aloe-pose-mono-8020"
C = f"{Q}/code"
MODE = sys.argv[1]


def env_from(path):
    out = {}
    for ln in open(path):
        ln = ln.strip()
        if not ln or ln.startswith("#") or "=" not in ln:
            continue
        k, v = ln.split("=", 1)
        out[k] = v
    return out


if MODE == "ref":
    E = env_from("/data/home/amartyavarma/ppm/aloe-v2pure-clip80/aloe_clip80.env")
else:
    E = env_from(f"{Q}/aloe_mono8020.env.tmpl")
for k, v in E.items():
    if "@" not in v:
        os.environ[k] = v
sys.path.insert(0, C)
import numpy as np  # noqa: E402
import data_aloe_pose as A  # noqa: E402
import data_qqa as QQ  # noqa: E402


def _ps(a):
    items, cfg, ref, r, w = a
    return A.prompt_stats_mono(items, cfg, ref, r, w)


def lpt_dummies(p, world):
    """what 23051's LPT layout (ALOE_MICRO_TOK bins) would give on this plan: rounds and empty bins (= dummy micros) per step."""
    R_all, empt = [], 0
    for t in range(p["steps"]):
        cl = [i for rw in p["rows"][t * A.ROWS_PER_STEP:(t + 1) * A.ROWS_PER_STEP] for i in rw]
        tot = sum(p["tok_ub"][i] for i in cl)
        R = max(1, -(-tot // (world * A.MICRO_TOK)))
        while True:
            bins, load = A._lpt([p["tok_ub"][i] for i in cl], world * R)
            if max(load) <= A.BUDGET:
                break
            R += 1
        R_all.append(R); empt += sum(1 for b in bins if not b)
    return {"R_mean": float(np.mean(R_all)), "R_max": max(R_all), "empty_bins_total": empt,
            "dummy_frac": empt / float(sum(R_all) * world)}


def main_ref():
    out = {}
    assert not A.MONO
    c1 = A.aloe_cfg()
    p = A.make_plan_v2(24, c1)
    out["clip80_rows_digest"] = A.rows_digest(p)
    out["clip80_plan_digest_w24"] = A.plan_digest(p)
    assert out["clip80_rows_digest"] == "e606adf327cb0f9d", out
    assert c1 == json.load(open(f"{Q}/tools/ref_cfg_24198.json"))["cfg"]["data"], "24198 data cfg"
    for k in [k for k in E if k.startswith("ALOE_") and k not in ("ALOE_STORE_DIR", "ALOE_ROWS_PER_STEP", "ALOE_MICRO_TOK", "ALOE_SAVE_ROOT",
                                                                   "ALOE_SAVE_EVERY_MIN", "ALOE_RESUME", "ALOE_LOCAL_EXPORT_ROOT", "ALOE_ROWS_DIGEST")]:
        os.environ.pop(k, None)
    c0 = A.aloe_cfg()
    assert c0 == json.load(open(f"{Q}/tools/ref_cfg_23051.json"))["cfg"]["data"], "23051 data cfg"
    p0 = A.make_plan_v2(24, c0)
    out["v2pure_plan_digest_w24"] = A.plan_digest(p0)
    assert out["v2pure_plan_digest_w24"] == "bfdf5ca5beab72db", out
    out["ok"] = True
    print(json.dumps(out), flush=True)
    json.dump(out, open(f"{Q}/tests/plan_mono_ref.json", "w"), indent=1)
    print("REF CHECKS PASSED", flush=True)


def main_mono(nproc):
    out = {}
    t0 = time.time()
    assert A.MONO and A.STORE_DIR == "/data/home/amartyavarma/ppm/aloe-pose-ft/stores"
    cfg = A.aloe_cfg()
    ref = QQ.qqa_cfg()
    assert ref == json.load(open(f"{Q}/tools/ref_cfg_23051.json"))["cfg"]["data"], "ref cfg != 23051's"
    out["cfg"] = cfg
    p = A.make_plan_v2(64, cfg)
    out.update(n_train_items=p["n_train_items"], n_eval_items=p["n_eval_items"], train_frames=p["train_frames"],
               train_hours=round(p["train_frames"] / 12.5 / 3600, 1), eval_frames=p["eval_frames"], n_rows=p["n_rows"], steps=p["steps"],
               R=[min(p["R"]), max(p["R"])], max_micro_tok_ub=p["max_micro_tok_ub"], max_eval_tok_ub=p["max_eval_tok_ub"],
               n_eval_micros=len(p["eval_micros"]), skipped_lt2=p["skipped_lt2"], rows_digest=A.rows_digest(p), plan_digest=A.plan_digest(p))
    assert p["n_train_items"] == 27517 and p["n_eval_items"] == 531, ("items != 22091's qualifying tracks", p["n_train_items"], p["n_eval_items"])
    ipr = [len(r) for r in p["rows"]]
    fpr = [sum(p["n_frames"][i] for i in r) for r in p["rows"]]
    out["items_per_row"] = {"mean": float(np.mean(ipr)), "max": max(ipr), "hist": np.bincount(ipr).tolist()}
    out["frames_per_row"] = {"mean": float(np.mean(fpr)), "p10": float(np.percentile(fpr, 10)), "min": min(fpr)}
    ndum = sum(1 for st in p["step_micros"] for g in st for x in g if not x)
    out["dummy_micros"] = ndum
    steps = p["steps"]
    out["warmup"] = round(10 * steps / 306)
    out["export_steps"] = [round(steps * k / 6) for k in range(1, 6)]
    out["lpt_layout_would_give"] = lpt_dummies(p, 64)
    # every item once; rows <= caps
    seen = sorted(i for rw in p["rows"] for i in rw)
    assert seen == list(range(p["n_train_items"]))
    assert max(fpr) <= A.ROW_CAP and max(sum(p["tok_ub"][i] for i in rw) for rw in p["rows"]) <= A.BUDGET
    flat = sorted(i for st in p["step_micros"] for g in st for x in g for i in x)
    assert flat == list(range(p["n_train_items"])), "every item exactly once in the step micros"
    out["plan_s"] = round(time.time() - t0, 1)
    print(json.dumps({k: v for k, v in out.items() if k != "cfg"}), flush=True)
    # sample micros: step 0 ranks 0..5 (packed), the last step's last rank (dummy), 4 eval micros under both cfgs
    t1 = time.time()
    sl = [p["step_micros"][t][r] for t in range(steps) for r in (0,)]
    smp = []
    for (t, r) in [(0, 0), (0, 1), (0, 2), (0, 63), (1, 5), (steps - 1, 63)]:
        ds = A.StepDS(p["items"], [p["step_micros"][t][r]], cfg, r, p["items"][0])
        b = ds[0][0]
        smp.append({"step": t, "rank": r, "T": int(len(b["pid"]) if "pid" in b else -1), "segs": [int(x) for x in b["segs"]],
                    "P": [int(x) for x in np.atleast_1d(b["P"])], "dummy": int(b.get("dummy", 0)), "own": np.asarray(b["own"]).sum(0).tolist()})
    out["sample_micros"] = smp
    ev = []
    for g in p["eval_micros"][:3] + p["eval_micros"][-2:]:
        for nm, c_ in (("run", cfg), ("ref23051", ref)):
            b = A.pack_micro([A.Stores().conv(p["eval_items"][j], c_, 0) for j in g])
            ev.append({"cfg": nm, "items": len(g), "T": int(len(b["pid"])), "P": [int(x) for x in np.atleast_1d(b["P"])]})
    out["sample_eval_micros"] = ev
    out["sample_s"] = round(time.time() - t1, 1)
    print(json.dumps({"sample_micros": smp, "sample_eval_micros": ev}), flush=True)
    sl = None
    for nm, its in (("eval", p["eval_items"]), ("train", p["items"])):
        t2 = time.time()
        with Pool(nproc) as pool:
            s = np.sum(pool.map(_ps, [(its, cfg, ref, r, nproc) for r in range(nproc)]), 0)
        f = {k: float(s[i] / s[0]) for i, k in enumerate(A.PSTAT_MONO_KEYS)}
        f["items"] = int(s[0])
        out[f"prompt_{nm}"] = f
        out[f"prompt_{nm}_s"] = round(time.time() - t2, 1)
        print(json.dumps({f"prompt_{nm}": f, "s": out[f"prompt_{nm}_s"]}), flush=True)
    json.dump(out, open(f"{Q}/tests/plan_mono_check.json", "w"), indent=1)
    print("MONO CHECKS PASSED", flush=True)


if __name__ == "__main__":
    if MODE == "ref":
        main_ref()
    else:
        main_mono(int(sys.argv[2]) if len(sys.argv) > 2 else 16)
