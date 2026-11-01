"""aloe-v2pure-clip80 pre-launch check (CPU, compute node): (1) the patched data module with 23051's cfg (no ALOE_* prompt env) reproduces
23051's plan digest bfdf5ca5beab72db at world 24 and its data cfg; (2) with this run's prompt env the rows digest (items + 64-row step groups +
eval packing) is 23051's at world 24 and 64, steps 200, every micro / eval micro <= BUDGET; (3) the realised prompt fractions over every train
and eval clip (data_aloe_pose.prompt_stats, the trainer's own function) + its cost.
  python plan_clip80_check.py <procs>   -> prints JSON lines; writes tests/plan_clip80_check.json"""
import json
import os
import sys
import time
from multiprocessing import Pool

C = "/data/home/amartyavarma/ppm/aloe-v2pure-clip80/code"
sys.path.insert(0, C)
os.environ.update(QQA_POSE_B="1", QQA_A_EMPTY="1", ALOE_STORE_DIR="/data/home/amartyavarma/ppm/aloe-v2pure-ft/stores")
NEW = {"ALOE_PERSONA_SIDES": "none", "ALOE_VOICE_RULE": "p10", "ALOE_VOICE_P10": "8", "ALOE_VOICE_TAG": "aloe-v2pure-clip80"}
for k in NEW:
    os.environ.pop(k, None)
import numpy as np  # noqa: E402
import data_aloe_pose as A  # noqa: E402

REF = json.load(open("/data/home/amartyavarma/ppm/aloe-v2pure-clip80/tools/ref_cfg_23051.json"))["cfg"]


def _ps(a):
    items, cfg, r, w = a
    return A.prompt_stats(items, cfg, r, w)


def main():
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    out = {}
    t0 = time.time()
    c0 = A.aloe_cfg()
    assert c0 == REF["data"], ("unset-env cfg != 23051's data cfg", c0, REF["data"])
    p0 = A.make_plan_v2(24, c0)
    out["ref_plan_digest_w24"] = A.plan_digest(p0)
    out["ref_rows_digest"] = A.rows_digest(p0)
    assert out["ref_plan_digest_w24"] == "bfdf5ca5beab72db", out
    out["ref_R"] = [min(p0["R"]), float(np.mean(p0["R"])), max(p0["R"])]
    out["ref_n_rows"], out["ref_steps"] = p0["n_rows"], p0["steps"]
    print(json.dumps(out), flush=True)
    os.environ.update(NEW)
    c1 = A.aloe_cfg()
    out["new_cfg"] = c1
    for w in (24, 64):
        p = A.make_plan_v2(w, c1)
        assert A.rows_digest(p) == out["ref_rows_digest"] and p["steps"] == 200 == p0["steps"], (w, A.rows_digest(p), p["steps"])
        assert p["items"] == p0["items"] and p["rows"] == p0["rows"] and p["eval_micros"] == p0["eval_micros"]
        out[f"new_w{w}"] = {"plan_digest": A.plan_digest(p), "rows_digest": A.rows_digest(p), "R": [min(p["R"]), float(np.mean(p["R"])), max(p["R"])],
                            "max_micro_tok_ub": p["max_micro_tok_ub"], "max_eval_tok_ub": p["max_eval_tok_ub"],
                            "tok_ub_total_per_step_mean": float(np.mean([sum(p["tok_ub"][i] for rw in p["rows"][t * 64:(t + 1) * 64] for i in rw) for t in range(p["steps"])]))}
    out["ref_tok_ub_total_per_step_mean"] = float(np.mean([sum(p0["tok_ub"][i] for rw in p0["rows"][t * 64:(t + 1) * 64] for i in rw) for t in range(p0["steps"])]))
    out["plan_s"] = round(time.time() - t0, 1)
    print(json.dumps({k: v for k, v in out.items() if k.startswith("new_w")}), flush=True)
    for nm, its in (("eval", p0["eval_items"]), ("train", p0["items"])):
        t1 = time.time()
        with Pool(nproc) as pool:
            s = np.sum(pool.map(_ps, [(its, c1, r, nproc) for r in range(nproc)]), 0)
        f = {k: float(s[i] / s[0]) for i, k in enumerate(A.PSTAT_KEYS)}
        f["items"] = int(s[0])
        out[f"prompt_{nm}"] = f
        out[f"prompt_{nm}_s"] = round(time.time() - t1, 1)
        print(json.dumps({f"prompt_{nm}": f, "s": out[f"prompt_{nm}_s"]}), flush=True)
    json.dump(out, open("/data/home/amartyavarma/ppm/aloe-v2pure-clip80/tests/plan_clip80_check.json", "w"), indent=1)
    print("ALL CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()
