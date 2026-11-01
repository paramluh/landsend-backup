"""aloe-pose-mono-8020-sam pre-launch check (CPU, compute node; never the login node). The trainer's OWN plan / config / schedule code
(code/data_aloe_pose.make_plan_v2, the CFG block of code/train_aloe_pose.py, sched_s3q.make_lambda) under a given env file, store dir and world.
  python plan_sam_check.py one <env file> <store dir> <world> <out.json> [code dir]   -> one plan / cfg / schedule digest record
  python plan_sam_check.py diff <a.json> <b.json> ...                                  -> asserts the plan / schedule records are identical
Record: items / rows / step groups / eval packing digests (rows_digest = the launch-asserted ALOE_ROWS_DIGEST, plan_digest), counts, steps,
warmup, export steps, the LR multiplier of every step (trunk groups) and every decoder update (K x steps), and the resolved CFG (WORLD_SIZE
given; the CFG block is exec'd from the trainer source so the check runs without GPUs)."""
import hashlib
import json
import os
import re
import sys


def env_from(path):
    out = {}
    for ln in open(path):
        ln = ln.strip()
        if ln and not ln.startswith("#") and "=" in ln:
            k, v = ln.split("=", 1)
            out[k] = v
    return out


def h(x):
    return hashlib.sha256(json.dumps(x, default=str).encode()).hexdigest()[:16]


def one(envf, store, world, out, code):
    E = env_from(envf)
    assert not any("@" in v for v in E.values()), "env has @placeholders@"
    os.environ.update(E)
    os.environ.update(ALOE_STORE_DIR=store, WORLD_SIZE=str(world), RANK="0", LOCAL_RANK="0",
                      QQA_INIT="/data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-11488")
    sys.path.insert(0, code); sys.path.insert(0, f"{code}/lib")
    import data_qqa as Q
    import data_aloe_pose as A
    import sched_s3q as SCH
    src = open(f"{code}/train_aloe_pose.py").read()
    a, b = src.index("\ndef E(k, d=None):"), src.index('\nPOSE = CFG["pose_b"]')
    c = src.index('\nCFG["sched"] = '); d = src.index("\nif POSE:\n    TRAIN_GROUPS")
    g = {"os": os, "A": A, "Q": Q, "D": Q.D}
    exec(src[a:b] + src[c:d], g)
    CFG = g["CFG"]
    assert A.STORE_DIR == store and A.MONO
    p = A.make_plan_v2(world, CFG["data"])
    lam, dlam = SCH.make_lambda(True, CFG["sched"]), SCH.make_lambda(True, CFG["dec_sched"])
    lr = [lam(s) for s in range(CFG["steps"] + 1)]
    dlr = [dlam(s) for s in range(CFG["steps"] * CFG["dec_k"] + 1)]
    ndum = sum(1 for st in p["step_micros"] for gg in st for x in gg if not x)
    rec = {"env": envf, "store": store, "world": world, "code": code,
           "rows_digest": A.rows_digest(p), "plan_digest": A.plan_digest(p), "items_digest": h(p["items"]), "rows_list_digest": h(p["rows"]),
           "eval_micros_digest": h(p["eval_micros"]), "step_micros_digest": h(p["step_micros"]),
           "n_train_items": p["n_train_items"], "n_eval_items": p["n_eval_items"], "n_rows": p["n_rows"], "steps": p["steps"],
           "train_frames": p["train_frames"], "eval_frames": p["eval_frames"], "R": [min(p["R"]), max(p["R"])], "n_eval_micros": len(p["eval_micros"]),
           "dummy_micros": ndum, "micro_layout": CFG.get("micro_layout"), "cfg_steps": CFG["steps"], "warmup": CFG["warmup"],
           "warmup_rule": round(10 * p["steps"] / 306), "export_steps": CFG["export_steps"],
           "export_rule": [round(p["steps"] * k / 6) for k in range(1, 6)], "sched": CFG["sched"], "dec_sched": CFG["dec_sched"],
           "lr_mult_digest": h(lr), "dec_lr_mult_digest": h(dlr), "lr_mult_first_last": [lr[:14], lr[-3:]], "cfg": CFG}
    # 64 real rows per step on ranks 0..63 at any world >= 64: the per-rank micro of every rank < 64 equals world 64's (digest per rank)
    rec["step_micros_ranks_lt64_digest"] = h([[st[r] for r in range(min(64, len(st)))] for st in p["step_micros"]])
    rec["step_micros_ranks_ge64_all_dummy"] = all(not x for st in p["step_micros"] for gg in st[64:] for x in gg)
    json.dump(rec, open(out, "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in rec.items() if k not in ("cfg", "lr_mult_first_last")}, default=str), flush=True)


def flat(d, pre=""):
    o = {}
    for k, v in d.items():
        if isinstance(v, dict):
            o.update(flat(v, f"{pre}{k}."))
        else:
            o[f"{pre}{k}"] = v
    return o


def diff(paths):
    rs = [json.load(open(x)) for x in paths]
    keys = ["rows_digest", "plan_digest", "items_digest", "rows_list_digest", "eval_micros_digest", "n_train_items", "n_eval_items",
            "n_rows", "steps", "train_frames", "eval_frames", "n_eval_micros", "cfg_steps", "warmup", "export_steps", "sched", "dec_sched",
            "lr_mult_digest", "dec_lr_mult_digest", "step_micros_ranks_lt64_digest"]
    bad = []
    for k in keys:
        vals = [json.dumps(r[k]) for r in rs]
        print(f"{k:32s} {'SAME' if len(set(vals)) == 1 else 'DIFF'} {vals[0][:70]}" + ("" if len(set(vals)) == 1 else f" vs {vals[1:]}"))
        if len(set(vals)) > 1:
            bad.append(k)
    for r in rs[1:]:
        a, b = flat(json.loads(json.dumps(rs[0]["cfg"], default=str))), flat(json.loads(json.dumps(r["cfg"], default=str)))
        dk = [k for k in sorted(set(a) | set(b)) if a.get(k, "<absent>") != b.get(k, "<absent>")]
        print(f"cfg diff {rs[0]['env']} ({rs[0]['world']}) vs {r['env']} ({r['world']}): {len(dk)} keys: {dk}")
    print("PLAN/SCHEDULE IDENTICAL" if not bad else f"DIFFERENT: {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    if sys.argv[1] == "one":
        one(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5],
            sys.argv[6] if len(sys.argv) > 6 else "/data/home/amartyavarma/ppm/aloe-pose-mono-8020-sam/code")
    else:
        diff(sys.argv[2:])
