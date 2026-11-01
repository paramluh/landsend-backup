#!/usr/bin/env python3
"""pose6 kit (B3, 10-27): the trainer's RESOLVED CFG with the trainer's own code, no model build, no GPU, no dist, nothing written except
argv[2] (= pose-s2p/tests/dry_cfg.py, which reproduced export-357's meta cfg with 0 keys different on mono8020's code at world 64).
Importing train_aloe_pose builds its module-level CFG from the env already sourced (RANK / WORLD_SIZE / LOCAL_RANK set by the caller).
argv: <code dir> <out.json> [plan] [--ref <ref cfg json>]: with 'plan' also builds the world-WORLD_SIZE plan (rows digest, rounds);
with --ref prints + stores the flattened diff vs that reference (the trainer's own [cfgdiff] rule: flatten nested dicts, compare values)."""
import json
import os
import sys
import time

C = sys.argv[1]
for p in (f"{C}/vendor", f"{C}/lib", C):
    sys.path.insert(0, p)
import train_aloe_pose as T  # noqa: E402


def flat(d, pre=""):
    o = {}
    for k_, v_ in d.items():
        if isinstance(v_, dict):
            o.update(flat(v_, f"{pre}{k_}."))
        else:
            o[f"{pre}{k_}"] = v_
    return o


cfg = json.loads(json.dumps(T.CFG, default=str))
out = {"cfg": cfg, "code": C, "world": int(os.environ["WORLD_SIZE"]), "env_file": os.environ.get("FT_ENV_FILE", "?"),
       "train_groups": list(getattr(T, "TRAIN_GROUPS", ())), "new_keys": list(getattr(T, "NEW_KEYS", ()))}
if "plan" in sys.argv[3:]:
    import numpy as np
    A = T.A
    t0 = time.time()
    PL = A.make_plan_v2(int(os.environ["WORLD_SIZE"]), T.CFG["data"])
    R = PL["R"]
    ndum = sum(1 for st in PL["step_micros"] for per in st for g in per if not g)
    nmic = sum(len(per) for st in PL["step_micros"] for per in st)
    out["plan"] = dict(steps=PL["steps"], n_rows=PL["n_rows"], rows_digest=A.rows_digest(PL), plan_digest=A.plan_digest(PL), R_sum=int(sum(R)),
                       R_mean=float(np.mean(R)), R_min=int(min(R)), R_max=int(max(R)), micros_total=nmic, dummy_micros=ndum,
                       max_micro_tok_ub=PL["max_micro_tok_ub"], n_eval_micros=len(PL["eval_micros"]), n_train_items=PL["n_train_items"],
                       n_eval_items=PL["n_eval_items"], secs=round(time.time() - t0, 1))
    print("PLAN", out["plan"], flush=True)
if "--ref" in sys.argv:
    ref = sys.argv[sys.argv.index("--ref") + 1]
    a_, b_ = flat(json.load(open(ref))["cfg"]), flat(cfg)
    diff = [(k_, a_.get(k_, "<absent>"), b_.get(k_, "<absent>")) for k_ in sorted(set(a_) | set(b_)) if a_.get(k_, "<absent>") != b_.get(k_, "<absent>")]
    out["diff_ref"] = ref
    out["diff"] = diff
    print(f"[cfgdiff] {C} world {out['world']} vs {ref}: {len(diff)} keys differ; every other key identical", flush=True)
    for k_, x_, y_ in diff:
        print(f"[cfgdiff]   {k_}: ref {x_!r} -> this {y_!r}", flush=True)
json.dump(out, open(sys.argv[2], "w"), indent=1)
print("wrote", sys.argv[2], flush=True)
