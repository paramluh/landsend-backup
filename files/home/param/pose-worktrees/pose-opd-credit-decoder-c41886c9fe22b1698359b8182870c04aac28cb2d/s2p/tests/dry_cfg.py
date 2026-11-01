#!/usr/bin/env python3
"""pose-s2p (e): resolved trainer CFG in a CPU dry mode (module-level CFG of train_aloe_pose.py: no dist, no GPU, nothing written) from the
code dir argv[1] and the env already sourced; dumps it to argv[2]; with argv[3] = 'plan' also builds the world-WORLD_SIZE plan (rows digest,
rounds per step)."""
import json, os, sys, time
C = sys.argv[1]
for p in (f"{C}/vendor", f"{C}/lib", C):
    sys.path.insert(0, p)
import train_aloe_pose as T  # noqa: E402
out = {"cfg": json.loads(json.dumps(T.CFG, default=str)), "code": C, "world": int(os.environ["WORLD_SIZE"])}
if len(sys.argv) > 3 and sys.argv[3] == "plan":
    import numpy as np
    A = T.A
    t0 = time.time()
    PL = A.make_plan_v2(int(os.environ["WORLD_SIZE"]), T.CFG["data"])
    R = PL["R"]
    ndum = sum(1 for st in PL["step_micros"] for per in st for g in per if not g)
    nmic = sum(len(per) for st in PL["step_micros"] for per in st)
    out["plan"] = dict(steps=PL["steps"], n_rows=PL["n_rows"], rows_digest=A.rows_digest(PL), plan_digest=A.plan_digest(PL), R_sum=int(sum(R)),
                       R_mean=float(np.mean(R)), R_min=int(min(R)), R_max=int(max(R)), micros_total=nmic, dummy_micros=ndum,
                       max_micro_tok_ub=PL["max_micro_tok_ub"], n_eval_micros=len(PL["eval_micros"]), secs=round(time.time() - t0, 1))
    print("PLAN", out["plan"], flush=True)
json.dump(out, open(sys.argv[2], "w"), indent=1)
print("wrote", sys.argv[2])
