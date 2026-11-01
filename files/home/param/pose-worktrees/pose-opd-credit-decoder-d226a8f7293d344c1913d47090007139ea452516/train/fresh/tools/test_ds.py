"""trainer-path test: data_ppm (lib speech_q) imported FIRST, then fresh_pose via the trainer's guarded import; FreshPoseDS items for this host's
rank-0..7 first rows; timing; identical to a direct FreshRows + pose_micro_ppm build; sys.path restored."""
import os, sys, time, json
import numpy as np
C = "/data/home/amartyavarma/ppm/code-811"
sys.path.insert(0, C)
import data_ppm as D  # noqa: E402
sp_ = list(sys.path); sys.path.insert(0, os.path.join(C, "fresh"))
try:
    import fresh_pose as FP  # noqa: E402
finally:
    sys.path[:] = sp_
assert "/data/home/amartyavarma/qwen-1ch/stage2" not in sys.path
assert FP.SP is D.SP, "fresh_pose must reuse the trainer's speech_q"
plan = FP.load_plan(FP.OUT)
cfg = D.default_cfg(pose_delay=2)
out = {}
for l in range(8):
    rows = FP.rank_rows(plan["n_rows"], l, 8)
    ds = FP.FreshPoseDS(plan, rows[:2], cfg, l)
    t0 = time.time(); m = ds[0]; dt = time.time() - t0
    m2 = D.packed(D.pose_micro_ppm(FP.FreshRows(plan)[int(rows[0])], cfg, l), cfg)
    for k in ("ids", "depth", "pdepth", "sel", "y", "cb0", "pcb0", "light", "plight"):
        assert np.array_equal(m[k], m2[k]), k
    out[l] = {"rows_rank": int(len(rows)), "steps_rank": int(len(rows) // 8), "first_row": int(rows[0]), "N": int(len(m["pid"])), "s": round(dt, 3)}
print(json.dumps({"host": FP.HOST, "n_rows": plan["n_rows"], "per_rank": out}), flush=True)
