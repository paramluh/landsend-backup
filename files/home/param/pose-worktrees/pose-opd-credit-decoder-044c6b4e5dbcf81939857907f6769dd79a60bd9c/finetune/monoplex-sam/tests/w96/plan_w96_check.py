"""w96 PLAN CHECK (CPU; aloe-pose-mono-8020-sam, 10-26): the trainer's own plan code (data_aloe_pose.make_plan_v2, env = the run's env file)
at world 64 (job 24272) vs world 96 (12 nodes), on the ORIGINAL stores (the -sam stores have the same items / n_frames, so the same plan).
Checks (each printed PASS / FAIL, JSON to out/plan_w96_check.json):
  P1 rows digest (items + rows + eval packing) world-independent and == ALOE_ROWS_DIGEST (3681cc26ea97cad7)
  P2 357 steps at both worlds; micro layout row_per_rank (R = 1 at every step)
  P3 every step t: rank j < 64 gets the SAME row at world 96 as at world 64; ranks 64..95 are dummies ([[]]) at every step
  P4 dummy micros: world 64 = 23 (last step only); world 96 = 32 x 356 + 55 (32 per step + the last step's 23 more)
  P5 eval packing (529 micros) identical; per-rank eval micros: max 9 at world 64 vs max 6 at world 96 (round-robin)
  P6 data micro is rank-INDEPENDENT: row j's micro built with rank argument j (world-64 index) vs another rank number -> byte-identical
     arrays (light frames are keyed on row_key under the system-prompt path; light_frac 1.0)
  P7 step-0 loss check on a DUMMY rank: original code (ALOE_LOSSCHECK_DUMMY=0) -> the dummy micro has NO 'singles' (pack_loss_check
     would raise KeyError); with ALOE_LOSSCHECK_DUMMY=1 -> 'singles' = [the fallback item's real micro]
usage (compute node): python plan_w96_check.py"""
import json
import os
import sys
import time

import numpy as np

CODE = os.environ.get("EQ_CODE", "/data/home/amartyavarma/ppm/aloe-pose-mono-8020-sam/code")
sys.path.insert(0, CODE)
import data_aloe_pose as A  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
res, ok_all = {}, True


def check(name, ok, info):
    global ok_all
    ok_all &= bool(ok)
    res[name] = {"pass": bool(ok), "info": info}
    print(f"{'PASS' if ok else 'FAIL'} {name}: {info}", flush=True)


t0 = time.time()
cfg = A.aloe_cfg()
assert A.MONO and A.ROWS_PER_STEP == 64, (A.MONO, A.ROWS_PER_STEP)
P = {W: A.make_plan_v2(W, cfg) for W in (64, 96)}
print(f"plans built in {time.time() - t0:.0f}s", flush=True)
want = os.environ.get("ALOE_ROWS_DIGEST", "3681cc26ea97cad7")
rd = {W: A.rows_digest(P[W]) for W in P}
check("P1 rows digest world-independent", rd[64] == rd[96] == want, {"w64": rd[64], "w96": rd[96], "want": want,
      "plan_digest_w64": A.plan_digest(P[64]), "plan_digest_w96": A.plan_digest(P[96]),
      "note": "plan_digest includes step_micros (world-dependent): only used for save/resume identity"})
check("P2 steps + layout", P[64]["steps"] == P[96]["steps"] == 357 and set(P[64]["R"]) == set(P[96]["R"]) == {1},
      {"steps": [P[64]["steps"], P[96]["steps"]], "R": [sorted(set(P[64]["R"])), sorted(set(P[96]["R"]))],
       "n_rows": P[64]["n_rows"], "max_micro_tok_ub": [P[64]["max_micro_tok_ub"], P[96]["max_micro_tok_ub"]]})
bad, bad_d = [], []
for t in range(357):
    a, b = P[64]["step_micros"][t], P[96]["step_micros"][t]
    for j in range(64):
        if a[j] != b[j]:
            bad.append((t, j))
    for j in range(64, 96):
        if b[j] != [[]]:
            bad_d.append((t, j))
check("P3 same row -> same rank index; ranks 64..95 dummies", not bad and not bad_d,
      {"mismatch_real": bad[:5], "nondummy_64plus": bad_d[:5]})
nd = {W: sum(1 for st in P[W]["step_micros"] for g in st if not g[0]) for W in P}
last_rows = P[64]["n_rows"] - 356 * 64
check("P4 dummy micros", nd[64] == 64 - last_rows and nd[96] == 32 * 356 + (96 - last_rows),
      {"w64": nd[64], "w96": nd[96], "last_step_rows": last_rows,
       "dummy_ranks_per_step_w96": "32 (ranks 64..95) at steps 1..356, 55 at step 357"})
em = {W: P[W]["eval_micros"] for W in P}
per = {W: [len(em[W][r::W]) for r in range(W)] for W in P}
check("P5 eval packing identical; per-rank eval micros", em[64] == em[96] and len(em[64]) == 529,
      {"n_eval_micros": len(em[64]), "per_rank_w64": [min(per[64]), max(per[64])], "per_rank_w96": [min(per[96]), max(per[96])]})
# P6 rank-independence of the micro arrays (step 0 rows 0, 5, 63 built with rank j and with rank j + 64 / 17)
fb = P[96]["items"][0]
mism = []
for j, other in ((0, 64), (5, 17), (63, 95)):
    sl = [P[96]["step_micros"][0][j]]
    m1 = A.StepDS(P[96]["items"], sl, cfg, j, fb, singles_at=-1)[0][0]
    m2 = A.StepDS(P[96]["items"], sl, cfg, other, fb, singles_at=-1)[0][0]
    for k in sorted(set(m1) | set(m2)):
        x, y = m1.get(k), m2.get(k)
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            same = isinstance(x, np.ndarray) and isinstance(y, np.ndarray) and x.dtype == y.dtype and np.array_equal(x, y)
        else:
            same = x == y
        if not same:
            mism.append((j, other, k))
check("P6 micro arrays rank-independent", not mism, {"rows_checked": "step-0 rows on ranks 0 / 5 / 63 vs rank args 64 / 17 / 95",
                                                    "mismatches": mism[:8]})
# P7 the step-0 loss check on a dummy rank (rank 64 at world 96)
sl = [P[96]["step_micros"][0][64]]
A.LOSSCHECK_DUMMY = False
b0 = A.StepDS(P[96]["items"], sl, cfg, 64, fb, singles_at=0)[0][0]
A.LOSSCHECK_DUMMY = True
b1 = A.StepDS(P[96]["items"], sl, cfg, 64, fb, singles_at=0)[0][0]
s1 = b1.get("singles", [None])[0]
check("P7 step-0 loss check on a dummy rank", int(b0.get("dummy", 0)) == 1 and "singles" not in b0 and s1 is not None and not s1.get("dummy")
      and float(np.asarray(s1["own"]).sum()) > 0,
      {"original_code_has_singles": "singles" in b0, "fixed_code_singles_items": None if s1 is None else s1["item"],
       "fixed_code_dummy_arrays_equal_original": all(np.array_equal(b0[k], b1[k]) for k in b0 if isinstance(b0[k], np.ndarray))})
res["all_pass"] = ok_all
res["time_s"] = time.time() - t0
json.dump(res, open(f"{OUT}/plan_w96_check.json", "w"), indent=1, default=str)
print("ALL PASS" if ok_all else "SOME CHECK FAILED", flush=True)
