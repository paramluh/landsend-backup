"""stage-time checks (CPU): STORES.json digests, the 64-rank plan (steps == QQA_STEPS), and a few real micros (train + eval, both swaps)."""
import json, os, sys, time
sys.path.insert(0, "/data/home/amartyavarma/ppm/aloe-pose-ft/code")
import numpy as np
import data_aloe_pose as A
t0 = time.time()
PL = A.make_plan(64)
print(f"plan: {PL['n_train_rows']} train rows -> {len(PL['items'])} train items, {PL['n_eval_rows']} eval rows -> {len(PL['eval'])} eval items, steps {PL['steps']}", flush=True)
assert PL["steps"] == int(os.environ["QQA_STEPS"]), (PL["steps"], os.environ["QQA_STEPS"])
cfg = A.aloe_cfg()
st = A.Stores()
picks = PL["items"][:6] + PL["eval"][:2] + [it for it in PL["items"] if it[2] == 1][:2]
for it in picks:
    m = A.as_micro(st.conv(it, cfg, 0))
    N = len(m["pid"])
    print(it, "N", N, "P", int(m["P"][0]), "sel", len(m["sel"]), "cb0", int((m["cb0"] >= 0).sum()), "light", len(m["light"]),
          "pcb0", int((m["pcb0"] >= 0).sum()), "plight", len(m["plight"]), "own", m["own"].tolist(), flush=True)
    assert m["pdepth"].shape == (N, 32) and len(m["plight"]) == int((m["pcb0"] >= 0).sum()) == len(m["light"])
json.dump({"total_steps": PL["steps"], "rows_per_step": 64, "eval_every": int(os.environ["QQA_EVAL_EVERY"]),
           "exports": [int(x) for x in os.environ["QQA_EXPORT_STEPS"].split(",")] + [PL["steps"]], "train_items": len(PL["items"]),
           "eval_items": len(PL["eval"])}, open("/data/home/amartyavarma/ppm/aloe-pose-ft/PAGE_CFG.json", "w"))
print(f"stage check OK ({time.time() - t0:.0f}s)")
