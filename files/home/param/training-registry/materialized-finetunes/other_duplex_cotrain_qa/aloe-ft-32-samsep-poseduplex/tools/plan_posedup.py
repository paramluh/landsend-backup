"""poseduplex PLAN BUILDER (CPU; no GPU, no launch): the plan the trainer will build from exactly these knobs (env: run_poseduplex.env sourced,
QQA_STEPS / POSEDUP_* knobs), its digests, the pose coverage / hours report, and the A-FLAC presence of every item (shared /data copy).
Writes <out dir>/PLAN_<digest>.json (report) and prints the env lines to paste (POSEDUP_PLAN_DIGEST, ALOE_EXPECT_TRAIN_MICROS, warm-up and
export steps by 29345's rules).
usage (compute node): python -I -B plan_posedup.py <code dir> <out dir>"""
import json
import os
import sys

sys.path.insert(0, sys.argv[1])
OUT = sys.argv[2]
os.environ["ALOE_FLAC_ROOT"] = os.environ["ALOE_SHARED_FLAC"]
import data_aloe_posedup as A  # noqa: E402

steps = int(os.environ["QQA_STEPS"])
PL = A.make_plan(64, roots=A.SRC_STORES, steps=steps)
dg, dg0 = A.plan_digest(PL)
rep = A.plan_report(PL) if (A.POSE or A.ITEMS != "all") else {}
miss, seen = [], set()
for it in [x for x in PL["items"] if x is not None] + [tuple(x) for x in PL["eval"]]:
    a = A.vox_index()[int(it[0])][int(it[1])]["A"]["assistant" if int(it[2]) else "user"]
    if a["flac"] in seen:
        continue
    seen.add(a["flac"])
    p = f"{A.FLAC_ROOT}/{a['flac']}"
    if not os.path.exists(p) or (a["kind"] == "gcs" and os.path.getsize(p) != int(a["flac_bytes"])):
        miss.append(a["flac"])
warm = round(10 * steps / 306)                                     # 26232 / 29345's rule
exports = [round(x * steps / 253) for x in (42, 84, 126, 169, 211)]   # 29345's rule: 26232's export steps of its 253 scaled to this run
res = {"knobs": A.cfg_view(), "steps": steps, "epoch_steps": PL["epoch_steps"], "n_epochs": PL["n_epochs"], "plan_digest": dg, "epoch0_digest": dg0,
       "epoch0_is_29345": dg0 == "d1a3e3721328da0a", "train_micros": PL["n_train_micros"], "dummy_micros": PL["n_dummy"], "eval_micros": len(PL["eval"]),
       "items_per_epoch": PL["n_items0"], "a_tracks": len(seen), "a_missing": len(miss), "warmup_rule": warm, "export_steps_rule": exports, "report": rep}
os.makedirs(OUT, exist_ok=True)
json.dump(res, open(f"{OUT}/PLAN_{dg}.json", "w"), indent=1)
print(json.dumps(res, indent=1))
print(f"\n# env lines (paste into run_poseduplex.env)\nQQA_STEPS={steps}\nQQA_WARMUP={warm}\nQQA_EXPORT_STEPS={','.join(map(str, exports))}\n"
      f"POSEDUP_PLAN_DIGEST={dg}\nALOE_EXPECT_TRAIN_MICROS={PL['n_train_micros']}\nALOE_EXPECT_EVAL_MICROS={len(PL['eval'])}")
assert not miss, ("A FLACs missing in the shared copy", miss[:5])
