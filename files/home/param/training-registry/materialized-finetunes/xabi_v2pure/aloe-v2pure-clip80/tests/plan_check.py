"""aloe-v2pure-ft CPU check of the training plan + packed step micros (no GPU). env: as aloe_v2pure.env (S3_MIN_FRAMES=1 PPM_CD=2
QQA_POSE_B=1 QQA_A_EMPTY=1). usage: python -I plan_check.py [world=24] [n_steps_to_build=2]"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/data/home/amartyavarma/ppm/aloe-v2pure-ft/code")
import data_aloe_pose as A  # noqa: E402

W = int(sys.argv[1]) if len(sys.argv) > 1 else 24
NB = int(sys.argv[2]) if len(sys.argv) > 2 else 2
cfg = A.aloe_cfg()
t0 = time.time()
PL = A.make_plan_v2(W, cfg)
R = np.asarray(PL["R"])
print(f"plan {A.plan_digest(PL)} in {time.time() - t0:.0f}s: train {PL['n_train_items']} clips {PL['train_frames']} frames -> rows {PL['n_rows']} "
      f"(mean frames/row {PL['train_frames'] / PL['n_rows']:.0f}) -> steps {PL['steps']} (last step rows {PL['n_rows'] - 64 * (PL['steps'] - 1)}); "
      f"rounds R min {R.min()} mean {R.mean():.2f} max {R.max()} sum {R.sum()}; max micro tok ub {PL['max_micro_tok_ub']}; "
      f"eval {PL['n_eval_items']} clips {PL['eval_frames']} frames in {len(PL['eval_micros'])} micros", flush=True)
fr = [sum(PL["n_frames"][i] for rw in PL["rows"][t * 64:(t + 1) * 64] for i in rw) for t in range(PL["steps"])]
print(f"frames/step mean {np.mean(fr):.0f} min {min(fr)} max {max(fr)} (22317: 317,436 mean)", flush=True)
seen = sorted(i for t in range(PL["steps"]) for rk in range(W) for g in PL["step_micros"][t][rk] for i in g)
assert seen == list(range(PL["n_train_items"])), "every train clip exactly once"
es = sorted(i for g in PL["eval_micros"] for i in g)
assert es == list(range(PL["n_eval_items"])), "every eval clip exactly once"
steps_ = int(PL["steps"])
print("env: QQA_STEPS", steps_, "QQA_WARMUP", round(10 * steps_ / 306), "QQA_EXPORT_STEPS", ",".join(str(round(steps_ * k / 6)) for k in range(1, 6)))
for t in range(NB):
    for rk in (0, W - 1):
        ds = A.StepDS(PL["items"], [PL["step_micros"][x][rk] for x in range(PL["steps"])], cfg, rk, PL["items"][0], start=t, singles_at=-1)
        t1 = time.time()
        bl = ds[0]
        Ts = [len(b["pid"]) for b in bl]
        print(f"step {t} rank {rk}: {len(bl)} micros, positions {Ts}, clips {[len(json.loads(b['item'])) for b in bl]}, "
              f"pose frames {[len(b['plight']) for b in bl]}, own sum {np.sum([b['own'].sum(0) for b in bl], 0).tolist()}, build {time.time() - t1:.1f}s", flush=True)
ev = A.eval_micros_v2(PL["eval_items"], PL["eval_micros"][0::W], cfg, 0)
print(f"eval rank 0: {len(ev)} micros, positions max {max(len(m['pid']) for m in ev)}, light max {max(len(m['light']) for m in ev)}, "
      f"swaps {sorted({int(m['swap']) for m in ev})}", flush=True)
