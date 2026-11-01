import os, sys, json
for l in open("/data/home/amartyavarma/ppm/aloe-pose-rosie/rosie.env"):
    l = l.split("#")[0].strip()
    if "=" in l:
        k, v = l.split("=", 1); os.environ.setdefault(k.strip(), v.strip())
sys.path.insert(0, "/data/home/amartyavarma/ppm/aloe-pose-rosie/code")
import data_aloe_pose as A
cfg = A.aloe_cfg()
pl = A.make_plan_v2(8, cfg)
print(json.dumps({"steps": pl["steps"], "rows": pl["n_rows"], "train_items": pl["n_train_items"], "eval_items": pl["n_eval_items"],
                  "train_h": round(pl["train_frames"] / 12.5 / 3600, 2), "eval_micros": len(pl["eval_micros"]), "rows_digest": A.rows_digest(pl),
                  "plan_digest": A.plan_digest(pl), "max_tok": pl["max_micro_tok_ub"], "warmup_rule": round(10 * pl["steps"] / 306)}))
