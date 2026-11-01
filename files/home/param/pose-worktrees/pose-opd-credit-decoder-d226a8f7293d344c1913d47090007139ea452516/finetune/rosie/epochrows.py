import os, sys, json
for l in open("/data/home/amartyavarma/ppm/aloe-pose-rosie/rosie.env"):
    l = l.split("#")[0].strip()
    if "=" in l:
        k, v = l.split("=", 1); os.environ.setdefault(k.strip(), v.strip())
sys.path.insert(0, "/data/home/amartyavarma/ppm/aloe-pose-rosie/code")
import data_aloe_pose as A
pl = A.make_plan_v2(8, A.aloe_cfg())
n1 = pl["n_train_items"] // 2
e1 = sum(1 for rw in pl["rows"] if rw[0] < n1)
print(json.dumps({"rows_epoch1": e1, "rows_epoch2": len(pl["rows"]) - e1, "steps": pl["steps"], "boundary_step": e1 // 8 + 1, "rows_in_boundary_step_from_epoch1": e1 % 8}))
