"""the dummy ranks' micro = the plan's first item (fallback): its length vs the per-step longest real row (are dummies ever the slowest rank?)."""
import os, sys
import numpy as np
sys.path.insert(0, "/data/home/amartyavarma/ppm/aloe-pose-mono-8020-sam/code")
import data_aloe_pose as A
cfg = A.aloe_cfg()
P = A.make_plan_v2(96, cfg)
fb_n, fb_t = P["n_frames"][0], P["tok_ub"][0]
mx = [max(sum(P["tok_ub"][i] for i in r) for r in P["rows"][t * 64:(t + 1) * 64]) for t in range(P["steps"])]
mxf = [max(sum(P["n_frames"][i] for i in r) for r in P["rows"][t * 64:(t + 1) * 64]) for t in range(P["steps"])]
print(f"fallback item {P['items'][0]}: {fb_n} frames, position bound {fb_t}; per-step longest real row: positions bound min {min(mx)} median {int(np.median(mx))}, "
      f"frames min {min(mxf)} median {int(np.median(mxf))}; steps whose longest row is shorter than the fallback (frames): {sum(1 for x in mxf if x < fb_n)}")
