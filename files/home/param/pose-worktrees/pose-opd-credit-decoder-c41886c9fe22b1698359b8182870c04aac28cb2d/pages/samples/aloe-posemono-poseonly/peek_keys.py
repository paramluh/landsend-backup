import os, sys
sys.path.insert(0, "/data/home/amartyavarma/ppm/aloe-v2pure-ft/code")
import data_aloe_pose as A
sp = A.splits_v2(A.STORES)
keys = [e[4] for s in ("train", "eval") for e in sp[s]]
print(len(keys), keys[:3])
for rid in ("1288bbe7", "51b7da36", "21bc9427"):
    print(rid, [k for k in keys if rid in k][:12])
