"""max packed-micro length over the whole plan in PPM format (T_k = P + n + CD, P = 5) from the store's frame counts."""
import json, socket, sys
sys.path.insert(0, "/data/home/amartyavarma/ppm/qqa-ft/code")
import numpy as np
import data_qqa as Q
h = socket.gethostname().split(".")[0]
PL = json.load(open(f"/data/home/amartyavarma/ppm/qqa-ft/plan/PLAN-{h}.json"))
node = Q.D.R3.load_node(PL["store"]); n = np.asarray(node["length"])
mx = []
for l, steps in PL["ranks"].items():
    for s, p in enumerate(steps):
        mx.append((int(sum(n[r] + 5 + 2 for r in p)), l, s))
mx.sort()
print(h, "max", mx[-5:], "over 7503:", sum(1 for x in mx if x[0] > 7503), "of", len(mx))
