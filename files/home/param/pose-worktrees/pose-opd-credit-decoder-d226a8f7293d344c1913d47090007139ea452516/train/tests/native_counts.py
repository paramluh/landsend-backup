"""per training node: FULL train rows of each local rank under the live run's speech plan (D2 extra-slice planner input)."""
import json, os, socket, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PPM_CD", "2")
import data_ppm as D
root = "/scratch/amartyavarma/s3-rows"
node = D.R3.load_node(root); idx = D.DS.load_index(root)
out = {"host": socket.gethostname().split(".")[0], "node": node["node"], "counts": []}
for l in range(8):
    p = D.speech_plan(node, idx, 20261011, 8, l, 0.0005, "ppm-v1", 2)
    out["counts"].append(int(len(p["rows"])))
print(json.dumps(out))
