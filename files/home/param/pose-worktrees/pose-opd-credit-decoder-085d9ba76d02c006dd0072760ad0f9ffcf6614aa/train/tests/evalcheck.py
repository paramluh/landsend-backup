import sys, socket
sys.path.insert(0, "/data/home/amartyavarma/ppm/code")
import data_ppm as D
node = D.R3.load_node("/scratch/amartyavarma/s3-rows"); idx = D.DS.load_index("/scratch/amartyavarma/s3-rows")
ok = []
for local in range(8):
    p = D.speech_plan(node, idx, 20261011, 8, local, 0.0005, "ppm-v1", 2); ok.append(len(p["eval_rows"]))
pn = D.PS.load_node("/scratch/amartyavarma/pose-s3-samdata/rows")
print(socket.gethostname().split(".")[0], "speech eval rows per local rank", ok, "| pose node", pn["node"], "eval rows", int((pn["split"] == 1).sum()), flush=True)
