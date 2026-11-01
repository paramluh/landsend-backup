import json, os, sys
E = "/data/home/amartyavarma/video/studio-posemono/pose-ft-samples/s20/control"; R = "/data/home/amartyavarma/ppm/ppm-pose-server/requests"
out = []
for t in ("xs16610_fs", "xs17352_fs", "xs17641_fs"):
    m = json.load(open(f"{E}/{t}.json"))
    rids = sorted(set(r for r, _ in m.values()))
    g = sum(os.path.exists(f"{R}/{r}/GEN_DONE") for r in rids); gf = sum(os.path.exists(f"{R}/{r}/GEN_FAIL") for r in rids)
    v = sum(os.path.exists(f"{R}/{r}/s{k}/VIDEO_DONE") for r, k in m.values()); vf = sum(os.path.exists(f"{R}/{r}/s{k}/VIDEO_FAIL") for r, k in m.values())
    out.append(f"{t[2:7]} gen {g}/{len(rids)} f{gf} vid {v}/{len(m)} f{vf}")
print(" | ".join(out))
