"""D31 evidence on real rows (run on a prod node, CPU): light frames per row vs the frames with a cb0 target, at the prod fractions.
usage: python tests/d31_light_counts.py [n_rows]"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_ppm as D  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
for lf, plf in ((0.1, 0.1), (1.0, 1.0)):
    cfg = dict(D.default_cfg(), light_frac=lf, pose_light_frac=plf)
    node = D.R3.load_node("/scratch/amartyavarma/s3-rows")
    idx = D.DS.load_index("/scratch/amartyavarma/s3-rows")
    plan = D.speech_plan(node, idx, 1, 8, 0, 0.0005, "ppm-v1", 2)
    ds = D.SpeechDS("/scratch/amartyavarma/s3-rows", plan["rows"][:N], cfg, 0)
    sp = [(len(m["light"]), int((m["cb0"] >= 0).sum())) for m in (ds[i] for i in range(N))]
    pn = D.PS.load_node("/scratch/amartyavarma/pose-s3-samdata/rows")
    pp = D.PS.plan(pn, 20261010, 8, 0, 1, 8, 0, 1)
    pds = D.PoseDS(pn, pp["rows"][:N], cfg, 0)
    po = []
    for i in range(N):
        m = pds[i]
        po.append((len(m["plight"]), int((m["pcb0"] >= 0).sum()), len(m["light"]), int((m["cb0"] >= 0).sum())))
        assert len(np.unique(m["plight"])) == len(m["plight"]) and len(np.unique(m["light"])) == len(m["light"])
    print(f"light_frac={lf} pose_light_frac={plf}")
    print("  speech rows: speech-dec light / frames with cb0 target: " + ", ".join(f"{a}/{b}" for a, b in sp))
    print("  pose rows:   pose-dec plight / frames with pose cb0: " + ", ".join(f"{a}/{b}" for a, b, _, _ in po))
    print("  pose rows:   speech-dec light / frames with speech cb0: " + ", ".join(f"{c}/{d}" for _, _, c, d in po), flush=True)
