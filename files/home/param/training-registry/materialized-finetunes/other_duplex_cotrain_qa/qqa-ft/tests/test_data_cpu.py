"""CPU check of data_qqa on this host: first steps of local ranks 0 and 7 + eval rows: packing invariants, P histogram, lengths vs run C."""
import json, os, socket, sys, time
from collections import Counter
sys.path.insert(0, "/data/home/amartyavarma/ppm/qqa-ft/code")
import numpy as np
import data_qqa as Q
D = Q.D
h = socket.gethostname().split(".")[0]
PL = json.load(open(f"/data/home/amartyavarma/ppm/qqa-ft/plan/PLAN-{h}.json"))
cfg = Q.qqa_cfg()
print("CD", D.CD, "cfg", cfg)
t0 = time.time()
Ph = Counter(); stats = []
for l in ("0", "7"):
    ds = Q.PackDS(PL["store"], PL["ranks"][l], cfg, int(l), singles_at=0)
    for s in (0, 1, 2, 222):
        b = ds[s]
        Ph.update(b["P"].tolist())
        K = len(b["segs"]); N = sum(b["segs"]); Lr = PL["L_runC"][l][s]
        assert K == len(Lr)
        # per conversation: T_k = P + n + 2; run C: P_c + n + 3
        dT = [t - lr for t, lr in zip(b["segs"], Lr)]
        stats.append((l, s, K, N, sum(Lr), min(dT), max(dT)))
        # invariants
        o = np.concatenate([[0], np.cumsum(b["segs"])[:-1]])
        for k, (ok, t) in enumerate(zip(o, b["segs"])):
            ids = b["ids"][:, ok:ok + t]; dep = b["depth"][:, ok:ok + t]; P = int(b["P"][k])
            assert (dep[:, P:P + D.CD] == -1).all() and (dep[:, P + D.CD:, 0] >= 0).all(), "codes at CD 2 over the conversation"
            assert (ids[0, P:] != D.PAD_ID).any() and (ids[1, P:] != D.PAD_ID).any(), "A and B lanes present"
        if s == 0:
            sg = b["singles"]
            assert sum(len(m["pid"]) for m in sg) == N and np.array_equal(np.concatenate([m["ids"] for m in sg], 1), b["ids"])
        print(f"rank {l} step {s}: K {K} N {N} (run C {sum(Lr)}), T_k - L_runC in [{min(dT)}, {max(dT)}], text targets {int(b['n_text'])}, "
              f"cb0 {int((b['cb0'] >= 0).sum())}, light {len(b['light'])}, rows {json.loads(b['prov'])[:2]}", flush=True)
ev = Q.eval_micros(PL["store"], PL["eval"]["0"], cfg, 0)
print("eval rank 0:", [(json.loads(json.dumps(m["row_key"])), len(m["pid"]), int(m["P"])) for m in ev])
print("P histogram", dict(Ph), f"{time.time() - t0:.1f}s")
m = ev[0]; P = int(m["P"])
print("eval[0] prefix ids", m["ids"][:, :P].tolist(), "present", m["present"][:, :P].tolist())
