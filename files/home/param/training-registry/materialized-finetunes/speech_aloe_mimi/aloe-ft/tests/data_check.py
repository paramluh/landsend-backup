"""CPU check of data_aloe on the staged stores: plan sizes, both orientations of sample rows (P, T, persona tokens per channel, B = the
right speaker's words), dummy micro counts, max T over a sample."""
import json, sys, time
sys.path.insert(0, "/data/home/amartyavarma/ppm/aloe-ft/code")
import numpy as np
import data_aloe as A
D = A.D
pl = A.make_plan(64, A.SRC_STORES)
print("plan", len(pl["items"]), "items", pl["steps"], "steps", len(pl["eval"]), "eval", pl["n_train_rows"], pl["n_eval_rows"])
assert len(set(map(tuple, pl["items"]))) == len(pl["items"])
tr, ev = A.rank_items(pl, 63, 64); print("rank63 dummies", sum(x is None for x in tr), "eval", len(ev))
tr0, ev0 = A.rank_items(pl, 0, 64); print("rank0 dummies", sum(x is None for x in tr0), "eval", len(ev0))
cfg = A.aloe_cfg(); st = A.Stores(A.SRC_STORES)
tok = D._tok_init()
t0 = time.time(); Ts = []; Ps = []
for i, it in enumerate(pl["items"][:40]):
    m = st.conv(it, cfg, 0); Ts.append(len(m["pid"])); Ps.append(m["P"])
    pv = json.loads(m["prov"])
    if i < 6:
        node, rows, _, _ = st.open(it[0]); row = rows[it[1]]
        side = "user" if it[2] else "assistant"
        ws = [w for w in row[f"words_{side}"]][:6] if f"words_{side}" in row else None
        yb = tok.decode([int(x) for x in m["y"][m["word"] > 0][:12]])
        print(it, "P", m["P"], "T", len(m["pid"]), "persona", pv.get("persona"), "clips", [(c["track"], c["data_side"]) for c in pv["clips"]],
              "chanB", pv["chan_B_data_side"], "| B words:", repr(yb[:80]), "| data words", str(ws)[:120])
mic = A.as_micro(m, dummy=True); print("dummy own", mic["own"].tolist(), "dummy flag", mic.get("dummy"))
print("T max", max(Ts), "P mean", np.mean(Ps), "P max", max(Ps), f"{(time.time() - t0) / 40:.2f}s/micro")
