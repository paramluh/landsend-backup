"""aloe-v2-ft CPU check (run on a training node): plan counts, steps / warmup / export steps, realised voice-clip fraction over the whole
plan (train + eval), and a few real micros (train cfg: light ~10 %, empty block P = 5 / B clip, A blank; eval cfg: light = all frames)."""
import json
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, "/data/home/amartyavarma/ppm/aloe-v2-ft/code")
import data_aloe_v2 as A  # noqa: E402

CFG = A.aloe_cfg()
_ST = None


def clip_of(item):
    global _ST
    if _ST is None:
        _ST = A.Stores()
    k, r, sw = item
    node, st, sib = _ST.open(int(k))
    row = st[int(r)]
    wv = A.want_voice(row["row_key"], sw, CFG["voice_seed"])
    if not wv:
        return 0, 0
    cl = A.D.voice_clips(row, sib, CFG)[1 ^ int(sw)]
    return 1, int(cl is not None)


if __name__ == "__main__":
    t0 = time.time()
    PL = A.make_plan(64)
    steps = PL["steps"]
    warm = round(10 * steps / 306)
    ex = [round(steps * i / 6) for i in range(1, 6)]
    out = {"n_all": PL["n_all"], "n_first": PL["n_first"], "train_items": len(PL["items"]), "eval_items": len(PL["eval"]), "steps": steps,
           "warmup": warm, "export_steps": ex, "dummies_last_step": steps * 64 - len(PL["items"])}
    with Pool(32) as p:
        tr = p.map(clip_of, PL["items"], chunksize=64)
        ev = p.map(clip_of, PL["eval"], chunksize=8)
    for name, v in (("train", tr), ("eval", ev)):
        a = np.asarray(v)
        out[f"{name}_voice_drawn_frac"] = float(a[:, 0].mean()); out[f"{name}_voice_clip_frac"] = float(a[:, 1].mean())
        out[f"{name}_drawn_but_no_clip"] = int((a[:, 0] - a[:, 1]).sum())
    st = A.Stores()
    ecfg = A.eval_cfg()
    for it in PL["items"][:6] + PL["eval"][:2]:
        m = st.conv(it, CFG, 0)
        me = st.conv(it, ecfg, 0)
        n_cb0 = int((np.asarray(m["cb0"]) >= 0).sum())
        P = int(m["P"])
        assert (m["ids"][0, :P] == A.D.PAD_ID).all() and (m["present"][0, :P] == 0).all() and (m["depth"][0, :P] < 0).all()
        assert np.array_equal(m["ids"], me["ids"]) and np.array_equal(m["sel"], me["sel"]) and np.array_equal(m["cb0"], me["cb0"])
        assert len(me["light"]) == n_cb0, (len(me["light"]), n_cb0)
        assert set(m["light"].tolist()) <= set(me["light"].tolist())
        print("item", it, "P", P, "clip", m["voice_clip"], "drawn", m["voice_drawn"], "T", len(m["pid"]), "light", len(m["light"]),
              "cb0", n_cb0, "ratio %.4f" % (len(m["light"]) / max(n_cb0, 1)), "eval light", len(me["light"]), json.loads(m["prov"]).get("clips"))
    out["time_s"] = round(time.time() - t0, 1)
    print("PLANCHECK " + json.dumps(out))
