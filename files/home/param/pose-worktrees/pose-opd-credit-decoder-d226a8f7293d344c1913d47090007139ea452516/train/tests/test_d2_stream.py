"""CPU check of the D2 speech stream on THIS host, all 8 local ranks: head unchanged, no repeats, nothing skipped (stream + holdout drops =
native + extra), holdout rows really gone, a few rows of every slot build. usage: test_d2_stream.py PLAN S0 [SALTS]"""
import json, os, socket, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PPM_CD", "2")
import data_ppm as D
from extra_digest import CFG
H = socket.gethostname().split(".")[0]
plan, s0 = D.load_extra_plan(sys.argv[1]), int(sys.argv[2])
salts = [x for x in (sys.argv[3] if len(sys.argv) > 3 else "").split(",") if x]
hp = plan["hosts"][H]
idx = D.DS.load_index("/scratch/amartyavarma/s3-rows"); node = D.R3.load_node("/scratch/amartyavarma/s3-rows")
IDX = sorted(plan["hosts"], key=lambda h: int(h.split("-")[1])).index(H)
hold_n = np.array([any(D.DS.is_eval_episode(D.DS.episode_key(a, b, c), 0.0005, s) for s in salts) for a, b, c in zip(idx["dataset"], idx["podcast_id"], idx["episode_id"])], bool)
hold_x = {k + 1: np.array([any(D.DS.is_eval_episode(e, 0.0005, s) for s in salts) for e in D.extra_episode_keys(sl)], bool) for k, sl in enumerate(hp["slots"])}
res = []
ds = D.MixedSpeechDS("/scratch/amartyavarma/s3-rows", hp, [], CFG, 0)
for l in range(8):
    p = D.speech_plan(node, idx, 20261011, 8, l, 0.0005, "ppm-v1", 2)
    nat = np.asarray(p["rows"], np.int64)
    rows, xs = D.d2_stream(nat, idx, hp, l, 8 * IDX + l, s0, 2, salts, 0.0005, 20261011)
    ext = np.asarray([D.code_of(k + 1, r) for k, r in hp["ranks"][l]], np.int64)
    assert np.array_equal(rows[:2 * s0], nat[:2 * s0]), "head changed"
    assert len(set(rows.tolist())) == len(rows), "a row repeats"
    tail_n = nat[2 * s0:]
    want = set(tail_n[~hold_n[tail_n]].tolist()) | set(c for c in ext.tolist() if not hold_x[D.split_code(c)[0]][D.split_code(c)[1]])
    assert set(rows[2 * s0:].tolist()) == want, "tail != native left + extra - holdout"
    for c in rows[2 * s0:].tolist():
        sl, r = D.split_code(c)
        assert not (hold_n[r] if sl == 0 else hold_x[sl][r]), "holdout row in the tail"
    # a few rows of every slot build (first 2 of each slot in the shuffled tail)
    seen = {}
    for c in rows[2 * s0:].tolist():
        sl = D.split_code(c)[0]
        if seen.get(sl, 0) < 2:
            m = ds.get(c); assert m["ids"].shape[1] == 7502; seen[sl] = seen.get(sl, 0) + 1
    res.append({"local": l, "rank": 8 * IDX + l, "rows": len(rows), "steps": len(rows) // 2, **xs, "built": seen})
print(json.dumps({"host": H, "s0": s0, "salts": salts, "ranks": res}))
