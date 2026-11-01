"""bit-exact row digests of PPM speech micros (D2 extra slices test).
usage: extra_digest.py native ROOT N OUT.json          sample N FULL train rows of the store at ROOT (deterministic), digest them read natively
       extra_digest.py extra  PLAN.npz HOST IN.json OUT.json   re-read the IN rows that PLAN hosts on HOST through the extra-slice path
Every micro array is hashed (dtype, shape, bytes) + the prov string. rank argument = 0 for every row (only the light seed uses it)."""
import hashlib
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.environ.setdefault("PPM_CD", "2")
import data_ppm as D  # noqa: E402

CFG = D.default_cfg(seed=20261011, sp_cap=2000, persona_sides="both", light_frac=0.1, pose_light_frac=0.1, txt_cap=7500, crop_conv=True,
                    chat_side="assistant", chat_eob=True, pack_attn="causal")
SALT, FRAC = "ppm-v1", 0.0005


def digest(m):
    out = {}
    for k in sorted(m):
        v = m[k]
        if isinstance(v, np.ndarray):
            out[k] = f"{v.dtype}|{v.shape}|" + hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
        else:
            out[k] = json.dumps(v, sort_keys=True, default=str)
    return out


_DS = {}


def work(a):
    key, root, hp, code = a
    if key not in _DS:
        _DS[key] = D.MixedSpeechDS(root, hp, [], CFG, 0)
    return D.split_code(code)[1], digest(_DS[key].get(code))


def main():
    mode = sys.argv[1]
    if mode == "native":
        root, n, out = sys.argv[2], int(sys.argv[3]), sys.argv[4]
        node = D.R3.load_node(root)
        idx = D.DS.load_index(root)
        ev = np.array([D.DS.is_eval_episode(D.DS.episode_key(a, b, c), FRAC, SALT) for a, b, c in zip(idx["dataset"], idx["podcast_id"], idx["episode_id"])])
        cand = np.flatnonzero((node["length"] == 7500) & ~ev)
        rng = np.random.default_rng(12345)
        rows = sorted(int(x) for x in rng.choice(cand, n, replace=False))
        with Pool(min(16, n)) as p:
            res = p.map(work, [("n", root, None, r) for r in rows], chunksize=1)
        json.dump({"node": node["node"], "root": root, "host": os.uname().nodename, "rows": {str(r): d for r, d in res}}, open(out, "w"))
        print("native", node["node"], len(res), "rows ->", out)
    else:
        plan, host, inp, out = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
        ref = json.load(open(inp))
        P = D.load_extra_plan(plan)
        hs = P["hosts"][host]
        slot = [k for k, s in enumerate(hs["slots"]) if s["node"] == ref["node"]]
        assert slot, ("host does not carry this slice", host, ref["node"])
        sl = hs["slots"][slot[0]]
        mine = set(int(r) for r in sl["rows"])
        rows = [int(r) for r in ref["rows"] if int(r) in mine]
        with Pool(min(16, max(1, len(rows)))) as p:
            res = p.map(work, [("x", None, hs, D.code_of(slot[0] + 1, r)) for r in rows], chunksize=1)
        bad = [r for r, d in res if d != ref["rows"][str(r)]]
        for r, d in res:
            if d != ref["rows"][str(r)]:
                print("DIFF row", r, [k for k in d if d[k] != ref["rows"][str(r)].get(k)])
        json.dump({"node": ref["node"], "host": host, "n": len(res), "bad": bad}, open(out, "w"))
        print(f"extra {ref['node']} on {host}: {len(res)} rows compared, {len(bad)} differ")


if __name__ == "__main__":
    main()
