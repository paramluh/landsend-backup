"""PERSONA CPU test (no GPU, no launch). usage: python -I -B identity.py <code dir> <module> <out json> [report]
Builds, exactly as the trainer at physical world 64 (1 slot / GPU), every logical rank's step-0 train element (AccDS[0]: micro + its
'singles' loss-check micro + the A waveform) and its first eval micro, and writes sha256 digests of every array / field + the plan digest.
<module> = data_aloe_v2vox (job 29345's path, run code dir) or data_aloe_persona (persona/code; knobs from env).
report: + (persona module) the plan-wide persona report, shift checks of the step-0 items vs their no-persona micros, and the 5-row dump."""
import hashlib
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

CODE, MOD, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
REPORT = len(sys.argv) > 4 and sys.argv[4] == "report"
sys.path.insert(0, CODE)
os.environ["ALOE_FLAC_ROOT"] = os.environ["ALOE_SHARED_FLAC"]      # CPU node: the shared copy the per-host symlink points to
A = __import__(MOD)
assert os.path.dirname(os.path.abspath(A.__file__)) == os.path.abspath(CODE), (A.__file__, CODE)
LW = 64


def dig(v):
    if isinstance(v, np.ndarray):
        h = hashlib.sha256(); h.update(str(v.dtype).encode()); h.update(str(v.shape).encode()); h.update(np.ascontiguousarray(v).tobytes())
        return h.hexdigest()[:20]
    if isinstance(v, (list, tuple)) and v and isinstance(v[0], dict):
        return [digm(x) for x in v]
    return repr(v)


def digm(m):
    return {k: dig(v) for k, v in sorted(m.items())}


PL = A.make_plan(LW)
if hasattr(A, "persona_setup"):
    A.persona_setup(PL)


def one(L):
    tr, ev = A.rank_items(PL, L, LW)
    ds = A.AccDS([tr[:249]], [L], A.aloe_cfg(), PL["items"][0], singles_at=0)
    m = ds[0]
    e = A.eval_micros(ev[:1], A.eval_cfg(), L)[0]
    return L, {"item0": list(tr[0]), "train": digm(m), "eval_item0": list(ev[0]), "eval": digm(e),
               "persona": [int(m.get("persona_block", 0)), int(e.get("persona_block", 0))]}


if __name__ == "__main__":
    out = {"module": A.__file__, "plan_digest": hashlib.sha256(json.dumps(PL["items"]).encode()).hexdigest()[:16],
           "n_items": len(PL["items"]), "n_eval": len(PL["eval"]), "steps": PL["steps"]}
    with Pool(int(os.environ.get("NPROC", "16"))) as pool:
        out["ranks"] = dict(pool.map(one, range(LW)))
    if REPORT:
        cfg = A.aloe_cfg()
        rep = A.summarize_report(A.plan_report(PL["items"], cfg), len(PL["items"]))
        rep_e = A.summarize_report(A.plan_report(PL["eval"], A.eval_cfg()), len(PL["eval"]))
        out["persona_report"] = {k: v for k, v in rep.items()}
        out["persona_report_eval_items_if_drawn"] = rep_e["text"]
        st = A.Stores()
        sc = []
        for L in range(LW):                                         # shift check of every step-0 train item vs its no-persona micro
            it = PL["items"][L]
            m0 = st.conv(it, cfg, L, audio=False, persona=False)
            m1 = st.conv(it, cfg, L, audio=False)
            sc.append([L, int(m1["persona_block"]), int(A.shift_check(m0, m1)), int(m1["P"]) - int(m0["P"])])
        out["shift_checks"] = sc
        txt = A.render_dump(PL["items"], cfg, 0, os.environ["DUMP_PATH"])
        out["dump_grep"] = {w: txt.count(w) for w in ("Persona", "Starting context", "Earlier conversation memory", "persona", "starting")}
        print(rep["text"])
    json.dump(out, open(OUT, "w"), indent=1)
    print("WROTE", OUT, out["plan_digest"])
