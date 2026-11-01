"""POSEDUP CPU test (no GPU, no launch). usage: python -I -B identity.py <code dir> <module> <out json> [report]
Builds, exactly as the trainer at physical world 64 (1 slot / GPU), every logical rank's step-0 train element (AccDS[0]: micro + its
'singles' loss-check micro + the A waveform) and its first eval micro, and writes sha256 digests of every array / field + the plan digest.
<module> = data_aloe_v2vox (job 29345's path, its run code dir) or data_aloe_posedup (poseduplex/code; knobs from env).
report (posedup module, POSEDUP_POSE=1): + per step-0 / eval item: the pose checks against the pose token FILE (pdepth at P + CD + f == the
file's codes of frame f + cb offsets, the same positions as B's Mimi frames, plight == every pose position at frac 1.0, pcb0 labels), the
multi-epoch plan checks (QQA_STEPS steps: epoch 0 == 29345's items, later epochs = permutations of the same items, dummies only at epoch
ends), the plan report, and the 3-row rendered dump (DUMP_PATH)."""
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
POSEDUP = MOD == "data_aloe_posedup"


def dig(v):
    if isinstance(v, np.ndarray):
        h = hashlib.sha256(); h.update(str(v.dtype).encode()); h.update(str(v.shape).encode()); h.update(np.ascontiguousarray(v).tobytes())
        return h.hexdigest()[:20]
    if isinstance(v, (list, tuple)) and v and isinstance(v[0], dict):
        return [digm(x) for x in v]
    return repr(v)


def digm(m):
    d = {k: dig(v) for k, v in sorted(m.items())}
    if "own" in m and np.asarray(m["own"]).shape[1] == 5:           # posedup pose micro: the 29345 own counts are its first 3 columns
        d["own3"] = dig(np.ascontiguousarray(np.asarray(m["own"])[:, :3]))
    return d


PL = A.make_plan(LW)


def one(L):
    tr, ev = A.rank_items(PL, L, LW)
    ds = A.AccDS([tr[:249]], [L], A.aloe_cfg(), PL["items"][0], singles_at=0)
    m = ds[0]
    e = A.eval_micros(ev[:1], A.eval_cfg(), L)[0]
    return L, {"item0": list(tr[0]), "train": digm(m), "eval_item0": list(ev[0]), "eval": digm(e),
               "pose_ok": [int(m.get("pose_ok", -1)), int(e.get("pose_ok", -1))], "kind": [m["kind"], e["kind"]]}


def pose_check(it, cfg, L):
    """the pose arrays of item it vs the pose token file itself (independent of pose_arrays' own asserts)."""
    st = A.Stores()
    m = st.conv(tuple(it), cfg, L, audio=False)
    e = A.pose_entry(it)
    P, n = int(m["P"]), int(json.loads(m["prov"])["n_frames"])
    pd = m["pdepth"]
    off = np.arange(32, dtype=np.int64) * 2048
    r = {"item": list(it), "pose_ok": int(m["pose_ok"]), "P": P, "n": n}
    if e["ok"]:
        f = np.load(f"{A.pose_index()['data_root']}/{e['path']}")
        pos = P + 2 + np.arange(n)
        r["file_eq"] = bool(np.array_equal(pd[pos] - off, f[:, :n].T.astype(np.int64)))
        r["only_there"] = int((pd[:, 0] >= 0).sum()) == n and bool((pd[np.setdiff1d(np.arange(len(pd)), pos)] < 0).all())
        db = np.flatnonzero(np.asarray(m["depth"][1])[:, 0] >= 0)
        r["mimi_pos_eq"] = bool(np.array_equal(pos, db[db >= P]))
        r["plight_all"] = bool(np.array_equal(np.sort(m["plight"]), pos)) if A.POSE_LIGHT_FRAC == 1.0 else len(m["plight"])
        r["pcb0_eq"] = bool(np.array_equal(m["pcb0"][pos], f[0, :n].astype(np.int64))) and int((m["pcb0"] >= 0).sum()) == n
    else:
        r["empty"] = bool((pd < 0).all()) and len(m["plight"]) == 0 and bool((m["pcb0"] < 0).all())
    m0 = st.conv(tuple(it), cfg, L, audio=False, pose=False, rule="v2_30")
    r["run_check"] = A.run_check(m0, m)
    return r


if __name__ == "__main__":
    items0 = PL["items0"] if POSEDUP else PL["items"]
    out = {"module": A.__file__, "plan_digest": hashlib.sha256(json.dumps([list(x) for x in items0]).encode()).hexdigest()[:16],
           "n_items": len([x for x in PL["items"] if x is not None]), "n_eval": len(PL["eval"]), "steps": PL["steps"]}
    with Pool(int(os.environ.get("NPROC", "16"))) as pool:
        out["ranks"] = dict(pool.map(one, range(LW)))
    if REPORT:
        assert POSEDUP and A.POSE
        cfg, ecfg = A.aloe_cfg(), A.eval_cfg()
        jobs = [(PL["items"][L], cfg, L) for L in range(LW)] + [(A.rank_items(PL, L, LW)[1][0], ecfg, L) for L in range(LW)]
        with Pool(int(os.environ.get("NPROC", "16"))) as pool:
            pc = pool.starmap(pose_check, jobs)
        out["pose_checks"] = pc
        keys = ("file_eq", "only_there", "mimi_pos_eq", "plight_all", "pcb0_eq")
        out["pose_checks_summary"] = {"n": len(pc), "pose_ok": sum(r["pose_ok"] for r in pc),
                                      "all_pass_ok": all(all(r[k] for k in keys) for r in pc if r["pose_ok"]),
                                      "all_pass_empty": all(r["empty"] for r in pc if not r["pose_ok"]), "run_check_all": all(r["run_check"] for r in pc)}
        S = int(os.environ.get("QQA_STEPS", "498"))
        PM = A.make_plan(LW, steps=S)
        E = PM["epoch_steps"]
        ep = [[x for x in PM["items"][e * E * LW:(e + 1) * E * LW] if x is not None] for e in range(-(-S // E))]
        out["multi_epoch"] = {"steps": S, "epoch_steps": E, "digest": A.plan_digest(PM)[0], "epoch0_eq_29345": ep[0] == PL["items0"],
                              "later_epochs_perm": all(sorted(x) == sorted(PL["items0"][:len(x)]) or (len(x) < len(PL["items0"]) and set(x) <= set(PL["items0"]))
                                                       for x in ep[1:]),
                              "later_epochs_differ": all(x != PL["items0"][:len(x)] for x in ep[1:]),
                              "dummy_positions_ok": all(PM["items"][t] is None for t in range(len(PM["items"])) if (t % (E * LW)) >= len(PL["items0"])),
                              "n_train_micros": PM["n_train_micros"], "n_dummy": PM["n_dummy"]}
        out["plan_report"] = A.plan_report(PM)
        txt = A.render_dump(PL["items"], cfg, os.environ["DUMP_PATH"])
        out["dump_lines"] = txt.count("\n")
        print(json.dumps({k: v for k, v in out.items() if k not in ("ranks", "pose_checks")}, indent=1))
    json.dump(out, open(OUT, "w"), indent=1)
    print("WROTE", OUT, out["plan_digest"])
