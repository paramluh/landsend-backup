"""plan-1 rows / micros: the pre-append fresh_pose (.bak-pre-append-1022) == the installed one (byte-identical behaviour with FRESH_TAG unset).
Run on a host that holds plan 1 (python -I)."""
import importlib.util, json, os, sys
import numpy as np
C = "/data/home/amartyavarma/ppm/code-811"
sys.path.insert(0, C)
import data_ppm as D  # noqa: E402

def load(path, name):
    sp_ = list(sys.path); sys.path.insert(0, f"{C}/fresh")
    try:
        import importlib.machinery
        ld = importlib.machinery.SourceFileLoader(name, path)
        spec = importlib.util.spec_from_loader(name, ld); m = importlib.util.module_from_spec(spec); ld.exec_module(m)
    finally:
        sys.path[:] = sp_
    return m
old, new = load(f"{C}/fresh/fresh_pose.py.bak-pre-append-1022", "fp_old"), load(f"{C}/fresh/fresh_pose.py", "fp_new")
po, pn = old.load_plan(old.OUT), new.load_plan(new.OUT)
assert po["digest"] == pn["digest"]
cfg = D.default_cfg(pose_delay=2)
picks = [0, 1, pn["n_rows"] // 2, pn["n_rows"] - 1] + [int(x) for x in sys.argv[1:]]
for i in picks:
    a, b = old.FreshRows(po)[i], new.FreshRows(pn)[i]
    assert a.keys() == b.keys(), (a.keys(), b.keys())
    for k in a:
        assert (np.array_equal(a[k], b[k]) if isinstance(a[k], np.ndarray) else json.dumps(a[k], sort_keys=True, default=str) == json.dumps(b[k], sort_keys=True, default=str)), (i, k)
    ma = old.FreshPoseDS(po, [i], cfg, 3)[0]; mb = new.FreshPoseDS(pn, [i], cfg, 3)[0]
    mc = new.FreshPoseMultiDS([pn], [i], cfg, 3)[0]                       # the multi-plan dataset on plan-1 ids == FreshPoseDS
    for k in ma:
        for m_ in (mb, mc):
            assert (np.array_equal(ma[k], m_[k]) if isinstance(ma[k], np.ndarray) else json.dumps(ma[k], default=str) == json.dumps(m_[k], default=str)), (i, k)
print(json.dumps({"host": new.HOST, "digest": pn["digest"], "rows_checked": picks, "row_key_0": new.FreshRows(pn)[0]["row_key"], "eq": True}))
