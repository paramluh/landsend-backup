#!/usr/bin/env python3
"""posefix: replay vs live session: first divergence of B's lane / Mimi / pose tokens, equality fractions, cb0 change rates.
usage: python cmp_repro.py <session dir> <sim npz>"""
import json
import sys

import numpy as np

sd, sim = sys.argv[1], sys.argv[2]
a, b = np.load(f"{sd}/session.npz"), np.load(sim)
out = {}
for k in ("tok", "mimi", "pose"):
    x, y = a[k], b[k]
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    eq = (x == y) if x.ndim == 1 else (x == y).all(-1)
    bad = np.flatnonzero(~eq)
    out[k] = {"n": int(n), "eq_frac": float(eq.mean()), "first_diff": int(bad[0]) if len(bad) else None}
for name, z in (("live", a), ("sim", b)):
    p0 = z["pose"][:, 0].astype(int)
    out[f"{name}_cb0_change"] = float((p0[1:] != p0[:-1]).mean())
    out[f"{name}_cb0_distinct"] = int(len(np.unique(p0)))
print(json.dumps(out))
