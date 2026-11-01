#!/usr/bin/env python3
"""Input parity: infer.py's prompt-package path (package + words.json + voice clip codes) must build exactly the teacher-forced
prefix [0, P + F0) of a reference Aloe-format row (tonight's studio rows, rows-studio/S-*.pkl). CPU only.
   python tests/check_package_inputs.py PACKAGE_DIR EXTRAS_DIR REF_ROW.pkl [--prefill-s 2]"""
import argparse
import json
import os
import pickle
import sys
import types

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "ppm"))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import infer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("package")
ap.add_argument("extras")
ap.add_argument("ref")
ap.add_argument("--prefill-s", type=float, default=2.0)
c = ap.parse_args()
a = types.SimpleNamespace(package=c.package, mode="all", no_voice_clip=False, voice_clip=f"{c.extras}/voice_clip_mimi.safetensors",
                          words=f"{c.extras}/words.json", no_prefill_words=False, first_word_space=False, force_audio=None, persona=None)
npf = int(round(c.prefill_s * 12.5))
x, n, info, f0, clip = infer.package_inputs(a, None, 750, npf)
r = pickle.load(open(c.ref, "rb"))
P = int(x["P"])
Np = P + npf + 2
res = {"P": P, "P_ref": int(r["P"])}
assert P == int(r["P"]), res
for k in ("ids", "depth", "present"):
    res[k] = bool(np.array_equal(np.asarray(x[k])[:, :Np], np.asarray(r[k])[:, :Np]))
res["pdepth"] = bool(np.array_equal(x["pdepth"][:Np], np.asarray(r["pdepth"])[:Np]))
res["prefix_positions"] = Np
print(json.dumps(res))
assert all(res[k] for k in ("ids", "depth", "present", "pdepth")), res
print("INPUT PARITY OK")
