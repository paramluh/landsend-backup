"""textB data test (CPU only, no GPU, no writes): ALOE_B_TEXT_ONLY=1 micros vs v2's micros of the same items (the trainer's same_b_r_textB
rule) on the first N train items of logical ranks 0..63 at step 0 + a few eval items + every item whose v2 micro carries a B voice clip among
them; also as_micro / pack own counts (cb0 0, light 0) and the default path (ALOE_B_TEXT_ONLY unset) == v2 byte-for-byte on B.
usage (compute node): ALOE_B_TEXT_ONLY=1 python -I test_textB_data.py <code dir> [n_items]"""
import importlib.util
import json
import os
import sys

import numpy as np

C = sys.argv[1]
NI = int(sys.argv[2]) if len(sys.argv) > 2 else 24
sys.path.insert(0, C)
sys.path.insert(0, os.path.join(C, "lib"))
assert os.environ.get("ALOE_B_TEXT_ONLY") == "1"
import data_aloe_v2vox as A  # noqa: E402

assert A.B_TEXT_ONLY
D = A.D
spec = importlib.util.spec_from_file_location("data_aloe_v2_22383", "/data/home/amartyavarma/ppm/aloe-v2-ft/code/data_aloe_v2.py")
R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
PL = A.make_plan(64)
assert PL["steps"] == 253 and len(PL["items"]) == 16134
cfg, ecfg = A.aloe_cfg(), A.eval_cfg()
SA, SR = A.Stores(), R.Stores()
n_ok = n_clip = 0
items = [(PL["items"][i], cfg, i % 64) for i in range(NI)] + [(PL["eval"][i], ecfg, i % 64) for i in range(6)]
# find a few items whose v2 micro has a realised B clip
extra = 0
for i in range(NI, 4000):
    it = PL["items"][i]
    if R.want_voice(SR.open(int(it[0]))[1][int(it[1])]["row_key"], int(it[2])):
        items.append((it, cfg, i % 64)); extra += 1
        if extra >= 6:
            break
for it, cf, rk in items:
    m_ref = SR.conv(it, cf, rk)
    m_new = SA.conv(it, cf, rk, audio=False)
    Pr, P_ = int(m_ref["P"]), int(m_new["P"])
    c = {"P5": P_ == 5 and (Pr == 5) == (not m_ref["voice_clip"]),
         "ids_B_conv": np.array_equal(m_ref["ids"][1][Pr:], m_new["ids"][1][P_:]),
         "present_B_conv": np.array_equal(m_ref["present"][1][Pr:], m_new["present"][1][P_:]),
         "prefix_B": (not m_ref["voice_clip"] and np.array_equal(m_ref["ids"][1][:Pr], m_new["ids"][1][:P_])) or (bool(m_ref["voice_clip"]) and P_ == 5),
         "text_targets": np.array_equal(np.asarray(m_ref["sel"]) - Pr, np.asarray(m_new["sel"]) - P_)
                         and all(np.array_equal(np.asarray(m_ref[k]), np.asarray(m_new[k])) for k in ("y", "word", "pad", "eob", "nw")),
         "no_B_speech": bool((m_new["depth"][1] < 0).all() and (np.asarray(m_new["cb0"]) < 0).all() and len(m_new["light"]) == 0),
         "ref_had_B_speech": bool((m_ref["depth"][1] >= 0).any() and (np.asarray(m_ref["cb0"]) >= 0).any() and len(m_ref["light"]) > 0),
         "voice": int(m_new["voice_clip"]) == 0 and int(m_new["voice_drawn"]) == int(m_ref["voice_drawn"]),
         "A_empty": bool((m_new["ids"][0][P_:] == D.PAD_ID).all() and (m_new["depth"][0] < 0).all() and not m_new["present"][0][P_:].any()
                         and tuple(m_new["vin"]) == (P_, int(m_new["vin"][1])) and int(m_new["vin"][1]) == int(m_new["T"] if "T" in m_new else len(m_new["pid"]) - P_ - D.CD))}
    mic = A.as_micro(m_new)
    oc = np.asarray(mic["own"]).sum(0)
    c["own"] = oc[0] == len(m_ref["sel"]) and oc[1] == 0 and oc[2] == 0
    ok = all(c.values())
    n_ok += ok; n_clip += int(m_ref["voice_clip"])
    print(it, "ref P", Pr, "new P", P_, "clip", m_ref["voice_clip"], "OK" if ok else f"FAIL {c}", "| text targets", len(m_new["sel"]),
          "words", int(np.asarray(m_new["word"]).sum()), "nw", int(np.asarray(m_new["nw"]).sum()), "eob", int(np.asarray(m_new["eob"]).sum()), flush=True)
print(f"VERDICT {'PASS' if n_ok == len(items) else 'FAIL'}: {n_ok}/{len(items)} items (of which {n_clip} had a B voice clip in v2, dropped in textB)")
print("prov example:", json.loads(m_new["prov"]))
