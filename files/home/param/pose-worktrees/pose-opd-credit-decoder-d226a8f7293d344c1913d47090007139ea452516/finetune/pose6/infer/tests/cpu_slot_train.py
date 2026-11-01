#!/usr/bin/env python3
"""pose6 A1 CPU check (training side of the SLOT): what the monoplex FT trainer (pose-s2p/code = mono8020 + Rosie fixes + s2p; data path
unchanged since mono8020) puts at P - 1 for real Aloe eval items of every prompt type (empty block / voice clip / persona / both), and the
real micros the GPU parity test (iii) runs on. Env = aloe_mono8020.env (the export-357 recipe's data cfg). Results -> tests/out/cpu_slot_train.json,
micros -> tests/out/micros/<type>.pkl (ids, depth, present, pdepth, P, row_key, item, prov: exactly Stores.conv's arrays).
If B1's pose6/code data_aloe_pose already emits g0pos, its single-item micro's g0pos is checked == [P - 1] too (else recorded as pending)."""
import json
import os
import pickle
import sys
import time

import numpy as np

C6 = "/data/home/amartyavarma/ppm/pose6/code"
CS = "/data/home/amartyavarma/ppm/pose-s2p/code"
OUT = "/data/home/amartyavarma/ppm/pose6/infer/tests/out"
C = CS
for p in (f"{C}/vendor", f"{C}/lib", C):
    sys.path.insert(0, p)
import data_aloe_pose as A  # noqa: E402
import train_aloe_pose as T  # noqa: E402   (module level only: CFG; no dist, no GPU)

T0 = time.time()
NL, IM_END = 198, 151645


def log(*a):
    print(f"[a1-slot {time.time() - T0:7.1f}s]", *a, flush=True)


cfg = T.CFG["data"]
RES = {"code": C, "host": os.uname().nodename, "t_start": time.strftime("%Y-%m-%d %H:%M:%S"),
       "pose6_code_has_g0pos": "g0pos" in open(f"{C6}/data_aloe_pose.py").read()}
sp = A.splits_v2()
ev = sorted(sp["eval"], key=lambda x: (x[3], x[4]))
lo, hi = int(os.environ.get("A1_MIN_N", "320")), int(os.environ.get("A1_MAX_N", "100000"))
st = A.Stores()
found, scanned, rule_ok, rule_bad = {}, 0, 0, []
os.makedirs(f"{OUT}/micros", exist_ok=True)
for it in ev:
    if it[3] < lo or it[3] > hi:
        continue
    m = st.conv(it[:3], cfg, 0)
    scanned += 1
    P = int(m["P"])
    pv = json.loads(m["prov"])
    typ = {(False, False): "empty", (True, False): "clip", (False, True): "persona", (True, True): "clip+persona"}[
        (len(pv.get("clips", [])) > 0, len(pv.get("persona", [])) > 0)]
    ids, depth, present, pdepth = (np.asarray(m[k]) for k in ("ids", "depth", "present", "pdepth"))
    s = P - 1
    cell = {"P": P, "ids_B[P-2:P]": ids[1, P - 2:P].tolist(), "ids_A[P-1]": int(ids[0, s]), "present[:, P-1]": present[:, s].tolist(),
            "depth_codes_at_P-1": int((depth[:, s] >= 0).sum()), "pdepth_codes_at_P-1": int((pdepth[s] >= 0).sum()),
            "first_pose_pos": int(np.flatnonzero(pdepth[:, 0] >= 0)[0]), "first_conv_speech_pos": int(P + np.flatnonzero(depth[1, P:, 0] >= 0)[0]),
            "voice_clip_code_positions": int((depth[1, :P, 0] >= 0).sum())}
    ok = (cell["ids_B[P-2:P]"] == [IM_END, NL] and cell["present[:, P-1]"] == [0.0, 1.0] and cell["depth_codes_at_P-1"] == 0
          and cell["pdepth_codes_at_P-1"] == 0 and cell["first_pose_pos"] == P + 2)
    rule_ok += int(ok)
    if not ok:
        rule_bad.append([list(it[:3]), cell])
    if typ not in found:
        cell.update(type=typ, item=list(it[:3]), row_key=it[4], n_frames=int(it[3]), N=int(len(m["pid"])), prov=pv)
        found[typ] = cell
        mic = {k: np.asarray(m[k]) for k in ("ids", "depth", "present", "pdepth", "pid")}
        mic.update(P=P, row_key=it[4], item=list(it[:3]), prov=pv, type=typ, n_frames=int(it[3]))
        pickle.dump(mic, open(f"{OUT}/micros/{typ}.pkl", "wb"))
        log(f"{typ}: {json.dumps({k: v for k, v in cell.items() if k != 'prov'})}")
    if len(found) == 4 and scanned >= int(os.environ.get("A1_MIN_SCAN", "100000")):
        break
RES.update(found=found, scanned=scanned, slot_rule_ok=rule_ok, slot_rule_bad=rule_bad[:20], n_frames_window=[lo, hi])
RES["pass"] = rule_ok == scanned and len(found) == 4
RES["t_end"] = time.strftime("%Y-%m-%d %H:%M:%S")
json.dump(RES, open(f"{OUT}/cpu_slot_train.json", "w"), indent=1, default=str)
log(f"scanned {scanned} items in [{lo}, {hi}] frames: P-1 rule held on {rule_ok}; types found {sorted(found)}; pass {RES['pass']}")
