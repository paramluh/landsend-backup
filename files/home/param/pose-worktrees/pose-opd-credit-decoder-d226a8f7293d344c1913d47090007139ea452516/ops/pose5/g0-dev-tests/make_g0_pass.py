#!/usr/bin/env python3
"""pose5 D: machine-checkable G0COND marker. Reads the comparison / phase JSONs of an out dir and writes <dir>/../G0_PASS (JSON) ONLY if every
required check passed (else prints what is missing / failing and writes nothing).

  make_g0_pass.py single <out dir>   G0COND alone in code-811-g0 (cfg, micros, f sanity, tiny zero / non-zero, resume A..I, real-width zero
                                     (+ non-zero if present))
  make_g0_pass.py combo  <out dir>   merged code-811-s2p: cfg for every combination, tiny zero-init per combination + non-zero combos,
                                     resume combos (C / D per combination + refusals), real-width zero-init per combination present"""
import glob
import json
import os
import sys
import datetime
import zoneinfo

mode, out = sys.argv[1], sys.argv[2].rstrip("/")


def ok(path, key="PASS"):
    try:
        return bool(json.load(open(path)).get(key))
    except (OSError, ValueError):
        return None


checks = {}
if mode == "single":
    req = {"cfg_flag_off_identical_flag_on_only_g0": "CMP_cfg.json", "real_micros_byte_checks": "micros.json",
           "tiny_flag_off_vs_orig_bitidentical": "CMP_tiny_orig_vs_copyoff.json", "tiny_zero_init_vs_orig_identical": "CMP_tiny_orig_vs_on0.json",
           "tiny_nonzero_pose_only": "CMP_tiny_orig_vs_nz_stfirst.json", "real_width_zero_init_vs_orig_identical": "CMP_real_orig_vs_on0.json"}
    req.update({f"resume_phase_{p}": f"resume_{p}.json" for p in "ABCDEGHI"})
    opt = {"real_width_nonzero_pose_only": "CMP_real_orig_vs_nz_stfirst.json"}
    for k, f in req.items():
        checks[k] = ok(f"{out}/{f}")
    for k, f in opt.items():
        if os.path.exists(f"{out}/{f}"):
            checks[k] = ok(f"{out}/{f}")
    fs = json.load(open(f"{out}/f_stats.json"))
    checks["f_sane_plan1_npl5"] = all(fs[s]["frac_invalid"] == 0.0 and fs[s]["has_orig_hw_all"] and fs[s]["clips"] > 10000 for s in ("plan1", "npl5"))
    combos = ["g0"]
else:
    checks["cfg_all_combinations"] = ok(f"{out}/CMP_cfg_combo.json")
    for c in ("off", "g0", "cb0", "full", "g0cb0", "g0full", "g0cb0_nz", "g0full_nz"):
        checks[f"tiny_{c}"] = ok(f"{out}/CMP_tiny_{c}.json")
    for t in ("g01_s2pnone", "g00_s2pcb0", "g00_s2pfull", "g01_s2pcb0", "g01_s2pfull"):
        for p in ("C", "D"):
            checks[f"resume_{p}_{t}"] = ok(f"{out}/resume_combo_{p}_{t}.json")
    checks["resume_A_standin"] = ok(f"{out}/resume_combo_A_g00_s2pnone.json")
    xs = sorted(glob.glob(f"{out}/resume_combo_X_*.json"))
    checks["resume_refusals"] = bool(xs) and len(xs) >= 5 and all(ok(x) for x in xs)
    for c in ("g0", "cb0", "full", "g0cb0", "g0full"):
        p = f"{out}/CMP_real_{c}.json"
        checks[f"real_width_{c}"] = ok(p)
    combos = ["g0", "cb0", "full", "g0+cb0", "g0+full"]
bad = {k: v for k, v in checks.items() if v is not True}
print(json.dumps(checks, indent=1))
if bad:
    print("NOT PASSED:", bad)
    sys.exit(1)
marker = os.path.join(os.path.dirname(out), "G0_PASS")
res = {"PASS": True, "mode": mode, "out": out, "time_pt": datetime.datetime.now(zoneinfo.ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S PT"),
       "combinations_passed": combos, "checks": checks}
json.dump(res, open(marker, "w"), indent=1)
print("wrote", marker)
