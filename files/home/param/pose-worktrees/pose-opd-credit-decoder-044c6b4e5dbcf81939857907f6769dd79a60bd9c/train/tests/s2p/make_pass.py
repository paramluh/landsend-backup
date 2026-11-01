#!/usr/bin/env python3
"""pose5 s2p: read every proof's result file in out/ and write ../S2P_PASS (JSON) ONLY if every proof of at least one variant passed;
the JSON lists which variants passed. Prints the per-proof table either way. usage: make_pass.py [--write]"""
import json
import os
import sys

D = "/data/home/amartyavarma/ppm/pose5/code-811-s2p/tests/s2p"
O = f"{D}/out"


def ok(f, key="PASS"):
    p = f"{O}/{f}"
    if not os.path.isfile(p):
        return None
    return bool(json.load(open(p)).get(key))


cfg = json.load(open(f"{O}/CMP_cfg.json")) if os.path.isfile(f"{O}/CMP_cfg.json") else {}
common = {
    "a1_cfg_flag_off_identical": bool(cfg.get("flag_off_cfg_identical") and cfg.get("flag_off_coef_resumekeys_nv_groups_identical")
                                      and cfg.get("flag_off_params_identical")),
    "a2_tiny_c128_orig_vs_copyoff": ok("CMP_tiny_c128_orig_vs_copyoff.json"), "a2_tiny_c1024_orig_vs_copyoff": ok("CMP_tiny_c1024_orig_vs_copyoff.json"),
    "noise_floor_tiny_c128_orig_vs_orig": ok("CMP_tiny_c128_orig_vs_orig.json"), "noise_floor_tiny_c1024_orig_vs_orig": ok("CMP_tiny_c1024_orig_vs_orig.json"),
}
per = {}
for v, cmpk, tz, tnz, real, rtag in (("full", "flag_full", "full0", "fullnz", "full0", ""), ("cb0", "flag_cb0", "cb00", "cb0nz", "cb00", "_cb0")):
    per[v] = {
        "a1_cfg_only_s2p_keys": bool(cfg.get(f"{cmpk}_cfg_only_s2p_keys_added")), "a1_params_only_s_emb": bool(cfg.get(f"{cmpk}_params_only_s_emb_added")),
        "b1_tiny_c128_zero_init": ok(f"CMP_tiny_c128_orig_vs_{tz}.json"), "b1_tiny_c1024_zero_init": ok(f"CMP_tiny_c1024_orig_vs_{tz}.json"),
        "opt_tiny_c128_nonzero_pose_only": ok(f"CMP_tiny_c128_orig_vs_{tnz}.json"), "opt_tiny_c1024_nonzero_pose_only": ok(f"CMP_tiny_c1024_orig_vs_{tnz}.json"),
        "b2_real_width_real_weights_zero_init": ok(f"CMP_real_orig_vs_{real}.json"),
        **{f"c_resume_{ph}": ok(f"resume{rtag}_{ph}.json") for ph in "ABCDEFGHIJKL"},
    }
res = {"common": common, **per}
passed = {v: all(x is True for x in common.values()) and all(x is True for x in per[v].values()) for v in per}
for k, d in res.items():
    print(f"[{k}]")
    for kk, x in d.items():
        print(f"  {'PASS' if x is True else ('MISSING' if x is None else 'FAIL'):8s} {kk}")
print("variants passed:", passed)
if "--write" in sys.argv:
    if any(passed.values()):
        import time
        import hashlib
        C_ = "/data/home/amartyavarma/ppm/pose5/code-811-s2p"
        md5 = lambda f: hashlib.md5(open(f"{C_}/{f}", "rb").read()).hexdigest()  # noqa: E731
        changed = ["model_ppm.py", "train_ppm.py", "ppm811.sbatch", "ppm811f_watchdog.sh", "ppm811f_watchdog.sbatch", "ppm811_watchdog.sh",
                   "ppm811_watchdog.sbatch"]
        exercised = ["safety_ppm.py", "data_ppm.py", "chat_ppm.py", "s811.py", "fresh/fresh_pose.py", "lib/heads_s2q.py", "lib/pose_s3q.py",
                     "lib/speech_q.py", "task_ppm.sh"]
        out = {**passed, "flag_off_byte_identical": all(x is True for x in common.values()),
               "md5_certified_changed_files": {f: md5(f) for f in changed}, "md5_exercised_unchanged_files": {f: md5(f) for f in exercised},
               "written": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "details": res, "tests": f"{D}/TESTS.md"}
        json.dump(out, open(f"{D}/S2P_PASS", "w"), indent=1)
        print("wrote", f"{D}/S2P_PASS")
    else:
        print("NOT written: no variant passed every proof")
