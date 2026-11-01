#!/usr/bin/env python3
"""pose5 D: the two ARMED markers for the MERGED code-811-s2p, written ONLY if every proof passed (else prints what is missing and exits 1).

  code-811-s2p/tests/g0/G0_PASS   {"PASS", "groups": ["g0cond"], "core_md5", "combinations_passed", "checks", ...}
  code-811-s2p/tests/s2p/S2P_PASS re-issued after the merge (A's keys "full", "cb0", "flag_off_byte_identical" kept; "groups": ["s2p"];
                                  "core_md5"; details = A's suite re-run on the merged code (tests/s2p_postmerge) + the combined proofs;
                                  A's pre-merge marker kept as tests/s2p/S2P_PASS.premerge-A)
core_md5 = md5 of the files in code-811-s2p NOW (model_ppm, train_ppm, data_ppm, safety_ppm, s811, chat_ppm, fresh/fresh_pose, lib/pose_s3q,
lib/codep_s3q, lib/heads_s2q, lib/speech_q + A's launch scripts); also checks that the test runs used these exact files (md5 recorded at run
time is not available for every run, so the files' mtimes must predate the first proof job of the final wave: FINAL_SINCE)."""
import datetime
import glob
import hashlib
import json
import os
import shutil
import sys
import zoneinfo

S = "/data/home/amartyavarma/ppm/pose5/code-811-s2p"
G = f"{S}/tests/g0/out"
P = f"{S}/tests/s2p_postmerge/out"
CORE = ["model_ppm.py", "train_ppm.py", "data_ppm.py", "safety_ppm.py", "s811.py", "chat_ppm.py", "fresh/fresh_pose.py", "lib/pose_s3q.py",
        "lib/codep_s3q.py", "lib/heads_s2q.py", "lib/speech_q.py", "ppm811.sbatch", "ppm811f_watchdog.sh", "ppm811f_watchdog.sbatch",
        "ppm811_watchdog.sh", "ppm811_watchdog.sbatch", "task_ppm.sh"]
FINAL_SINCE = float(os.environ.get("FINAL_SINCE", "0"))             # epoch seconds: start of the final proof wave


def ok(path, key="PASS"):
    try:
        return bool(json.load(open(path)).get(key))
    except (OSError, ValueError):
        return None


md5 = {f: hashlib.md5(open(f"{S}/{f}", "rb").read()).hexdigest() for f in CORE}
late = [f for f in CORE if os.path.getmtime(f"{S}/{f}") > FINAL_SINCE] if FINAL_SINCE else []
g = {}
g["code_unchanged_since_final_wave"] = not late
g["cfg_all_combinations"] = ok(f"{G}/CMP_cfg_combo.json")
for c in ("off", "g0", "cb0", "full", "g0cb0", "g0full", "g0cb0_nz", "g0full_nz"):
    g[f"tiny_{c}"] = ok(f"{G}/CMP_tiny_{c}.json")
for c in ("g0", "cb0", "full", "g0cb0", "g0full"):
    g[f"real_width_{c}"] = ok(f"{G}/CMP_real_{c}.json")
g["resume_A_standin"] = ok(f"{G}/resume_combo_A_g00_s2pnone.json")
for t in ("g01_s2pnone", "g00_s2pcb0", "g00_s2pfull", "g01_s2pcb0", "g01_s2pfull"):
    for p in ("C", "D"):
        g[f"resume_{p}_{t}"] = ok(f"{G}/resume_combo_{p}_{t}.json")
xs = sorted(glob.glob(f"{G}/resume_combo_X_*.json"))
g["resume_refusals_5"] = len(xs) >= 5 and all(ok(x) for x in xs)
for c in ("off", "g0cb0", "g0full"):
    g[f"start_eval_S9_{c}"] = ok(f"{G}/eval_{c}.json")
try:
    fs = json.load(open(f"{G}/f_stats.json"))
    g["f_sane_plan1_npl3_npl5"] = all(fs[s]["frac_invalid"] == 0.0 and fs[s]["has_orig_hw_all"] and fs[s]["clips"] > 5000 for s in ("plan1", "npl3", "npl5"))
except (OSError, ValueError, KeyError):
    g["f_sane_plan1_npl3_npl5"] = None
# A's suite re-run on the merged code (postmerge copy)
cfg = json.load(open(f"{P}/CMP_cfg.json")) if os.path.isfile(f"{P}/CMP_cfg.json") else {}
a_common = {"a1_cfg_flag_off_identical": bool(cfg.get("flag_off_cfg_identical") and cfg.get("flag_off_coef_resumekeys_nv_groups_identical") and cfg.get("flag_off_params_identical")),
            "a2_tiny_c128_orig_vs_copyoff": ok(f"{P}/CMP_tiny_c128_orig_vs_copyoff.json"), "a2_tiny_c1024_orig_vs_copyoff": ok(f"{P}/CMP_tiny_c1024_orig_vs_copyoff.json"),
            "noise_floor_tiny_c128_orig_vs_orig": ok(f"{P}/CMP_tiny_c128_orig_vs_orig.json"), "noise_floor_tiny_c1024_orig_vs_orig": ok(f"{P}/CMP_tiny_c1024_orig_vs_orig.json")}
a_per = {}
for v, cmpk, tz, tnz, rtag, mine in (("full", "flag_full", "full0", "fullnz", "", "full"), ("cb0", "flag_cb0", "cb00", "cb0nz", "_cb0", "cb0")):
    a_per[v] = {"a1_cfg_only_s2p_keys": bool(cfg.get(f"{cmpk}_cfg_only_s2p_keys_added")), "a1_params_only_s_emb": bool(cfg.get(f"{cmpk}_params_only_s_emb_added")),
                "b1_tiny_c128_zero_init": ok(f"{P}/CMP_tiny_c128_orig_vs_{tz}.json"), "b1_tiny_c1024_zero_init": ok(f"{P}/CMP_tiny_c1024_orig_vs_{tz}.json"),
                "opt_tiny_c128_nonzero_pose_only": ok(f"{P}/CMP_tiny_c128_orig_vs_{tnz}.json"), "opt_tiny_c1024_nonzero_pose_only": ok(f"{P}/CMP_tiny_c1024_orig_vs_{tnz}.json"),
                "b2_real_width_real_weights_zero_init (D's combined run CMP_real_" + mine + ")": g[f"real_width_{mine}"],
                **{f"c_resume_{ph}": ok(f"{P}/resume{rtag}_{ph}.json") for ph in "ABCDEFGHIJKL"},
                "with_g0cond_tiny_zero": g[f"tiny_g0{mine}"], "with_g0cond_real_width_zero": g[f"real_width_g0{mine}"],
                "with_g0cond_resume_C_D": g[f"resume_C_g01_s2p{mine}"] and g[f"resume_D_g01_s2p{mine}"],
                "with_g0cond_start_eval_S9": g[f"start_eval_S9_g0{mine}"]}
bad_g = {k: v for k, v in g.items() if v is not True}
s_pass = {v: all(x is True for x in a_common.values()) and all(x is True for x in d.values()) for v, d in a_per.items()}
print(json.dumps({"g0": g, "s2p_common": a_common, "s2p": a_per}, indent=1))
if late:
    print("files changed after the final wave started:", late)
if bad_g or not all(s_pass.values()):
    print("NOT WRITTEN. failing / missing:", bad_g, {v: {k: x for k, x in d.items() if x is not True} for v, d in a_per.items()},
          {k: x for k, x in a_common.items() if x is not True})
    sys.exit(1)
now = datetime.datetime.now(zoneinfo.ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S PT")
combos = ["g0cond", "s2p=cb0", "s2p=full", "g0cond+s2p=cb0", "g0cond+s2p=full"]
json.dump({"PASS": True, "groups": ["g0cond"], "core_md5": md5, "combinations_passed": combos, "flag": "PPM_G0COND=1",
           "written": now, "checks": g, "out": G, "patch_g0_vs_code811": "pose5/g0cond.patch (single-flag, pre-own-groups)",
           "merge": f"{S}/tests/g0/MERGE.json"}, open(f"{S}/tests/g0/G0_PASS", "w"), indent=1)
if os.path.exists(f"{S}/tests/s2p/S2P_PASS") and not os.path.exists(f"{S}/tests/s2p/S2P_PASS.premerge-A"):
    shutil.copy2(f"{S}/tests/s2p/S2P_PASS", f"{S}/tests/s2p/S2P_PASS.premerge-A")
s2p_marker = {**s_pass, "flag_off_byte_identical": all(x is True for x in a_common.values()) and g["tiny_off"] is True and g["start_eval_S9_off"] is True,
              "groups": ["s2p"], "core_md5": md5, "written": now, "reissued_by": "pose5 workstream D after the G0COND merge + own groups (audit S7)",
              "details": {"common": a_common, **a_per}, "premerge_marker_of_A": f"{S}/tests/s2p/S2P_PASS.premerge-A", "postmerge_suite": P}
for dst in (f"{S}/tests/s2p/S2P_PASS", f"{S}/tests/s2p_postmerge/S2P_PASS"):      # (same content at both paths ARMED may read)
    json.dump(s2p_marker, open(dst, "w"), indent=1)
print("wrote", f"{S}/tests/g0/G0_PASS", "and", f"{S}/tests/s2p/S2P_PASS", "+", f"{S}/tests/s2p_postmerge/S2P_PASS")
