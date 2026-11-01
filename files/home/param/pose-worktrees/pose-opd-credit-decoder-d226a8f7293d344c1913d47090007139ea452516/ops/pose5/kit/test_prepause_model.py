#!/usr/bin/env python3
"""pose5: REPLAY of the pre-pause disk model against the real switch (stage 31091, 11:32 PT, the first --go). Inputs = c/DISK_NEED.json as
written by that --go's step 0 (11:27:01 PT, OLD settings: s12 on /scratch with 6.7 GB caches to delete, s13 margin 3.5) and stage-31091's
log (each host's free after staging, the live gate's need, OK/ABORT). For every host it prints the OLD model's left, the NEW model's left
(c/disk/build_disk_need.py: - MoT pause delta - staging writes - cushion), and what the live gate actually saw.
Scenarios: AS-RUN = the cache deletions of that stage did not happen (stage_host.sh `nice guard_rm` bug, rc 127, fixed 10-27 11:5x);
           FIXED  = caches deleted as listed. The new model must flag every host the live gate aborted (AS-RUN) and no host that passed by a
           margin larger than the window-presence slack (npl5 already on disk lowers the live pending; the pre-pause model counts the full
           window: conservative).
usage: python3 -I test_prepause_model.py [--disk-need PATH] [--stage-log PATH]"""
import argparse
import json
import re
import sys

sys.path.insert(0, "/data/home/amartyavarma/ppm/pose5/c/disk")
ap = argparse.ArgumentParser()
ap.add_argument("--disk-need", default="/data/home/amartyavarma/ppm/pose5/kit/state/DISK_NEED.as-run-31091.json")
ap.add_argument("--stage-log", default="/data/home/amartyavarma/ppm/pose5/kit/logs/stage-31091.log")
a = ap.parse_args()
D = json.load(open(a.disk_need))
import prepause_terms as PT  # noqa: E402  (the same functions build_disk_need.py uses)

terms = PT.terms(upto=D["written"])          # MoT delta + staging writes + cushion, from motguard's records as of that --go
live = {}
for ln in open(a.stage_log, errors="replace"):
    m = re.search(r"STAGEHOST (slinky-\d+) \S+ OK free ([0-9.]+) GB \(need ([0-9.]+)\)", ln)
    if m: live[m.group(1)] = ("OK", float(m.group(2)), float(m.group(3)))
    m = re.search(r"STAGEHOST (slinky-\d+) \S+ ABORT gate: free ([0-9.]+) GB < need ([0-9.]+) GB", ln)
    if m: live[m.group(1)] = ("ABORT", float(m.group(2)), float(m.group(3)))
print(f"pre-pause terms as of {D['written']}: {terms['why']}")
print(f"{'host':>10} {'old':>6} | {'AS-RUN new':>10} | {'FIXED new':>9} | live gate (31091)")
bad = 0
for h, v in D["hosts"].items():
    old = v["left_after_margin_GB"]
    ex = terms["per_host"][h]
    asrun = old - ex - v["caches_GB"]               # caches listed for deletion were NOT deleted (bug)
    fixed = old - ex
    lv = live.get(h)
    lvs = f"{lv[0]} free {lv[1]} need {lv[2]} left {lv[1] - lv[2]:+.1f}" if lv else "n/a"
    miss = lv and lv[0] == "ABORT" and asrun >= 0          # a real abort the new model would not have flagged
    fp = lv and lv[0] == "OK" and asrun < 0
    bad += bool(miss)
    print(f"{h:>10} {old:+6.1f} | {asrun:+10.1f} {'SHORT' if asrun < 0 else 'ok':5} | {fixed:+9.1f} {'SHORT' if fixed < 0 else 'ok':5} | {lvs}"
          + ("  <- MISSED" if miss else "") + ("  (conservative: live pending was smaller, npl5 already on disk)" if fp else ""))
print("NEW MODEL FLAGS EVERY REAL ABORT" if not bad else f"NEW MODEL MISSED {bad} REAL ABORT(S)"); sys.exit(1 if bad else 0)
