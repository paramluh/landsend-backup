"""E2E support (spare only): rebuild plans 1 / npl2 / npl3 of ALL 16 plan keys off-host from the Weka scans with fresh_pose's own cmd_plan
(= ppmvox v5slot7/rebuild_plan.py: global_keep once per tag, then cmd_plan per host), so the end-to-end dry run can load every slot's real 4
plans without touching the production hosts. Checked: digest == dcp-17850 cfg.fresh.pose_digests[host][k] (and n_rows == the audit copy).
'written' is restored from the audit copy; the npz zip timestamps are NOT searched (bytes may differ from the hosts' files; digests are equal,
which is what the trainer checks). -> W/rebuild13/<host>/<plan dir>/{PLAN.json, PLAN.npz}, W/rebuild13/REBUILD-<tag>.json
usage (qwen-s2 venv, python -I):  rebuild13.py <npl|npl2|npl3>"""
import json
import os
import sys
import time

tag = sys.argv[1]
W = "/data/home/amartyavarma/ppm/pose5/npl5"
F = "/data/home/amartyavarma/ppm/code-811/fresh"
C = {"npl": dict(src="/scratch/amartyavarma/new-pose-llm-data/new-pose-llm-data/jobs", scan=f"{F}/scan", tag="", prior="", d="ppm-fresh-pose", k=0),
     "npl2": dict(src="/scratch/amartyavarma/npl-append/npl2/jobs", scan=f"{F}/append/npl2/scan", tag="npl2", prior=f"{F}/scan", d="ppm-fresh-pose-npl2", k=1),
     "npl3": dict(src="/scratch/amartyavarma/npl-append/npl3/jobs", scan=f"{F}/append/npl3/scan", tag="npl3", prior=f"{F}/scan,{F}/append/npl2/scan",
                  d="ppm-fresh-pose-npl3", k=2)}[tag]
O = f"{W}/rebuild13"
link = f"{O}/scanlink-{tag}"
os.makedirs(f"{link}/plans", exist_ok=True)
for fn in os.listdir(C["scan"]):
    if fn.endswith((".json", ".parquet")) and not os.path.lexists(f"{link}/{fn}"):
        os.symlink(f"{C['scan']}/{fn}", f"{link}/{fn}")
for k in ("FRESH_HOSTS", "FRESH_SEG_PER_SHARD", "FRESH_LIMIT_JOBS", "FRESH_TAG", "FRESH_PRIOR_SCANS", "FRESH_PRIOR_VIDEO"):
    os.environ.pop(k, None)
os.environ.update(FRESH_SRC=C["src"], FRESH_SCAN=link, FRESH_HOST="slinky-0", FRESH_OUT=f"{O}/slinky-0/{C['d']}", OMP_NUM_THREADS="1")
if C["tag"]:
    os.environ.update(FRESH_TAG=C["tag"], FRESH_PRIOR_SCANS=C["prior"], FRESH_PRIOR_VIDEO="overlap")
sys.path.insert(0, F)
import fresh_pose as FP  # noqa: E402
dg = json.load(open(f"{W}/leg1/slot9/DCP17850_FRESH.json"))["fresh"]["pose_digests"]
t0 = time.time()
keep, st = FP.global_keep()
FP.log(f"{tag}: global_keep {time.time() - t0:.0f}s")
FP.global_keep = lambda: (keep, st)
res = {}
for h in FP.HOSTS:
    out = f"{O}/{h}/{C['d']}"
    os.makedirs(out, exist_ok=True)
    if not os.path.exists(f"{out}/PLAN.json"):
        FP.HOST, FP.OUT = h, out
        FP.cmd_plan(None)
        audit = json.load(open(f"{C['scan']}/plans/PLAN-{h}.json"))
        p = json.load(open(f"{out}/PLAN.json")); p["written"] = audit["written"]
        json.dump(p, open(f"{out}/PLAN.json", "w"))
    m = FP.load_plan(out)
    res[h] = {"digest": m["digest"], "want": dg[h][C["k"]], "ok": m["digest"] == dg[h][C["k"]], "n_rows": int(m["n_rows"])}
    assert res[h]["ok"], (h, res[h])
json.dump(res, open(f"{O}/REBUILD-{tag}.json", "w"), indent=1)
FP.log(f"{tag}: 16/16 digests == dcp-17850's; {time.time() - t0:.0f}s")
