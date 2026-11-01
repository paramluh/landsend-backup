"""PROVISIONAL npl5 plan of slot slinky-0 built from slinky-0's scan alone (FRESH_HOSTS=slinky-0), for testing the rolling fetcher before all
16 scans exist. slinky-0 is first in the global dedup order (hosts sorted), so its keep mask - and so its plan digest - does not depend on the
other hosts' scans: the final plan (plan_offhost.py) must give the SAME digest (checked later; a determinism proof). Output W/prov/slinky-0/."""
import json, os, sys, time
W = "/data/home/amartyavarma/ppm/pose5/npl5"; F = "/data/home/amartyavarma/ppm/code-811/fresh"
for k in ("FRESH_SEG_PER_SHARD", "FRESH_LIMIT_JOBS"):
    os.environ.pop(k, None)
os.makedirs(f"{W}/prov/scan", exist_ok=True)
for x in ("parquet", "json"):
    d = f"{W}/prov/scan/slinky-0.{x}"
    if not os.path.lexists(d):
        os.symlink(f"{W}/scan/slinky-0.{x}", d)
os.environ.update(FRESH_SRC="/scratch/amartyavarma/npl-append/npl5/jobs", FRESH_SCAN=f"{W}/prov/scan", FRESH_TAG="npl5",
                  FRESH_PRIOR_SCANS=f"{F}/scan,{F}/append/npl2/scan,{F}/append/npl3/scan", FRESH_PRIOR_VIDEO="overlap", FRESH_HOSTS="slinky-0",
                  FRESH_HOST="slinky-0", FRESH_OUT=f"{W}/prov/slinky-0/ppm-fresh-pose-npl5", OMP_NUM_THREADS="1")
sys.path.insert(0, F)
import fresh_pose as FP
t0 = time.time(); FP.cmd_plan(None)
m = FP.load_plan(os.environ["FRESH_OUT"])
print(json.dumps({"digest": m["digest"], "n_rows": m["n_rows"], "kept_segments": m["kept_segments"], "kept_h": round(m["kept_frames"] / 12.5 / 3600),
                  "files": len(m["files"]), "s": round(time.time() - t0)}), flush=True)
