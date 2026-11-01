"""npl5 OFF-HOST PLAN BUILD (pose5 workstream B, 10-27): the per-slot npl5 plans exactly as `fresh_pose.py all` (plan step) would build them on
each of the 16 hosts, built on a spare from the staged scans (W/scan/<slot>.parquet). Same code, called in place (read-only):
code-811/fresh/fresh_pose.py global_keep() + cmd_plan(), with the env ingest_more.sh would give it for TAG npl5:
  FRESH_SRC  = /scratch/amartyavarma/npl-append/npl5/jobs     (the hosts' canonical data path; recorded as src_root, never read by the plan step)
  FRESH_SCAN = W/scan                                          (the staged scans = the on-host scans: they hold no paths)
  FRESH_TAG  = npl5, FRESH_PRIOR_SCANS = plan 1, npl2, npl3 scans (= ingest_more.sh's PRIOR for a tag outside code-811), FRESH_PRIOR_VIDEO=overlap
  FRESH_HOST = <slot> (one of fresh_pose.HOSTS = the plan keys of plans 1-3: slinky-7 = slot slinky-10, slinky-9 = slot 9), FRESH_OUT per slot.
global_keep() is deterministic and host-independent (every host computes the same), so it runs ONCE and the 16 cmd_plan calls (forked, POOL at a
time) reuse it (= ppmvox v5slot7/rebuild_plan.py, which rebuilt slinky-9's plans 1-3 byte-identically this way). Output:
  W/plans/<slot>/ppm-fresh-pose-npl5/{PLAN.json, PLAN.npz}  (copied on launch day to /scratch/amartyavarma/ppm-fresh-pose-npl5 on the slot's host)
  W/scan/plans/PLAN-<slot>.json                               (audit copies, as on-host)
  W/plans/PLANS.json                                           (digest, n_rows, kept h, sha256 of both files, load_plan OK per slot)
Only PLAN.json 'written' (a timestamp) and the npz zip member times differ from an on-host build; the digest (files + seg/row arrays) is equal.
usage (qwen-s2 venv, python -I, a spare, ~40 GB):  plan_offhost.py [--pool 6]"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time

W = os.environ.get("NPL5_W", "/data/home/amartyavarma/ppm/pose5/npl5")   # (NPL5_W: a test work dir)
F = "/data/home/amartyavarma/ppm/code-811/fresh"
PRIOR = f"{F}/scan,{F}/append/npl2/scan,{F}/append/npl3/scan"
ap = argparse.ArgumentParser()
ap.add_argument("--pool", type=int, default=6)
ap.add_argument("--slots", default="")
a = ap.parse_args()
for k in ("FRESH_HOSTS", "FRESH_SEG_PER_SHARD", "FRESH_LIMIT_JOBS"):
    os.environ.pop(k, None)
os.environ.update(FRESH_SRC="/scratch/amartyavarma/npl-append/npl5/jobs", FRESH_SCAN=f"{W}/scan", FRESH_TAG="npl5", FRESH_PRIOR_SCANS=PRIOR,
                  FRESH_PRIOR_VIDEO="overlap", FRESH_HOST="slinky-0", FRESH_OUT=f"{W}/plans/slinky-0/ppm-fresh-pose-npl5", OMP_NUM_THREADS="1")
sys.path.insert(0, F)
import fresh_pose as FP  # noqa: E402

assert os.path.abspath(FP.__file__) == f"{F}/fresh_pose.py", FP.__file__
SLOTS = [s for s in (a.slots.split(",") if a.slots else FP.HOSTS)]
assert set(SLOTS) <= set(FP.HOSTS) and len(FP.HOSTS) == 16, (SLOTS, FP.HOSTS)
missing = [h for h in FP.HOSTS if not os.path.exists(f"{W}/scan/{h}.json")]
assert not missing, ("scans missing", missing)
t0 = time.time()
keep, st = FP.global_keep()
FP.log(f"npl5 global_keep {time.time() - t0:.0f}s: kept {st['kept']:,} of {st['scanned']:,} scanned; "
       + json.dumps({k: v for k, v in st.items() if not k.startswith(("kept_slinky", "frames_kept_slinky"))}))
os.makedirs(f"{W}/plans", exist_ok=True)
json.dump(dict(st), open(f"{W}/plans/GLOBAL_KEEP.json", "w"), indent=1)
FP.global_keep = lambda: (keep, st)


def one(h):
    out = f"{W}/plans/{h}/ppm-fresh-pose-npl5"
    os.makedirs(out, exist_ok=True)
    if os.path.exists(f"{out}/PLAN.json"):
        FP.log(f"{h}: plan exists")
    else:
        FP.HOST, FP.OUT = h, out
        t1 = time.time(); FP.cmd_plan(None); FP.log(f"{h}: cmd_plan {time.time() - t1:.0f}s")
    m = FP.load_plan(out)                                       # digest-asserted
    assert m["host"] == h and m["tag"] == "npl5" and m["src_root"] == os.environ["FRESH_SRC"], (m["host"], m.get("tag"), m["src_root"])
    sha = {n: hashlib.sha256(open(f"{out}/{n}", "rb").read()).hexdigest() for n in ("PLAN.json", "PLAN.npz")}
    return h, {"dir": out, "digest": m["digest"], "n_rows": int(m["n_rows"]), "kept_segments": int(m["kept_segments"]),
               "kept_h": round(m["kept_frames"] / 12.5 / 3600, 1), "n_files": len(m["files"]), "n_shards": int(m["n_shards"]),
               "bytes": {n: os.path.getsize(f"{out}/{n}") for n in ("PLAN.json", "PLAN.npz")}, "sha256": sha, "written": m["written"]}


with mp.get_context("fork").Pool(min(a.pool, len(SLOTS))) as pool:
    res = dict(pool.map(one, SLOTS, chunksize=1))
old = json.load(open(f"{W}/plans/PLANS.json"))["slots"] if os.path.exists(f"{W}/plans/PLANS.json") else {}
old.update(res)
summ = {"slots": old, "total_rows": sum(v["n_rows"] for v in old.values()), "total_h": round(sum(v["kept_h"] for v in old.values())),
        "prior_scans": PRIOR.split(","), "src_root": os.environ["FRESH_SRC"], "builder": FP.__file__, "seconds": round(time.time() - t0, 1),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
json.dump(summ, open(f"{W}/plans/PLANS.json", "w"), indent=1)
FP.log(f"npl5 plans: {len(old)} slots, {summ['total_rows']:,} rows, {summ['total_h']:,} h; {summ['seconds']}s")
