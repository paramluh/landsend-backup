"""APPEND ingest step 2: place the listed files on the 16 training hosts' /scratch (= data/new-pose-llm-data-dl/place.py's rules: whole jobs on
one host, every release of a job together; README / manifests not needed). Balanced so the COMBINED fresh pose stream ends as late as
possible: BALANCE=tail (the trainer's PPM_FRESH_POSE_TAIL=1): each host's plan-1 rows + its new rows equal (the new rows estimated from bytes
with plan 1's rows per parquet byte); BALANCE=equal (TAIL=0): equal new bytes. Never fills a host below FLOOR_GB free (default 60: a
training save needs ~53 GB); warns below HOLD_GB (default 126: the HF watcher holds a save only with ~126 GB free).
usage (compute node, python -I):  src_place.py <W> host=free_GB [host=free_GB ..]   -> <W>/placement/<host>.tsv, <W>/PLACEMENT.json"""
import collections
import glob
import json
import os
import sys

W = sys.argv[1]
free = {h: float(g) for h, g in (x.split("=") for x in sys.argv[2:])}
HOSTS = list(free)
FLOOR, HOLD = float(os.environ.get("FLOOR_GB", "60")), float(os.environ.get("HOLD_GB", "126"))
BALANCE = os.environ.get("BALANCE", "tail")
assert BALANCE in ("tail", "equal"), BALANCE
P1 = "/data/home/amartyavarma/ppm/code-811/fresh/scan/plans"
PL1 = json.load(open("/data/home/amartyavarma/data/new-pose-llm-data-dl/PLACEMENT.json"))["hosts"]
n1 = {json.load(open(f))["host"]: int(json.load(open(f))["n_rows"]) for f in glob.glob(f"{P1}/PLAN-*.json")}
assert set(HOSTS) <= set(n1), ("hosts without a plan-1 plan", set(HOSTS) - set(n1))
rho = sum(n1[h] for h in HOSTS) / sum(PL1[h]["parquet_bytes"] * 1e9 for h in HOSTS)       # plan-1 packed rows per parquet byte
# 10-23 (delivery 3): PRIOR_TAGS=npl2[,..] = the earlier APPENDED plans (their Weka audit copies <A>/<tag>/scan/plans/PLAN-<host>.json): with
# BALANCE=tail their host row counts are added to plan 1's, so each host's rows over ALL plans come out equal (the TAIL=1 combined-stream end)
A_ = os.path.dirname(os.path.abspath(__file__))
for t_ in [x for x in os.environ.get("PRIOR_TAGS", "").split(",") if x]:
    nt_ = {json.load(open(f))["host"]: int(json.load(open(f))["n_rows"]) for f in glob.glob(f"{A_}/{t_}/scan/plans/PLAN-*.json")}
    assert set(HOSTS) <= set(nt_), ("hosts without a plan of prior tag", t_, set(HOSTS) - set(nt_))
    n1 = {h: n1[h] + nt_.get(h, 0) for h in n1}
jobs = collections.defaultdict(list)
for ln in open(f"{W}/listing.tsv"):
    rel, size, xh, job = ln.rstrip("\n").split("\t")
    jobs[job].append((rel, int(size), xh))
jb = {j: sum(s for _, s, _ in v) for j, v in jobs.items()}
jp = {j: sum(s for r, s, _ in v if r.endswith(".parquet")) for j, v in jobs.items()}
total, total_pq = sum(jb.values()), sum(jp.values())
cap = {h: max(0.0, (free[h] - FLOOR) * 1e9) for h in HOSTS}
assert sum(cap.values()) >= total * 1.02, (f"NOT ENOUGH /scratch: {total / 1e9:.1f} GB to place, room above the {FLOOR:g} GB floor "
                                           f"{sum(cap.values()) / 1e9:.1f} GB", {h: round(free[h]) for h in HOSTS})
# target new parquet bytes per host: tail -> n1_h + rho x B_h = L (water-filled, B_h >= 0, capped by disk); equal -> B_h = total / 16
ovh = total / max(total_pq, 1)                                    # all bytes per parquet byte (json pins)
if BALANCE == "tail":
    lo, hi = 0.0, max(n1[h] for h in HOSTS) + rho * total_pq
    for _ in range(200):
        L = (lo + hi) / 2
        s = sum(min(max(0.0, (L - n1[h]) / rho), cap[h] / ovh) for h in HOSTS)
        lo, hi = (L, hi) if s < total_pq else (lo, L)
    target = {h: min(max(0.0, (hi - n1[h]) / rho), cap[h] / ovh) for h in HOSTS}
else:
    lo, hi = 0.0, total_pq
    for _ in range(200):
        L = (lo + hi) / 2
        s = sum(min(L, cap[h] / ovh) for h in HOSTS)
        lo, hi = (L, hi) if s < total_pq else (lo, L)
    target = {h: min(hi, cap[h] / ovh) for h in HOSTS}
asg, abytes, job_host = {h: 0 for h in HOSTS}, {h: 0 for h in HOSTS}, {}
for j in sorted(jobs, key=lambda j: (-jp[j], j)):                  # largest jobs first, to the host furthest below its target
    ok = [h for h in HOSTS if abytes[h] + jb[j] <= cap[h]]
    assert ok, ("job does not fit anywhere", j, jb[j])
    h = min(ok, key=lambda h: ((asg[h] + jp[j]) / target[h]) if target[h] > 0 else 9e9)
    job_host[j] = h; asg[h] += jp[j]; abytes[h] += jb[j]
os.makedirs(f"{W}/placement", exist_ok=True)
for h in HOSTS:
    with open(f"{W}/placement/{h}.tsv.tmp", "w") as f:
        for j in sorted(j for j in job_host if job_host[j] == h):
            for rel, s, xh in sorted(jobs[j]):
                f.write(f"{rel}\t{s}\t{xh}\n")
    os.replace(f"{W}/placement/{h}.tsv.tmp", f"{W}/placement/{h}.tsv")
est = {h: int(n1[h] + rho * asg[h]) if BALANCE == "tail" else int(rho * asg[h]) for h in HOSTS}
warn = sorted(h for h in HOSTS if free[h] - abytes[h] / 1e9 < HOLD)
out = {"balance": BALANCE, "prior_tags": os.environ.get("PRIOR_TAGS", ""), "floor_gb": FLOOR, "hold_gb": HOLD, "rho_rows_per_GB": round(rho * 1e9, 1), "total_GB": round(total / 1e9, 2),
       "total_parquet_GB": round(total_pq / 1e9, 2), "jobs": len(jobs), "est_new_rows": int(rho * total_pq),
       "est_new_steps_at_64_rows_per_host_step": int(rho * total_pq / 1024),
       "hosts": {h: {"jobs": sum(1 for j in job_host if job_host[j] == h), "GB": round(abytes[h] / 1e9, 2), "parquet_GB": round(asg[h] / 1e9, 2),
                     "free_GB_before": round(free[h], 1), "free_GB_after": round(free[h] - abytes[h] / 1e9, 1), "plan1_rows": n1[h],  # (+ PRIOR_TAGS rows)
                     ("est_rows_plan1_plus_new" if BALANCE == "tail" else "est_new_rows"): est[h]} for h in HOSTS},
       "below_hold_gb_after": warn, "job_to_host": job_host}
json.dump(out, open(f"{W}/PLACEMENT.json", "w"))
print(json.dumps({k: v for k, v in out.items() if k != "job_to_host"}, indent=1), flush=True)
if warn:
    print(f"WARNING: after placement {warn} keep < {HOLD:g} GB free: the HF watcher will not hold (back up) new saves there until space is freed",
          flush=True)
