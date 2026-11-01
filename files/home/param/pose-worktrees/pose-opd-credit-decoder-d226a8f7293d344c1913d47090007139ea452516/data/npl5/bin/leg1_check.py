"""LEG 1 (dcp-17850 -> 18022 on plans [plan 1, npl2, npl3], no npl5) node-local check for ONE 811 slot, on the host that plays it.
usage (qwen-s2 venv, python -I):  leg1_check.py <root> <plan key> <out.json> [--dcp DCP17850_FRESH.json] [--micros] [--need-tsv out.tsv]
  root = /scratch/amartyavarma (plan dirs ppm-fresh-pose, -npl2, -npl3 under it); plan key = the plans' host (slinky-9 on slinky-15, slinky-7
  for slot slinky-10, else the host itself).
Checks: the 3 plans load (fresh_pose.load_plan: digest of files + index arrays), host == key, digests == dcp-17850's cfg.fresh.pose_digests[key]
(and == the node-local dcp-17850 train_state.json when present); every parquet file of every row each of the slot's 8 ranks reads from the
resume (stream position (17850 - 16861) x 8 = 7912) to the end of its plan 1-3 rows (leg 1 = positions < (18022 - 16861) x 8 = 9288, the rest
= the npl3 tail rows read at the start of leg 2) exists at src_root with its listed size; --micros: the trainer's micro (FreshPoseMultiDS ->
data_ppm.pose_micro_ppm -> packed) of every rank at the resume step and at leg 1's last step. Missing files -> --need-tsv (rel, size, xet of
the delivery-3 listing) for src_fetch. Exit 0 ok, 2 files missing, 1 mismatch / error."""
import argparse
import json
import os
import sys
import time

import numpy as np

C = "/data/home/amartyavarma/ppm/code-811"
A = f"{C}/fresh/append"
SF, NPM, LW, RESUME, END3 = 16861, 8, 8, 17850, 18022
DIRS = ("ppm-fresh-pose", "ppm-fresh-pose-npl2", "ppm-fresh-pose-npl3")
LISTINGS = {1: f"{A}/npl2/listing.tsv", 2: f"{A}/npl3/listing.tsv"}
LOGICAL = {"slinky-7": "slinky-10"}
SLOTS = ["slinky-0", "slinky-1", "slinky-2", "slinky-3", "slinky-5", "slinky-6", "slinky-8", "slinky-9", "slinky-10", "slinky-11", "slinky-12",
         "slinky-13", "slinky-14", "slinky-16", "slinky-17", "slinky-19"]
ap = argparse.ArgumentParser()
ap.add_argument("root"); ap.add_argument("key"); ap.add_argument("out")
ap.add_argument("--dcp", default="/data/home/amartyavarma/ppm/pose5/npl5/leg1/slot9/DCP17850_FRESH.json")
ap.add_argument("--micros", action="store_true")
ap.add_argument("--need-tsv", default="")
a = ap.parse_args()
sys.path.insert(0, C)
import data_ppm as D  # noqa: E402,F401
sys.path.insert(0, f"{C}/fresh")
import fresh_pose as FP  # noqa: E402

t0 = time.time()
res = {"key": a.key, "phys": os.uname().nodename.split(".")[0], "root": a.root, "ok": False}
pl = [FP.load_plan(f"{a.root}/{d}") for d in DIRS]
dg = [p["digest"] for p in pl]
want = json.load(open(a.dcp))["fresh"]["pose_digests"][a.key]
res.update(digests=dg, digests_dcp17850=want, hosts=[p["host"] for p in pl])
assert all(p["host"] == a.key for p in pl), ("plan host != key", res["hosts"], a.key)
assert dg == want, ("plan digests != dcp-17850 pose_digests", dg, want)
local_ts = "/scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-17850/train_state.json"
if os.path.isfile(local_ts):
    lt = json.load(open(local_ts))["cfg"]["fresh"]["pose_digests"][a.key]
    assert lt == dg, ("node-local dcp-17850 train_state digests differ", lt, dg)
    res["node_local_dcp17850_digests"] = "equal"
p0, e1 = (RESUME - SF) * NPM, (END3 - SF) * NPM
need, ranks = [set() for _ in pl], {}
for l in range(LW):
    s = FP.append_stream([int(p["n_rows"]) for p in pl], l, LW, NPM, SF, 1, ())
    w = s[p0:]
    ks = (w >> FP.PLAN_SHIFT)
    ranks[l] = {"stream_len": int(len(s)), "plans_from_resume": sorted({int(k) + 1 for k in ks}), "rows_from_resume": int(len(w)),
                "leg1_rows": int(min(len(s), e1) - p0), "tail_rows_after_18022": int(max(0, len(s) - e1))}
    for x in w:
        k, r = FP.split_row(int(x)); z = pl[k]["z"]
        for j in z["row_seg"][z["row_off"][r]:z["row_off"][r + 1]]:
            need[k].add(pl[k]["files"][int(z["seg_file"][j])])
res["ranks"] = ranks
assert not need[0], ("plan-1 rows after 17850: not expected", len(need[0]))
miss, nbytes = [], 0
for k in (1, 2):
    if not need[k]:
        continue
    lst = {}
    for ln in open(LISTINGS[k]):
        r = ln.rstrip("\n").split("\t"); lst[r[0]] = (int(r[1]), r[2])
    for f in sorted(need[k]):
        sz, xh = lst[f"jobs/{f}"]; nbytes += sz
        q = f"{pl[k]['src_root']}/{f}"
        if not (os.path.isfile(q) and os.path.getsize(q) == sz):
            miss.append((k, f"jobs/{f}", sz, xh))
res.update(files_needed={DIRS[k]: len(need[k]) for k in (1, 2)}, bytes_needed=nbytes, missing=len(miss), missing_bytes=sum(m[2] for m in miss),
           missing_examples=[m[1] for m in miss[:5]])
if a.need_tsv:
    with open(a.need_tsv, "w") as f_:
        for m in miss:
            assert m[0] == 2, "only npl3 files are fetched by the hook"
            f_.write(f"{m[1]}\t{m[2]}\t{m[3]}\n")
if not miss and a.micros:
    cfg = D.default_cfg(pose_delay=2)
    order = sorted(LOGICAL.get(h, h) for h in SLOTS)
    order = sorted(order, key=lambda h: int(h.split("-")[1]))
    mic = []
    for l in range(LW):
        s = FP.append_stream([int(p["n_rows"]) for p in pl], l, LW, NPM, SF, 1, ())
        picks = {"resume_17850": p0, "leg1_last_18021": e1 - 1}
        ds = FP.FreshPoseMultiDS(pl, [int(s[i]) for i in picks.values()], cfg, 8 * order.index(LOGICAL.get(a.key, a.key)) + l)
        for j, (nm, i) in enumerate(picks.items()):
            t1 = time.time(); m = ds[j]; N = len(m["pid"])
            assert m["ids"].shape == (2, N) and m["pdepth"].shape == (N, 32) and (m["present"][1] == 1).all()
            k_, r_ = FP.split_row(int(s[i]))
            mic.append({"local": l, "what": nm, "plan": k_ + 1, "row": r_, "N": int(N), "s": round(time.time() - t1, 2)})
    res["micros"] = mic
res["ok"] = not miss
res["seconds"] = round(time.time() - t0, 1)
json.dump(res, open(a.out, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k not in ("ranks", "micros")}), flush=True)
sys.exit(0 if not miss else 2)
