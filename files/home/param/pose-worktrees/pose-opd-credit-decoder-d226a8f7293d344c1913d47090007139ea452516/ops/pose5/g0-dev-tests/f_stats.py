#!/usr/bin/env python3
"""pose5 G0COND: sanity of the frame-0 features f (data_ppm.g0_feat) on REAL clips + aspect-ratio availability -> out/f_stats.json.

Sources (read-only):
  plan1   plan-1's Weka copy data/new-pose-llm-data/new-pose-llm-data/jobs: a seeded sample of jobs, every parquet present, every segment.
  npl5    delivery 5 parquet staged by workstream B under pose5/npl5/raw/<host>/jobs (whatever is there now; read-only).
  samdata PPM's samdata pose store (eval / codependence rows of the 8,1,1 run) words.arrow clip records, if present on this node's
          /scratch (pose_s3q.load_node / PoseRows: exactly the trainer's clip records).
Frame-0 decode = fresh_pose.decode_new's frame-0 block verbatim (the code that fills every fresh row's clip record): global_state /
frame0_pose / orig_h / orig_w from the parquet pose struct -> root, width, frame0_ok; then f = data_ppm.g0_feat(clip record)."""
import glob
import json
import os
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, "/data/home/amartyavarma/ppm/pose5/code-811-g0/tests/g0")
import g0_cpu as H  # noqa: E402

sys.path.insert(0, H.COPY)
sys.path.insert(0, "/data/home/amartyavarma/qwen-1ch-data/pose-s3-samdata/scripts")
import data_ppm as D  # noqa: E402
import build_pose_rows_sam as B  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

NAMES = ["r_y", "r_x", "log_w", "-r_y/w", "(1-r_y)/w", "-r_x/w", "(1-r_x)/w", "log(H/W)"]
PCT = [0, 0.1, 1, 5, 25, 50, 75, 95, 99, 99.9, 100]
MAXF = int(os.environ.get("G0_MAX_FILES", "300"))


def frame0_of(P):
    """= fresh_pose.decode_new's frame-0 block (verbatim logic) -> clip-record fields."""
    frame0_ok, Hh, Ww, root, width = False, 0.0, 0.0, np.zeros(2, np.float32), 0.0
    try:
        gs = np.asarray(P["global_state"], np.float32).reshape(-1, 3)
        f0 = np.asarray(P["frame0_pose"], np.float32).reshape(-1, 3)
        Hh, Ww = float(P["orig_h"]), float(P["orig_w"])
        if len(gs) and np.isfinite(gs).all() and (gs == gs[0]).all() and f0.shape == (133, 3) and np.isfinite(f0).all() and Hh > 0 and Ww > 0:
            j55 = f0[B.JOINTS] / np.array([Hh, Ww, 1.0], np.float32)
            root, width = (j55[5, :2] + j55[6, :2]) / 2, abs(j55[5, 1] - j55[6, 1])
            frame0_ok = bool(width > 0) and np.allclose(gs[0], [*root, width], atol=1e-4)
    except Exception:  # noqa: BLE001
        frame0_ok = False
    return {"frame0_root": [float(root[0]), float(root[1])], "frame0_width": float(width), "orig_h": Hh, "orig_w": Ww, "frame0_ok": frame0_ok,
            "has_orig": P.get("orig_h") is not None and P.get("orig_w") is not None}


def read_pose(f):
    """the parquet pose struct of every row (the whole 'pose' column, as fresh_pose.COLS reads it)."""
    return [r["pose"] for r in pq.read_table(f, columns=["pose"]).to_pylist()]


def summarize(name, recs, extra=None):
    fs, oks, asp, ws, why = [], [], Counter(), [], Counter()
    for c in recs:
        f, ok = D.g0_feat(c)
        fs.append(f); oks.append(ok)
        if c.get("orig_h") and c.get("orig_w"):
            asp[f"{int(c['orig_h'])}x{int(c['orig_w'])}"] += 1
        if not ok:
            why["frame0_ok_false" if not c.get("frame0_ok", True) else ("no_orig" if not (c.get("orig_h") and c.get("orig_w")) else "w<=0/nonfinite")] += 1
        ws.append(float(c.get("frame0_width") or 0.0))
    fs, oks = np.stack(fs) if fs else np.zeros((0, 8), np.float32), np.asarray(oks, bool)
    v = fs[oks]
    out = {"source": name, "clips": int(len(oks)), "valid": int(oks.sum()), "invalid": int((~oks).sum()),
           "frac_invalid": float((~oks).mean()) if len(oks) else None, "invalid_why": dict(why),
           "has_orig_hw_all": bool(all(c.get("orig_h", 0) > 0 and c.get("orig_w", 0) > 0 for c in recs)),
           "aspect_HxW_top": asp.most_common(12), "n_distinct_HxW": len(asp),
           "frac_portrait": float(np.mean(v[:, 7] > 0.01)) if len(v) else None, "frac_landscape": float(np.mean(v[:, 7] < -0.01)) if len(v) else None,
           "frac_square": float(np.mean(np.abs(v[:, 7]) <= 0.01)) if len(v) else None,
           "pct": {n: [round(float(x), 5) for x in np.percentile(v[:, i], PCT)] for i, n in enumerate(NAMES)} if len(v) else {},
           "pct_levels": PCT,
           "abs_f_max_over_any_dim_pct": [round(float(x), 4) for x in np.percentile(np.abs(v).max(1), [50, 99, 99.9, 100])] if len(v) else [],
           "w_lt_0.02": int((v[:, 2] < np.log(0.02)).sum()) if len(v) else 0, "w_lt_0.05": int((v[:, 2] < np.log(0.05)).sum()) if len(v) else 0,
           "root_outside_frame": int(((v[:, 0] < 0) | (v[:, 0] > 1) | (v[:, 1] < 0) | (v[:, 1] > 1)).sum()) if len(v) else 0,
           "l1_norm_pct": [round(float(x), 3) for x in np.percentile(np.abs(v).sum(1), [50, 99, 100])] if len(v) else []}
    if extra:
        out.update(extra)
    H.log(name, json.dumps({k: out[k] for k in ("clips", "valid", "frac_invalid", "has_orig_hw_all", "n_distinct_HxW", "frac_portrait", "frac_landscape")}))
    return out


R = {}
t0 = time.time()
# ---- plan 1 (Weka copy): seeded job sample, every parquet present
SRC1 = "/data/home/amartyavarma/data/new-pose-llm-data/new-pose-llm-data/jobs"
jobs = sorted(os.listdir(SRC1))
rng = np.random.default_rng(20261027)
recs, nf, scanned_jobs = [], 0, 0
for j in rng.permutation(len(jobs)):
    if nf >= MAXF:
        break
    scanned_jobs += 1
    for f in sorted(glob.glob(f"{SRC1}/{jobs[j]}/*/part-*.parquet")):
        recs += [frame0_of(P) for P in read_pose(f)]
        nf += 1
R["plan1"] = summarize("plan1", recs, {"files": nf, "jobs_sampled": scanned_jobs, "jobs_total": len(jobs), "src": SRC1, "s": round(time.time() - t0)})
# ---- npl5 (B's staged raw parquet)
t0 = time.time()
fs5 = sorted(glob.glob("/data/home/amartyavarma/ppm/pose5/npl5/raw/*/jobs/*/*/part-*.parquet"))
pick = [fs5[i] for i in sorted(np.random.default_rng(5).choice(len(fs5), min(MAXF, len(fs5)), replace=False))] if fs5 else []
recs = []
for f in pick:
    try:
        recs += [frame0_of(P) for P in read_pose(f)]
    except Exception as e:  # noqa: BLE001  (a file B is still writing / deleting)
        H.log("npl5 skip", f, repr(e)[:120])
R["npl5"] = summarize("npl5", recs, {"files": len(pick), "files_staged_now": len(fs5), "s": round(time.time() - t0)})
# ---- npl3 (delivery 3 parquet staged for slot 9 by workstream B, read-only) -- audit NIT coverage
t0 = time.time()
fs3 = sorted(glob.glob("/data/home/amartyavarma/ppm/pose5/npl5/leg1/slot9/npl3/jobs/*/*/part-*.parquet"))
pick = [fs3[i] for i in sorted(np.random.default_rng(3).choice(len(fs3), min(MAXF, len(fs3)), replace=False))] if fs3 else []
recs = []
for f in pick:
    recs += [frame0_of(P) for P in read_pose(f)]
R["npl3"] = summarize("npl3", recs, {"files": len(pick), "files_staged": len(fs3), "s": round(time.time() - t0)})
# ---- samdata store clip records (if this node has it)
t0 = time.time()
roots = sorted({os.path.dirname(x) for pat in ("/scratch/amartyavarma/pose-s3-samdata*/NODE.json", "/scratch/amartyavarma/pose-s3-samdata*/*/NODE.json",
                                                 "/scratch/amartyavarma/pose-s3-samdata*/*/*/NODE.json") for x in glob.glob(pat)})
H.log("samdata stores on this node:", roots, "| top-level:", sorted(glob.glob("/scratch/amartyavarma/pose-s3-samdata*/*"))[:12])
if roots:
    import pyarrow as pa
    import pyarrow.ipc as ipc
    recs, used = [], None
    for root in roots:
        try:
            node = D.PS.load_node(root)
            for sh in node["shards"][:8]:
                tb = ipc.open_file(pa.memory_map(f"{root}/{sh}.words.arrow", "r")).read_all()
                for cl in tb.column("clips").to_pylist():
                    recs += cl
            used = (root, node["node"], min(8, len(node["shards"])))
            break
        except Exception as e:  # noqa: BLE001
            H.log("samdata store unusable", root, repr(e)[:200])
    R["samdata"] = summarize("samdata", recs, {"root": used[0], "node": used[1], "shards_read": used[2], "s": round(time.time() - t0)}) if used else \
        {"source": "samdata", "note": f"no usable samdata store on {os.uname().nodename}: {roots}"}
else:
    R["samdata"] = {"source": "samdata", "note": f"no samdata pose store on {os.uname().nodename} /scratch"}
json.dump(R, open(f"{H.OUTD}/f_stats.json", "w"), indent=1, default=str)
H.log("-> f_stats.json")
