"""aloe-v2vox-raw: channel-A loudness into Voxtral, raw (VOX_A_GAIN=raw) vs 22420 (recorded Mimi LUFS gain).
  (a) MEASURED on a seeded random sample of this run's train + eval items through the run's own loader (data_aloe_v2vox.Stores.conv, the
      24 kHz waveform handed to Voxtral): BS.1770-4 integrated LUFS (pyloudnorm, = the 10-17 loudness audit's 'w24'), sample peak, share of
      rows with any |x| >= 1; the same rows through the old path for comparison.
  (b) FULL v2 train population from the 10-17 audit (aloe-m1vox-ft/loudness_audit/results.jsonl, every v2vox train row measured with the
      recorded gain): raw LUFS = audit LUFS - applied gain (gcs) / - recorded gain (mimidec) (integrated loudness shifts 1:1 with a pure gain).
usage: VOX_A_GAIN=raw ALOE_FLAC_ROOT=<A tracks> python -I loudness_report.py --n 360 --workers 12 --out <json>"""
import argparse
import concurrent.futures as cf
import json
import math
import os
import sys

import numpy as np

R = "/data/home/amartyavarma/ppm/aloe-v2vox-raw"
sys.path.insert(0, f"{R}/code"); sys.path.insert(0, f"{R}/code/lib")
AUD = "/data/home/amartyavarma/ppm/aloe-m1vox-ft/loudness_audit"


def stats(x):
    x = np.asarray([v for v in x if v is not None and math.isfinite(v)], np.float64)
    return {"n": int(len(x)), "mean": round(float(x.mean()), 3), "std": round(float(x.std()), 3), "p5": round(float(np.percentile(x, 5)), 3),
            "p50": round(float(np.percentile(x, 50)), 3), "p95": round(float(np.percentile(x, 95)), 3), "min": round(float(x.min()), 3),
            "max": round(float(x.max()), 3)}


_ST = {}


def one(job):
    name, it = job
    import pyloudnorm as pyln
    import data_aloe_v2vox as A
    if "st" not in _ST:
        _ST["st"] = A.Stores()
    cfg = A.aloe_cfg() if name == "train" else A.eval_cfg()
    out = {"set": name, "item": list(it)}
    for mode in ("raw", "mimi"):
        A.GAIN_MODE = mode
        m = _ST["st"].conv(tuple(it), cfg, 0)
        w = m["wav"].astype(np.float64)
        lu = float(pyln.Meter(24000).integrated_loudness(w))
        out[mode] = {"lufs": lu if math.isfinite(lu) else None, "peak": float(np.abs(w).max()), "ge1": int((np.abs(w) >= 1.0).sum()),
                     "gain_db": m["aref"]["gain_db"], "kind": m["aref"]["kind"]}
    A.GAIN_MODE = "raw"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=360); ap.add_argument("--workers", type=int, default=12); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import data_aloe_v2vox as A
    PL = A.make_plan(64)
    rng = np.random.default_rng(20261017)
    tr = [PL["items"][i] for i in rng.permutation(len(PL["items"]))[:a.n]]
    ev = [PL["eval"][i] for i in rng.permutation(len(PL["eval"]))[:max(1, a.n // 6)]]
    jobs = [("train", tuple(x)) for x in tr] + [("eval", tuple(x)) for x in ev]
    import multiprocessing as mp
    with cf.ProcessPoolExecutor(a.workers, mp_context=mp.get_context("fork")) as ex:
        res = list(ex.map(one, jobs))
    rep = {"sample": {}}
    for grp, sel in (("all", res), ("train", [r for r in res if r["set"] == "train"]), ("eval", [r for r in res if r["set"] == "eval"]),
                     ("gcs", [r for r in res if r["raw"]["kind"] == "gcs"]), ("mimidec", [r for r in res if r["raw"]["kind"] == "mimidec"])):
        if not sel:
            continue
        rep["sample"][grp] = {"raw_lufs": stats([r["raw"]["lufs"] for r in sel]), "old_lufs": stats([r["mimi"]["lufs"] for r in sel]),
                              "raw_peak": stats([r["raw"]["peak"] for r in sel]),
                              "raw_rows_any_ge1": int(sum(r["raw"]["ge1"] > 0 for r in sel)), "old_rows_any_ge1": int(sum(r["mimi"]["ge1"] > 0 for r in sel))}
    # (b) full v2 train population from the audit
    items = {}
    for ln in open(f"{AUD}/items.jsonl"):
        d = json.loads(ln)
        if d["in_v2vox"]:
            items[d["i"]] = d
    rg = A.rawgain()
    raw_full, old_full = [], []
    for ln in open(f"{AUD}/results.jsonl"):
        d = json.loads(ln)
        it = items.get(d["i"])
        if it is None or d["w24"]["lufs"] is None:
            continue
        old_full.append(d["w24"]["lufs"])
        if it["kind"] == "gcs":
            raw_full.append(d["w24"]["lufs"] - it["gain_db"])
        else:
            raw_full.append(d["w24"]["lufs"] - rg[it["flac"]]["recorded_lufs_gain_db"])
    rep["full_train_population_from_audit"] = {"raw_lufs": stats(raw_full), "old_lufs": stats(old_full)}
    json.dump({"rows": res, "report": rep}, open(a.out, "w"), indent=1)
    for k, v in rep["sample"].items():
        print(f"[sample {k}] raw LUFS {v['raw_lufs']} | old LUFS {v['old_lufs']} | raw peak {v['raw_peak']} | rows any |x|>=1 raw {v['raw_rows_any_ge1']} old {v['old_rows_any_ge1']}")
    print(f"[full v2 train population, audit-derived] raw LUFS {rep['full_train_population_from_audit']['raw_lufs']} | old {rep['full_train_population_from_audit']['old_lufs']}")


if __name__ == "__main__":
    main()
