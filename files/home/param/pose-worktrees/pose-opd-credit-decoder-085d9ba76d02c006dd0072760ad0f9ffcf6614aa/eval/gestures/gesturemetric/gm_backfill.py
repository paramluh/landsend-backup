#!/usr/bin/env python3
"""gesturemetric -> playground: write genweb/runs/<id>/gesture_stats.json (= gm_run.run_gesture) for every run whose status.json state is
'done' (hidden runs included: the page filters). Agreed with aloedemo 10-26: atomic tmp + rename in the run dir, no other file touched.
Recomputes when tokens.npz is newer than gesture_stats.json; a failing run is retried only after its tokens.npz changes.
usage: gm_backfill.py [--loop SECONDS]   (one pass over all runs, then optionally keep polling for new runs)"""
import json
import os
import sys
import time

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
import gm_run  # noqa: E402

RUNS = "/data/home/amartyavarma/ppm/aloe-posedup-live/genweb/runs"
OUT_NAME = "gesture_stats.json"


def _mt(p):
    try:
        return os.path.getmtime(p)
    except OSError:
        return None


def one_pass(failed):
    n_ok = n_fail = 0
    for r in sorted(os.listdir(RUNS)):
        d = os.path.join(RUNS, r)
        tok = _mt(os.path.join(d, "tokens.npz"))
        if tok is None:
            continue
        try:
            if json.load(open(os.path.join(d, "status.json"))).get("state") != "done":
                continue
        except (OSError, ValueError):
            continue
        out = _mt(os.path.join(d, OUT_NAME))
        if (out is not None and out >= tok) or failed.get(r) == tok:
            continue
        try:
            res = gm_run.run_gesture(d)
        except Exception as e:  # noqa: BLE001
            failed[r] = tok
            n_fail += 1
            print(f"FAIL {r}: {type(e).__name__} {e}", flush=True)
            continue
        tmp = os.path.join(d, "." + OUT_NAME + ".tmp")
        with open(tmp, "w") as f:
            json.dump(res, f)
        os.replace(tmp, os.path.join(d, OUT_NAME))
        n_ok += 1
        print(f"{r} kappa={res['gesture_timing_kappa']} loud={res['gesture_timing_loud']} spm={res['strokes_per_min']} "
              f"{res['secs']}s", flush=True)
    return n_ok, n_fail


def main():
    import torch
    torch.set_grad_enabled(False)
    loop = float(sys.argv[sys.argv.index("--loop") + 1]) if "--loop" in sys.argv else None
    failed = {}
    first = True
    while True:
        n_ok, n_fail = one_pass(failed)
        if first:
            print(f"BACKFILL DONE {time.strftime('%H:%M:%S')} written {n_ok} failed {n_fail}", flush=True)
            first = False
        if loop is None:
            break
        time.sleep(loop)


if __name__ == "__main__":
    main()
