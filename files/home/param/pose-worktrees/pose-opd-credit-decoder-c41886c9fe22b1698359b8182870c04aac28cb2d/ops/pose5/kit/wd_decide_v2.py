#!/usr/bin/env python3
"""pose5 watchdog v2 resume decision (main 10-27). stdin = one scan of every host: "<host> L <step>" (CKPT/dcp-N COMPLETE), "<host> H <step>"
(chained-hold dir hold/dcp-N COMPLETE), "<host> END". Prints exactly one line:
  FROM <step> <dir>      resume from a save COMPLETE on ALL hosts in one place (CKPT, or the hold) -- the newest such save C, when no HF-verified
                         save of this run is newer than C
  RESTAGE <step>         the newest save on the hosts is incomplete / marked bad and the newest HF-VERIFIED save V of this run is newer than
                         every complete local save: restage V from HF on every host (kit/restage_dcp_hf.py), resume from the per-host link
  FROM <pin> <pin dir>   only if NO save > pin exists anywhere (never back to 17850 once a newer save exists)
  ALERT <why>            the scan did not answer on every host; or saves > pin exist but none is complete locally and none is HF-verified
"HF-verified" = setup-logs/hfb done/dcp-N-final (or dcp-N-j*-final) == ok AND a PASS verify record per scanned host AND hfb's dcp record for
N names one of this run's jobs (--jobs). Bad saves (--bad: a relaunch from them died before any new step) never count as complete.
v2 (audit 4 R4-1): FROM / RESTAGE of a step S becomes ALERT when a COMPLETE save > S stays in CKPT on any host (keep-1 prune hazard).
usage: <scan> | wd_decide_v2.py --nh 16 --pin 17850 [--resume-pin DIR] --ckpt DIR --holdd DIR [--bad 1,2] [--jobs 31101,..] [--hfb DIR]"""
import argparse
import glob
import json
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--nh", type=int, required=True); ap.add_argument("--pin", type=int, required=True); ap.add_argument("--resume-pin", default="")
ap.add_argument("--ckpt", required=True); ap.add_argument("--holdd", required=True); ap.add_argument("--bad", default="")
ap.add_argument("--jobs", default=""); ap.add_argument("--hfb", default="/data/home/amartyavarma/setup-logs/hfb")
a = ap.parse_args()
bad = {int(x) for x in a.bad.split(",") if x.strip().isdigit()}
jobs = {x for x in a.jobs.split(",") if x.strip()}
L, H, E = {}, {}, set()
for ln in sys.stdin:
    p = ln.split()
    if len(p) == 2 and p[1] == "END":
        E.add(p[0].split(".")[0])
    elif len(p) == 3 and p[1] in ("L", "H") and p[2].isdigit():
        (L if p[1] == "L" else H).setdefault(p[0].split(".")[0], set()).add(int(p[2]))
if len(E) != a.nh:
    print(f"ALERT scan answered on {len(E)}/{a.nh} hosts (failed / partial scan)"); sys.exit()
hosts = sorted(E)
anyw = sorted({s for d in (L, H) for v in d.values() for s in v if s > a.pin})


def where(s):
    if all(s in L.get(h, set()) for h in hosts): return f"{a.ckpt}/dcp-{s}"
    if all(s in H.get(h, set()) for h in hosts): return f"{a.holdd}/dcp-{s}"
    return None


complete = [s for s in anyw if s not in bad and where(s)]
C = complete[-1] if complete else None


def verified(s):
    ok = False
    for f in glob.glob(f"{a.hfb}/done/dcp-{s}-final") + glob.glob(f"{a.hfb}/done/dcp-{s}-j*-final"):
        ok = ok or open(f).read().strip() == "ok"
    if not ok:
        return False
    for h in hosts:
        good = False
        for f in glob.glob(f"{a.hfb}/verify/InternalCan__ppm-v1*__dcp-{s}__{h}.json") + glob.glob(f"{a.hfb}/verify/InternalCan__ppm-v1*__dcp-{s}-j*__{h}.json"):
            try:
                good = good or json.load(open(f)).get("PASS") is True
            except Exception:
                pass
        if not good:
            return False
    if jobs:
        try:
            rs = [json.loads(l) for l in open(f"{a.hfb}/records.jsonl") if f'"step": {s},' in l]
        except OSError:
            rs = []
        if not any(r.get("kind") in ("dcp", "dcp-pinned") and str(r.get("job")) in jobs and r.get("PASS") is True for r in rs):
            return False
    return True


cand = sorted({int(f.split("dcp-")[1].split("-")[0]) for f in glob.glob(f"{a.hfb}/done/dcp-*-final") if f.split("dcp-")[1].split("-")[0].isdigit()})
V = next((s for s in sorted((s for s in cand if s > a.pin), reverse=True) if verified(s)), None)
newest = max(anyw + sorted(bad)) if (anyw or bad) else None
def newer_ckpt(s):   # v2 (audit 4 R4-1): COMPLETE saves > s left in CKPT on any host -> keep-1 prune would delete the relaunched run's saves
    return {k: sorted(h for h in hosts if k in L.get(h, set())) for k in sorted({x for v in L.values() for x in v if x > s})}


def guarded(s, line):
    nk = newer_ckpt(s)
    if nk:
        print(f"ALERT resuming from dcp-{s} while a NEWER COMPLETE save stays in CKPT ("
              + "; ".join(f"dcp-{k} on {len(v)} host(s) {v[:4]}" for k, v in nk.items())
              + "): the trainer's keep-1 prune would delete the relaunched run's own saves -> NOT relaunched; main moves it out of ckpt/ppm-v1")
    else:
        print(line)


if C is not None and (V is None or C >= V):
    guarded(C, f"FROM {C} {where(C)}")
    if C != newest:
        print(f"NOTE newest save {newest} is incomplete or bad: complete local {C} >= newest HF-verified {V}", file=sys.stderr)
elif V is not None:
    guarded(V, f"RESTAGE {V}")
    print(f"NOTE newest save {newest} incomplete/bad; newest complete local {C}; restaging HF-verified {V}", file=sys.stderr)
elif anyw or bad:
    print(f"ALERT save(s) > {a.pin} exist ({(anyw or sorted(bad))[-3:]}) but none is complete on all hosts and none is HF-verified: NOT rewinding to {a.pin}")
elif a.resume_pin:
    print(f"FROM {a.pin} {a.resume_pin}")
else:
    print("ALERT no save > pin and no resume pin")
