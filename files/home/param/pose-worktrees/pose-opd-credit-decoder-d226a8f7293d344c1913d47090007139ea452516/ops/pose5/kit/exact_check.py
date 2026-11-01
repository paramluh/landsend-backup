#!/usr/bin/env python3
"""pose5 switch, step 6 (audit 2 R2-1): the start-eval exactness check of the new run vs a reference run at the same step.
Key rule = the trainer's own (code-811-s2p/train_ppm.py, START EVAL): keys starting eval_ / codependence/ / xdep/, minus keys ending time_s
(wall clock). Record = the LAST line at <step> carrying an eval_ key (as the trainer reads it).
Verdict (exit code):
  0  EXACT          every reference key present in the new run and equal (NaN == NaN)
  10 WARN           every reference key present, max relative diff <= --warn-rel (default 1e-6; slot 9 now runs on slinky-15): LIVE + WARN
                    line listing the keys that differ (and any keys only the new run has)
  1  NOT VERIFIED   a reference key missing in the new run, fewer than --min-keys common keys, a record missing, or a relative diff > warn-rel
relative diff = |a - b| / max(|a|, |b|, 1e-6)   (values below 1e-6 are compared at an absolute 1e-12)
usage: exact_check.py <new metrics.jsonl> <ref metrics.jsonl> <step> [--min-keys 184] [--warn-rel 1e-6] [--label NAME]"""
import argparse
import json
import math
import sys

ap = argparse.ArgumentParser()
ap.add_argument("new"); ap.add_argument("ref"); ap.add_argument("step", type=int)
ap.add_argument("--min-keys", type=int, default=184); ap.add_argument("--warn-rel", type=float, default=1e-6); ap.add_argument("--label", default="ref")
a = ap.parse_args()


def rec(path, step):
    r = None
    try:
        for ln in open(path, errors="replace"):
            if f'"step": {step},' in ln and '"eval_' in ln:
                try: r = json.loads(ln)
                except ValueError: pass
    except OSError:
        return None
    return r


def keys(r):
    return {k for k, v in r.items() if k.startswith(("eval_", "codependence/", "xdep/")) and not k.endswith("time_s")
            and isinstance(v, (int, float)) and not isinstance(v, bool)}


def rel(x, y):
    if x == y or (isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y)): return 0.0
    if any(isinstance(v, float) and not math.isfinite(v) for v in (x, y)): return math.inf
    return abs(x - y) / max(abs(x), abs(y), 1e-6)


n, r = rec(a.new, a.step), rec(a.ref, a.step)
if not n or not r:
    print(f"   NOT VERIFIED: no step-{a.step} eval record in {'the new run' if not n else a.label} ({a.new if not n else a.ref})"); sys.exit(1)
kn, kr = keys(n), keys(r)
miss, extra, com = sorted(kr - kn), sorted(kn - kr), sorted(kr & kn)
d = sorted(((rel(n[k], r[k]), abs(n[k] - r[k]) if math.isfinite(n[k]) and math.isfinite(r[k]) else math.inf, k) for k in com), reverse=True)
diff = [(x, ad, k) for x, ad, k in d if x > 0]
print(f"   start eval vs {a.label} @{a.step}: {len(com)} common metrics (trainer rule), exact {len(com) - len(diff)}, "
      + (f"max rel {d[0][0]:.3g} ({d[0][2]})" if d else "max rel n/a") +(f"; only in new run: {extra[:8]}" if extra else ""))
if miss or len(com) < a.min_keys:
    print(f"   NOT VERIFIED: key mismatch: {len(miss)} reference keys missing in the new run {miss[:8]}; common {len(com)} (min {a.min_keys})"); sys.exit(1)
if not diff and not extra:
    print("   EXACT"); sys.exit(0)
if diff and diff[0][0] > a.warn_rel:
    print(f"   NOT VERIFIED: {sum(1 for x, _, _ in diff if x > a.warn_rel)} keys differ by more than rel {a.warn_rel:g}: "
          + ", ".join(f"{k} rel {x:.3g} abs {ad:.3g}" for x, ad, k in diff[:8])); sys.exit(1)
print(f"   WARN: {len(diff)} keys differ within rel {a.warn_rel:g}: " + ", ".join(f"{k} rel {x:.3g}" for x, _, k in diff[:20])
      + (f"; keys only in the new run: {extra[:20]}" if extra else "")); sys.exit(10)
