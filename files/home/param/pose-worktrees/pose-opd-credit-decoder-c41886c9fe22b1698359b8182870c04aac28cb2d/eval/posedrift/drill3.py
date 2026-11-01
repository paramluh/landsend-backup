import sys, numpy as np
sys.path.insert(0, "/data/home/amartyavarma/ppm/aloe-posedup-live/posedrift")
from analyze import load, diff, ci
rs, real = load()
dup = [r for r in rs if r["model"] == "duplex-498" and r["gen_s"] >= 20]
live = lambda r: r["client"] in ("browser", "playground")
L = [r for r in dup if live(r) and r["end_s"] >= 60]
print("live long calls n cfg1/cfg0:", sum(r["cfg"] for r in L), sum(1 - r["cfg"] for r in L))
for k in ("body_conf", "head_roll", "shear", "lean", "pitch", "below_floor", "conf"):
    a = [(r.get("late_minus_early") or {}).get(k) for r in L if r["cfg"]]; b = [(r.get("late_minus_early") or {}).get(k) for r in L if not r["cfg"]]
    sa = [r["slope_per_min"].get(k) for r in L if r["cfg"]]; sb = [r["slope_per_min"].get(k) for r in L if not r["cfg"]]
    print(f"{k:12s} late-early cfg1 {ci(a)} cfg0 {ci(b)} diff {diff(a, b)} | slope/min cfg1 {ci(sa)} cfg0 {ci(sb)}")
aloe = [r for r in real if r["kind"] == "aloe"]
for k in ("body_conf", "head_roll", "shear", "lean"):
    print("real late-early", k, ci([(r.get("late_minus_early") or {}).get(k) for r in aloe]), "slope", ci([r["slope_per_min"].get(k) for r in aloe]))
# live calls first 20 s only: cfg on vs off (all live calls >= 20 s)
from analyze import window
Lv = [r for r in dup if live(r)]
for k in ("body_conf", "head_roll", "shear", "lean", "red", "below_floor", "conf"):
    print("live 0-20s", k, diff([window(r, k, 0, 20) for r in Lv if r["cfg"]], [window(r, k, 0, 20) for r in Lv if not r["cfg"]]))
