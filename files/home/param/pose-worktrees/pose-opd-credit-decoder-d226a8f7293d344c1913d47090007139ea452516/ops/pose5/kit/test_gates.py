"""pose5: the pre-pause gate (c/disk/build_disk_need.py -> gate_rules imported) and the live gate (kit/stage_host.sh -> `gate_rules.py eval|ahead`
CLI) get the SAME input and must agree, for every host of c/DISK_NEED.json and for boundary cases on slinky-13 (left = +0.05 / -0.05 GB)."""
import json, subprocess, sys
sys.path.insert(0, "/data/home/amartyavarma/ppm/pose5/kit")
import gate_rules as GR
D = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "/data/home/amartyavarma/ppm/pose5/c/DISK_NEED.json")); HF = float(D["hfb_floor"]); bad = 0
# (have_GB_exact already has the pre-pause-only terms subtracted = the expected free at the live gate, so both gates see the same input)
cli = lambda *a: subprocess.run([sys.executable, "-I", "/data/home/amartyavarma/ppm/pose5/kit/gate_rules.py", *map(str, a)], capture_output=True, text=True).stdout.split()
for h, v in D["hosts"].items():
    have = v["have_GB_exact"]                                           # what the host has after the switch frees its items (unrounded)
    ext = GR.EXTRAS_S15 if h == "slinky-15" else 0.0
    a1, w1 = GR.choose_ahead(h, have, HF, autoshrink=True); a2, w2 = cli("ahead", h, have, HF, 1)
    pre = GR.evaluate(h, have, w1 + ext, HF)                            # pre-pause: nothing staged yet
    live = cli("eval", h, round(have - ext, 6), w1, HF)                  # live: extras staged (on disk), window still to fetch
    agree = int(a2) == a1 and float(w2) == w1 and float(live[1]) == pre[1] and (live[2] == "OK") == pre[2] and pre[1] == v["left_after_margin_GB"]
    bad += not agree
    print(f"{h:>10} ahead {a1}/{a2} window {w1}/{w2} | pre-pause left {pre[1]:6.1f} {'OK' if pre[2] else 'SHORT'} | live left {live[1]:>6} {live[2]} | DISK_NEED {v['left_after_margin_GB']:6.1f} {'agree' if agree else 'DISAGREE'}")
for d in (+0.05, -0.05):                                                 # boundary on slinky-13
    need = GR.evaluate("slinky-13", 0, 8.38, HF)[0]
    pre = GR.evaluate("slinky-13", need + d, 8.38, HF); live = cli("eval", "slinky-13", need + d, 8.38, HF)
    agree = (live[2] == "OK") == pre[2] == (d > 0); bad += not agree
    print(f"slinky-13 boundary free = need {d:+.2f}: pre {'OK' if pre[2] else 'SHORT'} live {live[2]} {'agree' if agree else 'DISAGREE'}")
print("GATES AGREE" if not bad else f"GATES DISAGREE ({bad})"); sys.exit(1 if bad else 0)
