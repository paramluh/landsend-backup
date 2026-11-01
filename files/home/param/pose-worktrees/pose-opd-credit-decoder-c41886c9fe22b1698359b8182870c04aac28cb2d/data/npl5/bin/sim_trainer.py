"""SIMULATED TRAINER v2 for the npl5 rolling-fetcher test (spare only; audit round 1 S12). Replays one host's real read sequence (fresh_pose.
append_stream over [plan 1, npl2, npl3, npl5], tail 1; step t micro m of local rank l reads stream row (t - 16861) x 8 + m) from --start and
before each step checks that every npl5 parquet that step reads is present at its size (what FreshRows would open). Production semantics:
  - saves every --save-every steps into a FAKE CKPT with keep=1 (PPM_KEEP_CKPT=1); the replaced save moves to a FAKE hold dir (C's chained
    hold) and is released --hold-release steps later (the HF verify); a fake pin dir holds dcp-<start> (never a floor candidate)
  - --relaunch-at S: crash + watchdog relaunch from the OLDEST save in CKPT + hold (a held save): writes 'resumed <dir> at step N', then
    --init-s seconds with NO step line (the trainer's init), then continues from N
  - --lowmem A:B: the fake trainer-job cgroup (NPL5_FAKE_CGROUP dir: memory.max / memory.current / memory.stat) reports < 6 GB headroom for
    steps [A, B) (with a large page cache that must NOT count), > 6 GB otherwise
-> --out JSON: misses (must be 0), relaunch, lowmem window, peak GB.
usage: sim_trainer.py --root R --key K --end E --stop S --dir D [--audit13] [--rate 0.5] [--fast-rate 1.0] [--relaunch-at S] [--lowmem A:B]"""
import argparse, glob, json, os, shutil, sys, time
import numpy as np
SF, NPM, LW = 16861, 8, 8
ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True); ap.add_argument("--key", required=True); ap.add_argument("--dir", required=True)
ap.add_argument("--end", type=int, required=True); ap.add_argument("--stop", type=int, required=True)
ap.add_argument("--start", type=int, default=17850); ap.add_argument("--rate", type=float, default=0.5); ap.add_argument("--fast-rate", type=float, default=1.0)
ap.add_argument("--save-every", type=int, default=52); ap.add_argument("--hold-release", type=int, default=30)
ap.add_argument("--relaunch-at", type=int, default=0); ap.add_argument("--init-s", type=float, default=25); ap.add_argument("--lowmem", default="")
ap.add_argument("--audit13", action="store_true"); ap.add_argument("--code", default="/data/home/amartyavarma/ppm/pose5/code-811-s2p")
a = ap.parse_args()
sys.path.insert(0, f"{a.code}/fresh")
import fresh_pose as FP
if a.audit13:
    ad = ["/data/home/amartyavarma/ppm/code-811/fresh/scan/plans"] + [f"/data/home/amartyavarma/ppm/code-811/fresh/append/{t}/scan/plans" for t in ("npl2", "npl3")]
    pl = [json.load(open(f"{d}/PLAN-{a.key}.json")) for d in ad] + [FP.load_plan(f"{a.root}/ppm-fresh-pose-npl5")]
else:
    pl = [FP.load_plan(f"{a.root}/{d}") for d in ("ppm-fresh-pose", "ppm-fresh-pose-npl2", "ppm-fresh-pose-npl3", "ppm-fresh-pose-npl5")]
p5 = pl[3]; z = p5["z"]; root = p5["src_root"]
S = [FP.append_stream([int(p["n_rows"]) for p in pl], l, LW, NPM, SF, 1, ()) for l in range(LW)]
def files_at(t):
    out = set()
    for l in range(LW):
        for m in range(NPM):
            q = (t - SF) * NPM + m
            if q < len(S[l]) and (int(S[l][q]) >> FP.PLAN_SHIFT) == 3:
                r = int(S[l][q]) & ((1 << FP.PLAN_SHIFT) - 1)
                out |= {p5["files"][int(j)] for j in np.unique(z["seg_file"][z["row_seg"][z["row_off"][r]:z["row_off"][r + 1]]])}
    return out
CK, HOLD, PIN, CG = (f"{a.dir}/{x}" for x in ("ckpt", "hold", "pin", "cgroup"))
for d in (CK, HOLD, f"{PIN}/dcp-{a.start}", CG):
    os.makedirs(d, exist_ok=True)
open(f"{PIN}/dcp-{a.start}/COMPLETE", "w").write("pin\n")
lm = tuple(int(x) for x in a.lowmem.split(":")) if a.lowmem else (0, 0)
def cgroup(t):                                    # 560 GB job; 300 GB page cache always (must not count); anon 500 GB inside [A, B) else 200 GB
    anon = 556e9 if lm[0] <= t < lm[1] else 200e9
    open(f"{CG}/memory.max", "w").write(f"{int(560e9)}\n"); open(f"{CG}/memory.current", "w").write(f"{int(anon + 300e9)}\n")
    open(f"{CG}/memory.stat", "w").write(f"anon {int(anon)}\nfile {int(300e9)}\n")
ptm = lambda: time.strftime("%m-%d %H:%M:%S PT")
L = open(f"{a.dir}/trainer.log", "a")
res = {"start": a.start, "stop": a.stop, "end": a.end, "misses": [], "steps": 0, "npl5_steps": 0, "relaunch": None, "peak_GB": 0.0, "lowmem": a.lowmem,
       "holds": [], "releases": []}
held = {}
t = a.start; relaunched = False; t0 = time.time()
L.write(f"[{ptm()}] resumed /pin/dcp-{a.start} at step {a.start}\n"); L.flush()
while t < a.stop:
    cgroup(t)
    L.write(f"[{ptm()}] step {t}/{a.end} 68.0s (sim)\n"); L.flush()
    fs = files_at(t)
    time.sleep(1.0 / (a.rate if fs else a.fast_rate))
    miss = [f for f in fs if not os.path.isfile(f"{root}/{f}")]
    if miss:
        res["misses"].append({"step": t, "n": len(miss), "e.g.": miss[0]})
    res["steps"] += 1; res["npl5_steps"] += bool(fs)
    t += 1
    if (t - a.start) % a.save_every == 0:
        d = f"{CK}/dcp-{t}"; os.makedirs(d, exist_ok=True); open(f"{d}/COMPLETE", "w").write("sim\n")
        for x in sorted(int(y.rsplit("-", 1)[1]) for y in glob.glob(f"{CK}/dcp-*"))[:-1]:     # keep=1: the replaced save goes to the hold
            shutil.move(f"{CK}/dcp-{x}", f"{HOLD}/dcp-{x}"); held[x] = t; res["holds"].append(x)
    for x, at in list(held.items()):
        if t - at >= a.hold_release and os.path.isdir(f"{HOLD}/dcp-{x}"):
            shutil.rmtree(f"{HOLD}/dcp-{x}"); del held[x]; res["releases"].append(x)
    if a.relaunch_at and not relaunched and t >= a.relaunch_at:
        ks = sorted(int(y.rsplit("-", 1)[1]) for y in glob.glob(f"{CK}/dcp-*") + glob.glob(f"{HOLD}/dcp-*"))
        back = ks[0] if ks else a.start
        res["relaunch"] = {"at": t, "from_oldest_save": back, "ckpt+hold": ks, "init_s_without_step_line": a.init_s}
        L.write(f"[{ptm()}] (sim) CRASH at {t}; watchdog relaunch\n[{ptm()}] resumed /fake/dcp-{back} at step {back}\n"); L.flush()
        time.sleep(a.init_s)
        t = back; relaunched = True
    if res["steps"] % 20 == 0:
        gb = sum(os.path.getsize(f"{root}/{f}") for f in p5["files"] if os.path.isfile(f"{root}/{f}")) / 1e9
        res["peak_GB"] = max(res["peak_GB"], round(gb, 2))
        print(f"sim step {t} misses {len(res['misses'])} present {gb:.2f} GB {time.time() - t0:.0f}s", flush=True)
res["seconds"] = round(time.time() - t0)
json.dump(res, open(f"{a.dir}/SIM.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "misses"}), "misses", len(res["misses"]), res["misses"][:3], flush=True)
sys.exit(0 if not res["misses"] else 3)
