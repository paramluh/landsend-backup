#!/usr/bin/env python3
"""pose5 HAND-BACK helper (main 10-27): motguard's MoT resume needs >= 64 GiB free /scratch on every host. Once the pose run's FINAL save
is HF-verified 16/16, this lists THIS RUN'S OWN removable files per host (tier order, smallest set that reaches 64 GiB) and PRINTS it for
main's OK. PRINT-ONLY: it deletes nothing and writes only its own report (kit/state/handback/handback-<time>.{txt,json}).

Modes
  live (default)   refuses (exit 2, nothing listed) unless dcp-<FINAL> is HF-verified 16/16 (setup-logs/hfb done/dcp-FINAL-final == ok + a
                   PASS verify record per physical host). Per-host inventory = ONE tiny read-only srun over the 16 hosts (-c1 --mem=1G, nice;
                   or --jobid J to overlap a running job): df /scratch, the run's paths, and per group the bytes that removing the WHOLE
                   group would free (find -printf inode/links/size: a file counts only if every hard link to it is inside the group, so a
                   pin that shares inodes with someone else's copy frees 0)
  --records        dry run on records, no host touched: per-host free from the switch's stage log (kit/logs/stage-<SJ>.log, the gate
                   line after staging) - the final save (46.09 GB) - the npl5 window still to fetch; candidate sizes from records
  --hosts h1,.. --assume-final-verified   tests on a spare only (output marked TEST)
Candidates, tier order (= removal order; per host it stops as soon as free >= 64 GiB):
  T1 superseded pose saves 17850 < N < FINAL: CKPT/ppm-v1/dcp-N + ckpt-keep-pose5/hold/dcp-N (one group), only if dcp-N is HF-verified for
     THIS host and hfb's state/dcp-N.hosts names one of this run's jobs (LAUNCHED_RUN J + watchdog relaunches)
  T2 npl5: npl-append/npl5 (window; on Weka-served hosts only the symlink + fetcher state) + ppm-fresh-pose-npl5 (plan); HF-sourced rows,
     regenerable (B's src_fetch); only after the npl5 roll supervisor is stopped (end_of_run stops it)
  T3 npl-append/npl3 (plan-3 rows, read until 18022; HF-sourced)
  T4 npl-append/npl2 (plan-2 rows, not read after 17850; HF-sourced)
  T5 the dcp-17850 pin: ckpt-keep-pose5/dcp-17850 (+ CKPT/dcp-17850 if still there): dcp-17850 HF-verified 16/16 (ppm-v1-c) + the Weka copy
  T6 slinky-15 only: the slot-9 PPM pose store pose-s3-samdata/rows, only if its NODE.json says slinky-9 (created by the pose5 switch;
     source on /data: pose5/slot9). FLAG: MoT's mot.env names this path (PPM_POSE_ROOT); MoT ran on s15 without it -> main + motguard OK
  (0 B, listed for tidiness only: pose5-resume links)
NEVER listed: the final save (the run's resume point; if a host is still short its size is printed for main), dcp-16861 (PROTECT), any save
not written by this run, plan-1 rows (new-pose-llm-data: shared dataset), s3-rows / pose-s3-samdata on other hosts / txtdup, anything under
ppm-mot*, next-arch, ppm-next, mot*, any MoT cache (MoT's dcp-14839 and its hold live under /scratch/amartyavarma/ppm-mot)."""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--records", action="store_true")
ap.add_argument("--final", type=int, default=0, help="final step (default: kit/state/LAUNCHED_RUN M)")
ap.add_argument("--jobid", default="", help="overlap this running job for the inventory step")
ap.add_argument("--hosts", default="", help="tests: comma list (spares only)")
ap.add_argument("--assume-final-verified", action="store_true", help="tests only")
ap.add_argument("--need-gib", type=float, default=64.0)
a = ap.parse_args()
os.environ["TZ"] = "America/Los_Angeles"; time.tzset()
P5, K = "/data/home/amartyavarma/ppm/pose5", "/data/home/amartyavarma/ppm/pose5/kit"
HFB, S = "/data/home/amartyavarma/setup-logs/hfb", "/scratch/amartyavarma"
GIB, SAVE_GB = 2 ** 30, 46.09
NEED = a.need_gib * GIB
TEST = bool(a.hosts or a.assume_final_verified)
sys.path.insert(0, K); import gate_rules as GR  # noqa: E402

LR = dict(re.findall(r"^(\w+)=(.*)$", open(f"{K}/state/LAUNCHED_RUN").read(), re.M)) if os.path.isfile(f"{K}/state/LAUNCHED_RUN") else {}
FINAL = a.final or int(LR.get("M", "0") or 0)
JOBS = set(re.findall(r"^J=(\d+)", open(f"{K}/state/LAUNCHED_RUN").read(), re.M)) if os.path.isfile(f"{K}/state/LAUNCHED_RUN") else set()   # every J= line (pause_resume.sh appends the resumed job)
try:
    JOBS |= {l.split()[1] for l in open("/data/home/amartyavarma/ppm/control/ppm-v1/watchdog_pose5_state/relaunches") if len(l.split()) > 1}
except OSError:
    pass
HOSTS = (a.hosts or open(f"{P5}/nodes16_s7_s15.txt").read().strip()).split(",")
MOT_BAD = ("/scratch/amartyavarma/ppm-mot", "/scratch/amartyavarma/next-arch", "/scratch/amartyavarma/ppm-next", "/scratch/amartyavarma/mot")


def verified(n, h):
    if not any((open(f).read().strip() == "ok") for f in glob.glob(f"{HFB}/done/dcp-{n}-final") + glob.glob(f"{HFB}/done/dcp-{n}-j*-final")):
        return False
    for f in glob.glob(f"{HFB}/verify/InternalCan__ppm-v1*__dcp-{n}__{h}.json") + glob.glob(f"{HFB}/verify/InternalCan__ppm-v1*__dcp-{n}-j*__{h}.json"):
        try:
            if json.load(open(f)).get("PASS") is True:
                return True
        except Exception:
            pass
    return False


def this_runs(n):
    for f in (f"{HFB}/state/dcp-{n}.hosts", f"{HFB}/state/lineage-archive/dcp-{n}.hosts"):
        try:
            return str(json.load(open(f)).get("job")) in JOBS
        except Exception:
            pass
    return False


def guard(p):   # hard assert: never anything MoT / outside /scratch/amartyavarma
    assert p.startswith(S + "/") and not p.startswith(MOT_BAD) and "/ppm-mot" not in p, p
    return p


INV = r'''h=$(hostname -s); S=/scratch/amartyavarma
echo "FREE $h $(df -B1 --output=avail /scratch | tail -1 | tr -dc 0-9)"
grp() { local tag=$1; shift; local ex=() x; for x in "$@"; do [ -e "$x" ] || [ -L "$x" ] && ex+=("$x"); done; [ ${#ex[@]} -gt 0 ] || return 0
  echo "GRP $h $tag $(find "${ex[@]}" -xdev -type f -printf '%i %n %s\n' 2>/dev/null | awk '{c[$1]++; n[$1]=$2; s[$1]=$3} END {for (i in c) {t+=s[i]; if (c[i]>=n[i]) f+=s[i]}; printf "%d %d", t, f}') ${ex[*]}"; }
for n in $(ls -d $S/ppm/ckpt/ppm-v1/dcp-* $S/ppm/ckpt-keep-pose5/hold/dcp-* 2>/dev/null | grep -oE '/dcp-[0-9]+$' | grep -oE '[0-9]+' | sort -un); do
  c=$([ -f $S/ppm/ckpt/ppm-v1/dcp-$n/COMPLETE ] && echo C)$([ -f $S/ppm/ckpt-keep-pose5/hold/dcp-$n/COMPLETE ] && echo H)
  [ "$n" = 17850 ] || grp "save-$n-${c:-none}" $S/ppm/ckpt/ppm-v1/dcp-$n $S/ppm/ckpt-keep-pose5/hold/dcp-$n; done
grp pin17850 $S/ppm/ckpt-keep-pose5/dcp-17850 $S/ppm/ckpt/ppm-v1/dcp-17850
grp npl5 $S/npl-append/npl5 $S/ppm-fresh-pose-npl5
grp npl3 $S/npl-append/npl3
grp npl2 $S/npl-append/npl2
grp resume $S/pose5-resume
echo "NODE $h $(grep -o '"node": *"[^"]*"' $S/pose-s3-samdata/rows/NODE.json 2>/dev/null | cut -d'"' -f4)"
grp slot9store $S/pose-s3-samdata/rows
echo "END $h"
'''


def inventory():
    unset = [f"-u{k}" for k in os.environ if re.match(r"^(SLURM|SBATCH|SRUN|SALLOC|PPM)_", k)]
    cmd = ["timeout", "900", "env", *unset, "srun", *(["--jobid", a.jobid, "--overlap"] if a.jobid else ["-p", "all", "-t", "15", "-J", "pose5-handback"]),
           f"-N{len(HOSTS)}", f"-n{len(HOSTS)}", "--ntasks-per-node=1", "-w", ",".join(HOSTS), "-c1", "--mem=1G", "--cpu-bind=none",
           "nice", "-n", "10", "bash", "-c", INV]
    r = subprocess.run(cmd, capture_output=True, text=True); out = r.stdout
    if r.returncode != 0 or not out.strip():
        print(f"(inventory step rc {r.returncode}; stderr: {r.stderr.strip()[-400:]})")
    inv = {h: {"free": None, "grp": {}, "node": "", "end": False} for h in HOSTS}
    for ln in out.splitlines():
        p = ln.split()
        if len(p) >= 2 and p[1] in inv:
            h = p[1]
            if p[0] == "FREE" and len(p) > 2 and p[2].isdigit(): inv[h]["free"] = int(p[2])
            elif p[0] == "GRP" and len(p) >= 6: inv[h]["grp"][p[2]] = {"total": int(p[3]), "free": int(p[4]), "paths": p[5:]}
            elif p[0] == "NODE": inv[h]["node"] = p[2] if len(p) > 2 else ""
            elif p[0] == "END": inv[h]["end"] = True
    return inv


def items_live(h, v):
    out = []
    for tag, g in sorted(v["grp"].items(), key=lambda t: t[0]):
        if not tag.startswith("save-"): continue
        n = int(tag.split("-")[1])
        if n == FINAL:
            v["final_GB"] = g["free"] / 1e9; continue
        if 17850 < n < FINAL and this_runs(n) and verified(n, h):
            out.append(("T1", f"superseded pose save dcp-{n} (HF-verified on {h}; job {sorted(JOBS)})", g))
        else:
            v.setdefault("kept", []).append(f"dcp-{n} ({g['free'] / 1e9:.1f} GB): " + ("NOT this run's / not HF-verified for this host" if n != FINAL else "final"))
    for t, tag, why in (("T2", "npl5", "npl5 window + plan (HF-sourced rows, regenerable; after the roll supervisor stopped)"),
                        ("T3", "npl3", "npl3 rows (plan 3, read until 18022; HF-sourced)"), ("T4", "npl2", "npl2 rows (plan 2, not read after 17850; HF-sourced)")):
        if tag in v["grp"]: out.append((t, why, v["grp"][tag]))
    if "pin17850" in v["grp"] and verified(17850, {"slinky-15": "slinky-9"}.get(h, h)):
        out.append(("T5", "dcp-17850 pin (+ CKPT copy if present): HF-verified 16/16 + Weka copy", v["grp"]["pin17850"]))
    if h == "slinky-15" and "slot9store" in v["grp"] and v["node"] == "slinky-9":
        out.append(("T6", "slot-9 PPM pose store (pose5-created; NODE slinky-9) [FLAG: MoT's mot.env names this path -> main + motguard OK]", v["grp"]["slot9store"]))
    for _, _, g in out:
        for p in g["paths"]: guard(p)
    return out


def records():
    sj = LR.get("SJ", "")
    inv = {}
    for ln in open(f"{K}/logs/stage-{sj}.log", errors="replace") if sj and os.path.isfile(f"{K}/logs/stage-{sj}.log") else []:
        m = re.search(r"STAGEHOST (slinky-\d+) \S+ OK free ([0-9.]+) GB", ln)
        if m: inv.setdefault(m.group(1), {})["free_stage"] = float(m.group(2))
    cur = None
    for ln in open(f"{K}/logs/stage-{sj}.log", errors="replace") if sj and os.path.isfile(f"{K}/logs/stage-{sj}.log") else []:
        m = re.search(r"window still to fetch ([0-9.]+)\)", ln)
        if m: cur = float(m.group(1))
        m2 = re.search(r"STAGEHOST (slinky-\d+) ", ln)
        if m2 and cur is not None:
            inv.setdefault(m2.group(1), {})["pend"] = cur; cur = None
    pins = {"slinky-1", "slinky-2", "slinky-7", "slinky-14"}     # c/DCP17850_LOCATION.json launch_day_staging verify_only (+ pinned)
    res = {}
    for h in HOSTS:
        r = inv.get(h, {})
        if "free_stage" not in r:
            res[h] = {"free": None}; continue
        f_end = (r["free_stage"] - SAVE_GB - r.get("pend", 0.0)) * 1e9
        w = 0.0 if h in GR.WEKA_HOSTS else GR.windows(h).get(100, 11.7)
        its = [("T2", "npl5 window + plan (est: B's a100 window + 0.26 plan)", {"free": int((w + 0.26) * 1e9), "paths": [f"{S}/npl-append/npl5", f"{S}/ppm-fresh-pose-npl5"]}),
               ("T3", "npl3 rows (est ~10 GB, B)", {"free": int(10e9), "paths": [f"{S}/npl-append/npl3"]}),
               ("T4", "npl2 rows (est ~14 GiB: npl2's row share of B's 42 GiB plan1+npl2 figure)", {"free": int(14 * GIB), "paths": [f"{S}/npl-append/npl2"]})]
        if h in pins:
            its.append(("T5", "dcp-17850 pin (46.09 GB; HF-verified 16/16 + Weka copy)", {"free": int(SAVE_GB * 1e9), "paths": [f"{S}/ppm/ckpt-keep-pose5/dcp-17850"]}))
        if h == "slinky-15":
            its.append(("T6", "slot-9 PPM pose store (9.2 GB) [FLAG: MoT-named path -> main + motguard OK]", {"free": int(9.2e9), "paths": [f"{S}/pose-s3-samdata/rows"]}))
        res[h] = {"free": f_end, "items": its, "final_GB": SAVE_GB, "src": f"stage-{sj} free {r['free_stage']} GB - final {SAVE_GB} - window to fetch {r.get('pend', 0.0)}"}
    return res


def main():
    stamp = time.strftime("%m%d-%H%M%S")
    lines = [f"pose5 HAND-BACK free-64GiB plan ({'RECORDS DRY RUN' if a.records else 'LIVE'}{' TEST' if TEST else ''}) {time.strftime('%Y-%m-%d %H:%M:%S %Z')}; "
             f"final dcp-{FINAL}; this run's jobs {sorted(JOBS)}; need >= {a.need_gib:.0f} GiB free /scratch per host. PRINT-ONLY: nothing deleted."]
    if a.records:
        res = records()
    else:
        miss = [h for h in HOSTS if not verified(FINAL, h)]
        if (miss or not FINAL) and not a.assume_final_verified:
            print(lines[0] + f"\nREFUSED: final dcp-{FINAL} not HF-verified 16/16 yet (missing: {miss[:6]}); nothing listed. Re-run after hfb verifies it.")
            return 2
        inv = inventory()
        res = {}
        for h in HOSTS:
            v = inv[h]
            if v["free"] is None or not v["end"]:
                res[h] = {"free": None}; continue
            res[h] = {"free": v["free"], "items": items_live(h, v), "final_GB": v.get("final_GB"), "kept": v.get("kept", [])}
    plan, tot, short = {}, 0, []
    for h in HOSTS:
        r = res[h]
        if r.get("free") is None:
            lines.append(f"{h:>10}: NO DATA (inventory step missing for this host) -> re-run"); short.append(h); continue
        fr, take = r["free"], []
        for t, why, g in r["items"]:
            if fr >= NEED: break
            if g["free"] <= 0: continue
            take.append((t, why, g)); fr += g["free"]
        tot += sum(g["free"] for _, _, g in take)
        st = "OK" if fr >= NEED else "SHORT"
        if st == "SHORT": short.append(h)
        lines.append(f"{h:>10}: free {r['free'] / GIB:6.1f} GiB -> {fr / GIB:6.1f} GiB after {len(take)} item(s) [{st}]"
                     + (f"  ({r['src']})" if r.get("src") else ""))
        for t, why, g in take:
            lines.append(f"{'':>12}{t} {g['free'] / 1e9:6.1f} GB  {' + '.join(g['paths'])}  :: {why}")
        if st == "SHORT":
            fg = r.get("final_GB")
            lines.append(f"{'':>12}STILL SHORT by {(NEED - fr) / GIB:.1f} GiB: " + (f"the HF-verified final dcp-{FINAL} ({fg:.1f} GB) is NOT listed (the run's "
                         "resume point): main decides" if fg else f"(no dcp-{FINAL} on this host; nothing else of this run's here)"))
        rest = [(t, g) for t, _, g in r["items"] if all(g is not x for _, _, x in take)]
        if rest:
            lines.append(f"{'':>12}(available, not needed: " + ", ".join(f"{t} {g['free'] / 1e9:.1f} GB" for t, g in rest) + ")")
        for k in r.get("kept", []):
            lines.append(f"{'':>12}(kept: {k})")
        plan[h] = {"free_now_GiB": round(r["free"] / GIB, 2), "free_after_GiB": round(fr / GIB, 2), "status": st,
                   "remove": [{"tier": t, "GB": round(g["free"] / 1e9, 2), "paths": g["paths"], "why": why} for t, why, g in take]}
    lines.append(f"TOTAL to remove: {tot / 1e9:.1f} GB on {sum(1 for v in plan.values() if v['remove'])} host(s); "
                 + (f"SHORT/NO DATA: {short}" if short else "every host reaches the target"))
    lines.append("/data (not needed for the 64 GiB rule; main decides separately): npl5 Weka windows "
                 + ", ".join(f"{h} {GR.weka_window_gb(h):.1f} GB" for h in sorted(GR.WEKA_HOSTS)) + "; Weka dcp-17850 copy pose5/dcp17850-weka (~507 GB)")
    lines.append("Deletion is NOT done here: after main's OK, remove exactly the listed paths (rm with ${VAR:?} guards / literal paths), one host at a time.")
    txt = "\n".join(lines)
    print(txt)
    od = f"{K}/state/handback"; os.makedirs(od, exist_ok=True)
    base = f"{od}/handback-{'records' if a.records else 'live'}{'-TEST' if TEST else ''}-{stamp}"
    open(base + ".txt", "w").write(txt + "\n"); json.dump({"final": FINAL, "jobs": sorted(JOBS), "need_GiB": a.need_gib, "hosts": plan}, open(base + ".json", "w"), indent=1)
    print(f"(report: {base}.txt / .json)")
    return 0 if not short else 3


sys.exit(main())
