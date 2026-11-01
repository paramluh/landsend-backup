#!/bin/bash
# pose5 SWITCH (MoT -> the 8,1,1 pose run). ONE command, run by main on the user's word:
#   bash /data/home/amartyavarma/ppm/pose5/switch_to_pose.sh --armed            # read-only readiness check (exit 0 = ARMED)
#   bash /data/home/amartyavarma/ppm/pose5/switch_to_pose.sh --dry-run          # no-op rehearsal of the whole chain (touches neither MoT nor the hosts)
#   bash /data/home/amartyavarma/ppm/pose5/switch_to_pose.sh --go               # THE SWITCH (needs ARMED; run detached: setsid nohup ... &)
#   options: --max-steps N (default = the full npl5 pass, env PPM_MAX_STEPS from B's dry run; constant LR, so no maths change; watchdog/chain
#            TARGET follow it) | --fallback-two-leg (documented fallback only: prints the flags-off leg-1 commands; see kit/FALLBACK_TWO_LEG.txt)
# Chain (--go): 1 MoT pause (motguard's mot_pause.sh --export=none, never edited) -> wait for its "GPUS FREE for pose5" line (abort + alert on
#   failure; MoT stays resumable) -> 2 HF watcher FLOOR_GB=20 (s13's 20 GB floor; restarted only if needed) -> 3 pose5-stage job on the 16 hosts
#   (kit/stage_host.sh: 17798 drop, s13 caches, pins, resume links, B's hooks, slot-9 pose store, fail-closed gate) -> 4 ppm811 afterok (env =
#   prod_811_fresh_append5_s2p.env, code-811-s2p, PPM_RESUME_FROM = /scratch/amartyavarma/pose5-resume/dcp-17850) -> 5 watchdog (RESUME_PIN),
#   chained hold, eval logger, B's npl5 roll daemons, live page refresh, end-of-run hand-back watcher -> 6 verify: dcp-17850 loaded, guard
#   lines, steps 17851/17852 losses sane vs 27653 -> "POSE RUN LIVE <job>".
set -u
P5=/data/home/amartyavarma/ppm/pose5; K=$P5/kit; A=$P5/code-811-s2p; ENV0=${TEST_ENV0:-$P5/prod_811_fresh_append5_s2p.env}; NODEF=$P5/nodes16_s7_s15.txt
CTL=/data/home/amartyavarma/ppm/control/ppm-v1; MD=/data/home/amartyavarma/ppm/control/overnight_monitor_811f.md
MOTP=/data/home/amartyavarma/ppm/mot/guard/stop/mot_pause.sh; MOTJ=$(cat /data/home/amartyavarma/ppm/mot/JOB 2>/dev/null)
HFB=/data/home/amartyavarma/setup-logs/hfb; RL=/scratch/amartyavarma/pose5-resume/dcp-17850; RS=17850
HOOKS=${HOOKS:-$P5/npl5/SWITCH_HOOKS.env}     # workstream B fills: B_NPL5_PREP B_ROLL_START B_ROLL_STOP ROLL_GB_LEFT B_DRY (see kit/SWITCH_HOOKS.example)
SPARE=${SPARE:-slinky-4}; SPARE2=${SPARE2:-slinky-4}    # main 10-27: slinky-10/18 go to motasr2 at the switch: never them
MODE=armed; MAXS=""; FALLBACK=0
AFTERPAUSE=0
for a in "$@"; do case $a in --armed) MODE=armed;; --dry-run) MODE=dry;; --go) MODE=go;; --after-pause) MODE=go; AFTERPAUSE=1;; --max-steps=*) MAXS=${a#*=};; --fallback-two-leg) FALLBACK=1;;
  *) echo "usage: switch_to_pose.sh --armed | --dry-run | --go | --after-pause [--max-steps=N] [--fallback-two-leg]"; exit 2;; esac; done
MSD=/data/home/amartyavarma/ppm/mot/guard/stop        # motguard's markers: GPUS_FREE (first line "GPUS FREE for pose5"), PAUSE_ABORT, PAUSE_DONE
LOG=$K/logs/switch-$(date +%m%d-%H%M%S)-$MODE.log; mkdir -p $K/logs $K/state
exec > >(tee -a "$LOG") 2>&1
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
say() { echo "[$(pt)] $*"; }
FAILS=(); fail() { FAILS+=("$*"); say "  NOT READY: $*"; }; okm() { say "  ok: $*"; }
die() { say "ABORT: $*"; echo "- $(pt): [pose5 switch] ABORT: $*" >> $MD; exit 1; }
UNSET=$(env | grep -oE '^(SLURM|SBATCH|SRUN|SALLOC|PPM|ILP|MOT)_[A-Za-z0-9_]*=' | tr -d = | sed 's/^/-u /' | tr '\n' ' ')
csb() { env $UNSET sbatch "$@"; }
ev() { grep -E "^$1=" $ENV0 | tail -1 | cut -d= -f2-; }
if [ -n "${TEST_ENV0:-}${TEST_SP:-}${TEST_GP:-}${TEST_CD:-}" ]; then        # offline tests of the ARMED checks only (env / PASS / cfg-diff stand-ins)
  echo "TEST OVERRIDES: env=${TEST_ENV0:-} sp=${TEST_SP:-} gp=${TEST_GP:-} cd=${TEST_CD:-}"; [ "$MODE" = armed ] || { echo "ABORT: TEST_* overrides are for --armed only"; exit 2; }
fi
say "pose5 switch ($MODE) log $LOG"
if [ $FALLBACK = 1 ]; then cat $K/FALLBACK_TWO_LEG.txt; exit 0; fi
B_NPL5_PREP="bash /data/home/amartyavarma/ppm/pose5/npl5/bin/npl5_prep.sh"   # B 10-27 (needs npl5/state/dryrun.ok); SWITCH_HOOKS.env may override
B_ROLL_START='csb --parsable --export=ALL,END=$M,AHEAD=100,LOW=30 /data/home/amartyavarma/ppm/pose5/npl5/bin/npl5_roll_super.sbatch'   # B FINAL 10-27: once,
#   after CURRENT_JOB_811 is written (slinky-4 -c1 --mem=1G; per-host ahead/back from kit/state/AHEAD-/BACKMODE-<host>; stop: touch npl5/state/STOP_ROLL)
B_ROLL_STOP="touch /data/home/amartyavarma/ppm/pose5/npl5/state/STOP_ROLL"; ROLL_GB_LEFT=auto; B_DRY=$P5/npl5/dryrun   # B's dryrun_offhost.sh output   # auto = each host reserves its window minus npl5 already present
[ -f $HOOKS ] && . $HOOKS

# ================================================================ ARMED check (every mode; read-only)
say "ARMED check"
HOSTS=$(cat $NODEF)
[ "$(tr , '\n' < $NODEF | sort -V | paste -sd,)" = "slinky-0,slinky-1,slinky-2,slinky-3,slinky-5,slinky-6,slinky-7,slinky-8,slinky-11,slinky-12,slinky-13,slinky-14,slinky-15,slinky-16,slinky-17,slinky-19" ] \
  && okm "hosts = MoT's 16 (no slinky-9)" || fail "hosts file $NODEF"
V=$(ev PPM_S2P); case $V in cb0) VN=cb0;; 1|full) VN=full;; *) VN=""; fail "env PPM_S2P=$V (cb0 | 1)";; esac
[ -n "$VN" ] && okm "s2p variant: $VN (env line PPM_S2P=$V)"
[ "$(ev PPM_G0COND)" = 1 ] && okm "PPM_G0COND=1" || fail "env PPM_G0COND"
[ "$(ev PPM_KEEP_CKPT)" = 1 ] && okm "PPM_KEEP_CKPT=1 (+ chained hold)" || fail "env PPM_KEEP_CKPT"
[ "$(ev PPM_EXPORT_KEEP)" = 0 ] && okm "PPM_EXPORT_KEEP=0 (last line wins over append3's 2: the trainer never rotates exports; w-15973 safe)" \
  || fail "env PPM_EXPORT_KEEP=$(ev PPM_EXPORT_KEEP) (must be 0: audit 2 R2-7, else rotation deletes w-15973 at the 2nd pose export)"
[ "$(ev PPM_HOST_ALIAS)" = "slinky-7:slinky-10,slinky-15:slinky-9" ] || fail "env PPM_HOST_ALIAS"
[ "$(ev PPM_RESUME_FROM)" = auto ] || fail "env PPM_RESUME_FROM must stay auto"
echo "$(ev PPM_FRESH_POSE_ROOT2)" | grep -q 'ppm-fresh-pose-npl5$' || fail "env ROOT2 lacks npl5"
diff -q <(sed -n 1,308p $ENV0) /data/home/amartyavarma/ppm/code-811/prod_811_fresh_append3.env >/dev/null && okm "env lines 1-308 == append3 (verbatim)" || fail "env not verbatim"
M=${MAXS:-$(ev PPM_MAX_STEPS)}
[[ "$M" =~ ^[0-9]+$ ]] && [ "$M" -gt "$RS" ] && okm "max steps $M$([ -n "$MAXS" ] && echo ' (--max-steps)')" || fail "PPM_MAX_STEPS=$M (B's dry run fills it)"
grep -q "^C=$A;" $A/ppm811.sbatch && okm "code-811-s2p sbatch runs its own code" || fail "$A/ppm811.sbatch C= line"
grep -q PPM_S2P $A/train_ppm.py && grep -q PPM_G0COND $A/train_ppm.py && okm "trainer knows PPM_S2P + PPM_G0COND" || fail "trainer lacks PPM_S2P/PPM_G0COND (D's merge?)"
SP=${TEST_SP:-$A/tests/s2p_postmerge/S2P_PASS}; GP=${TEST_GP:-$A/tests/g0/G0_PASS}   # main 10-27: ONLY the post-merge S2P proof counts (no fallback to tests/s2p)
python3 -c "import json,sys; d=json.load(open('$SP')); v=d.get('variants', d) if isinstance(d, dict) else d; sys.exit(0 if ((v.get('$VN') is True) if isinstance(v, dict) else ('$VN' in v)) and (not isinstance(d, dict) or d.get('flag_off_byte_identical', True) is True) else 1)" 2>/dev/null \
  && okm "A's S2P_PASS lists '$VN'" || fail "A's $SP missing / does not list '$VN'"
python3 -c "import json,sys; d=json.load(open('$GP')); c=d.get('combinations_passed', []); sys.exit(0 if d.get('PASS') is True and 'g0cond+s2p=$VN' in c and 's2p=$VN' in c else 1)" 2>/dev/null \
  && okm "D's G0_PASS: PASS, combinations g0cond+s2p=$VN + s2p=$VN certified" || fail "D's $GP missing / not PASS / lacks 'g0cond+s2p=$VN' or 's2p=$VN' in combinations_passed"
for PF in $SP $GP; do                                    # audit 1 #13: each PASS must certify THESE files (md5 recorded by A / D)
  [ -f $PF ] || continue                                 # a missing PASS is already a failure above
  python3 - "$PF" "$A" <<'EOF' && okm "$(basename $PF): core_md5 == code-811-s2p's files + groups field" || fail "$(basename $PF): core_md5 / groups missing, or md5 != current code-811-s2p (re-run the proofs after the merge)"
import hashlib, json, os, sys
d = json.load(open(sys.argv[1])); n = os.path.basename(sys.argv[1])
core = {"model_ppm.py", "train_ppm.py", "data_ppm.py", "safety_ppm.py"}
m = d.get("core_md5")                                  # main 10-27: D writes core_md5 (model/train/data/safety + changed files) into BOTH markers; required, no fallback
if not isinstance(m, dict) or not core <= set(m): print(f"   {n}: core_md5 missing or lacks {sorted(core - set(m or {}))}"); sys.exit(1)
for extra in ("md5_certified_changed_files", "md5_exercised_unchanged_files"):   # any other md5 map in the marker must agree too
    m = {**d.get(extra, {}), **m}
bad = [f for f, h in m.items() if hashlib.md5(open(os.path.join(sys.argv[2], f), "rb").read()).hexdigest() != h]
if bad: print(f"   {n}: md5 != current for {bad}"); sys.exit(1)
g = d.get("groups")
if not isinstance(g, list) or not g: print(f"   {n}: no 'groups' list"); sys.exit(1)
print(f"   {n}: {len(m)} files md5-bound, groups {g}")
EOF
done
CD=${TEST_CD:-$K/state/CFG_DIFF_switch.txt}
if [ -f $CD ] && [ $CD -nt $ENV0 ] && [ $CD -nt $A/train_ppm.py ]; then
  python3 - "$CD" "$VN" "$M" "$RL" "$SP" "$GP" <<'EOF' && okm "resolved cfg vs 27653: only the expected keys (see $CD)" || fail "resolved cfg diff has unexpected keys/values (above; $CD)"
import json, sys
cd, vn, m, rl, sp, gp = sys.argv[1:7]
rows, k = {}, None
for ln in open(cd):
    if ln.startswith("#") or not ln.strip(): continue
    if not ln.startswith(" "): k = ln.strip(); rows[k] = {}
    elif ":" in ln: side, v = ln.split(":", 1); rows[k][side.strip()] = json.loads(v.strip())
groups, gerr = set(), []
for f, want in ((sp, ["s2p"]), (gp, ["g0cond"])):     # main 10-27: s_emb / inp.g0 in their OWN optimizer groups; names come ONLY from the PASS JSONs
    try: g = json.load(open(f)).get("groups")
    except Exception: g = None
    if g != want: gerr.append(f"{f.split('/')[-1]} groups {g} != {want}")
    groups |= set(g or [])
if groups != {"s2p", "g0cond"}: gerr.append(f"PASS groups {sorted(groups)} != cross-check {{s2p, g0cond}}")
if gerr: print("   BAD " + "; ".join(gerr)); groups = set()
bad = []
for k, v in rows.items():
    old, new = v.get("PPM"), v.get("8,1,1")
    ok = {"fresh.pose_root2": lambda: new == (old or []) + ["/scratch/amartyavarma/ppm-fresh-pose-npl5"], "max_steps": lambda: str(new) == m,
          "resume_from": lambda: new == rl, "run_tag": lambda: True, "s2p": lambda: new == vn, "s2p_from_step": lambda: new in (0, 17850),
          "keep_ckpt": lambda: new == 1}.get(k)
    if ok: good = ok()
    elif k.startswith("g0cond"): good = True
    elif k == "warmup_groups": good = isinstance(new, list) and new[:len(old or [])] == (old or []) and set(new[len(old or []):]) <= groups
    else: good = bool(groups) and k.split(".")[-1] in groups and old in (None, "<absent>")
    print(f"   {'ok ' if good else 'BAD'} {k}: {json.dumps(old)[:60]} -> {json.dumps(new)[:80]}")
    if not good: bad.append(k)
import re
ga = [l for l in open(cd) if l.startswith("# GROUPS added:")]
if not ga: print("   BAD no GROUPS line (re-run kit/cfgdiff_switch.sh)"); bad.append("GROUPS")
else:
    add = re.findall(r"'([^']+)'", ga[0].split("removed:")[0]); rem = re.findall(r"'([^']+)'", ga[0].split("removed:")[1])
    good = not rem and set(add) == groups and len(add) == 2
    print(f"   {'ok ' if good else 'BAD'} optimizer GROUPS added {add} removed {rem} (expected exactly the PASS JSON groups {sorted(groups)})")
    if not good: bad.append("GROUPS")
for need in ("fresh.pose_root2", "max_steps", "s2p"):
    if need not in rows: print(f"   BAD missing expected difference {need}"); bad.append(need)
if not any(k.startswith("g0cond") for k in rows): print("   BAD no g0cond key"); bad.append("g0cond")
sys.exit(1 if bad else 0)
EOF
else fail "cfg diff $CD missing or older than env/code (run: bash $K/cfgdiff_switch.sh)"; fi
[ -f $P5/dcp17850-weka/VERIFIED.json ] && okm "Weka dcp-17850 copy VERIFIED (11 dirs)" || fail "Weka dcp-17850 copy not VERIFIED"
python3 -c "import json,sys; d=json.load(open('$P5/slot9/posestore/SLOT9_POSESTORE_READY.json')); sys.exit(0 if d.get('PASS') else 1)" 2>/dev/null \
  && [ -x $P5/slot9/stage_slot9_extras.sh ] && okm "slot-9 pose store READY + hook" || fail "slot-9 pose store / hook"
[ -f $P5/npl5/bin/leg1_slot9_hook.sh ] && okm "B's slot-9 plans/npl3 hook" || fail "B's leg1_slot9_hook.sh"
for v in B_NPL5_PREP B_ROLL_START ROLL_GB_LEFT; do [ -n "${!v:-}" ] || fail "B's $v not set ($HOOKS)"; done
for f in npl5_prep.sh npl5_roll_super.sbatch; do [ -f $P5/npl5/bin/$f ] && okm "B's $f present" || fail "B's npl5/bin/$f missing"; done
[ -f $P5/npl5/state/dryrun.ok ] && okm "B's npl5 dryrun.ok" || fail "B's npl5/state/dryrun.ok missing (npl5_prep.sh needs it)"
grep -hoE -- '-w slinky-[0-9]+' $P5/npl5/bin/npl5_roll_super.sbatch 2>/dev/null | grep -qE 'slinky-(9|10|18)$' && fail "B's roll supervisor targets slinky-9/10/18"
if [ -n "${B_DRY:-}" ] && [ -f ${B_DRY}/RESULT.json ]; then
  python3 - "$B_DRY" "$M" <<'EOF' && okm "B's dry run: guard ok, append at 17850, end == max steps" || fail "B's dry run mismatch"
import json, sys
R, G = json.load(open(f"{sys.argv[1]}/RESULT.json")), json.load(open(f"{sys.argv[1]}/GUARD.json")); m = int(sys.argv[2])
ok = G.get("guard") == "ok" and G.get("pose_append_at") == [17352, 17641, 17850] and R.get("resume_step") == 17850 and G.get("resume_ckpt_step") == 17850 and R["ends"][:3] == [17352, 17641, 18022] and R["ends"][-1] == R["max_steps"] \
     and (R["max_steps"] == m or m < R["max_steps"])
print("   B:", R.get("ends"), R.get("max_steps"), G.get("guard")); sys.exit(0 if ok else 1)
EOF
else fail "B's dry run (B_DRY) missing"; fi
E2E_JOB=${E2E_JOB:-31048}                                # main 10-27: B's end-to-end dry run is part of READY (all 16 slots, real plans + HF parquet + trainer micros)
python3 - "$P5/npl5" "$E2E_JOB" "$M" <<'EOF' && okm "B's e2e $E2E_JOB: 16/16 slots OK, end == max steps" || fail "B's e2e $E2E_JOB not PASS (npl5/e2e/E2E.json + logs/e2e-$E2E_JOB.out)"
import json, os, sys
w, j, m = sys.argv[1], sys.argv[2], int(sys.argv[3])
E, L, R = f"{w}/e2e/E2E.json", f"{w}/logs/e2e-{j}.out", json.load(open(f"{w}/dryrun/RESULT.json"))
slots = ["slinky-%d" % i for i in (0, 1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 19)]
why = []
if not (os.path.exists(E) and os.path.exists(L)): why.append("E2E.json / job log missing (still running?)")
else:
    e, log = json.load(open(E)), open(L).read()
    if "e2e: 16 slots OK" not in log: why.append(f"job {j} log has no 'e2e: 16 slots OK'")
    if os.path.getmtime(E) < os.path.getmtime(L) - 120: why.append(f"E2E.json older than job {j}'s log (another run's)")
    if e.get("slots_ok") != 16: why.append(f"slots_ok {e.get('slots_ok')}")
    if e.get("end") != R["max_steps"] or m > e.get("end", 0): why.append(f"end {e.get('end')} vs max_steps {R['max_steps']} / M {m}")
    for k in slots:
        try: h = json.load(open(f"{w}/e2e/host-{k}.json"))
        except Exception: why.append(f"host-{k}.json missing"); continue
        if not (h.get("key") == k and h.get("digests_ok") is True and h.get("npl5_sha_ok") is True): why.append(f"{k} digests/sha not ok")
        if os.path.getmtime(f"{w}/e2e/host-{k}.json") > os.path.getmtime(E): why.append(f"host-{k}.json newer than E2E.json")
print("   e2e:", "; ".join(why[:6]) if why else f"16/16 slots, end {R['max_steps']}")
sys.exit(1 if why else 0)
EOF
for f in stage_host.sh pose5_stage.sbatch stage_dcp17850.py hold_chain.sh hold_chain.sbatch ppm811s2p_watchdog.sh ppm811s2p_watchdog.sbatch \
         eval_log_pose5.py eval_log_pose5.sbatch end_of_run.sh; do [ -f $K/$f ] || fail "kit file $f missing"; done
[ -x $MOTP ] && okm "motguard's mot_pause.sh present" || fail "$MOTP missing"
for x in $SPARE $SPARE2; do case $x in slinky-10|slinky-18|slinky-9) fail "companion host $x (slinky-10/18 go to motasr2; never slinky-9)";; esac; done
grep -hoE -- '-w slinky-[0-9]+' $K/ppm811s2p_watchdog_v3.sbatch $K/hold_chain_v4.sbatch $K/hfb_throttle.sbatch $K/eval_log_pose5.sbatch | grep -qE 'slinky-(9|10|18)$' && fail "a kit sbatch defaults to slinky-9/10/18"
HWN=$(squeue -h -u amartyavarma -n hfb-watch -t R -o '%i %N'); HFB_FLOOR=$(grep -a 'watcher start' $HFB/logs/watch.log | tail -1 | grep -oE 'FLOOR_GB [0-9.]+' | grep -oE '[0-9.]+$')
[ "$(echo "$HWN" | grep -c .)" = 1 ] && [ "$(echo $HWN | awk '{print $2}')" = slinky-10 ] \
  && okm "HF watcher: exactly one hfb-watch, on slinky-10 ($HWN), FLOOR_GB ${HFB_FLOOR:-?} (the pose run's only watcher; main keeps it there)" \
  || fail "HF watcher: need exactly one RUNNING hfb-watch on slinky-10 (got: ${HWN:-none})"
[ "${HFB_FLOOR%.*}" = 20 ] || fail "hfb-watch FLOOR_GB ${HFB_FLOOR:-?} (the s13 gate assumes 20)"
# main 10-27 (uploads stalled ~13:00 with no controller: every control/<host>.workers stayed 0): the HF upload throttle must run. ARMED is green
# with exactly one RUNNING hfb-throttle, or with none (step 2 then starts kit/hfb_throttle.sbatch and requires a job id); two -> red.
HTJ=$(squeue -h -u amartyavarma -n hfb-throttle -t R,PD -o '%i %T %N')
case "$(echo "$HTJ" | grep -c .)" in
  1) okm "HF upload throttle: hfb-throttle $HTJ" ;;
  0) okm "HF upload throttle: none running -> the switch starts $K/hfb_throttle.sbatch at step 2 (required: numeric job id)" ;;
  *) fail "HF upload throttle: more than one hfb-throttle job ($(echo $HTJ | tr '\n' ' '))" ;;
esac
[ -f $K/hfb_throttle.sbatch ] && [ -f $K/hold_chain_v4.sbatch ] && [ -f $K/hold_chain_v4.sh ] && [ -f $K/ppm811s2p_watchdog_v3.sbatch ] && [ -f $K/wd_decide_v2.py ] && [ -f $K/wd_loadfail.py ] && [ -f $K/restage_dcp_hf_v2.py ] \
  || fail "kit: hfb_throttle.sbatch / hold_chain_v4.* / watchdog v3 (+ wd_decide_v2.py, wd_loadfail.py, restage_dcp_hf_v2.py) missing"
OFF=$(squeue -h -u amartyavarma -o '%i|%j|%N|%o' | awk -F'|' '($3 ~ /slinky-(10|18)/) && ($2 ~ /^(pose5|npl5)/ || $4 ~ /\/ppm\/pose5\//) && $2 != "hfb-watch" && $2 !~ /^klweb/ {print $1" "$2" "$3}')
[ -z "$OFF" ] && okm "no pose5 / npl5 job on slinky-10/18" || fail "pose5/npl5 jobs still on slinky-10/18 (must be off by ARMED): $(echo $OFF | tr '\n' ';')"
HFB_FLOOR=${HFB_FLOOR:-30}
grep -q 'ppm/pose5/code-811-s2p' $HFB/snap_code.sh && okm "hfb codesnap knows code-811-s2p" || fail "hfb snap_code.sh"
for f in STOP_AFTER_SAVE SAVE_NOW WATCHDOG811_OFF; do [ -e $CTL/$f ] && fail "$CTL/$f present"; done
if [ $AFTERPAUSE = 1 ]; then head -1 $MSD/GPUS_FREE 2>/dev/null | grep -q '^GPUS FREE for pose5' && okm "--after-pause: $MSD/GPUS_FREE present" || fail "--after-pause needs $MSD/GPUS_FREE"
else for f in GPUS_FREE PAUSE_ABORT; do [ -e $MSD/$f ] && fail "stale MoT marker $MSD/$f (a pause already ran: use --after-pause or ask motguard)"; done; fi
[ -e $P5/npl5/state/STOP_ROLL ] && say "   note: npl5/state/STOP_ROLL exists (B v2: only a STOP_ROLL newer than the supervisor's start counts; harmless)"
squeue -h -u amartyavarma -o %j | grep -qxE 'ppm811|pose5-stage' && fail "a ppm811 / pose5-stage job is already queued"
DF=$(df -BG --output=avail /data | tail -1 | tr -dc 0-9); [ "$DF" -ge 330 ] && okm "/data free $DF GB (floor 300)" || fail "/data free $DF GB"
python3 - <<'EOF'
import json
D = json.load(open("/data/home/amartyavarma/ppm/pose5/c/DISK_NEED.json"))
s = [f"{h}:{v['left_after_margin_GB']:.1f}" for h, v in D["hosts"].items() if v["left_after_margin_GB"] < 0]
t = sorted((v["left_after_margin_GB"], h) for h, v in D["hosts"].items())[:3]
print(f"   disk (records {D['written']}; window {str(D['window_src'])[:60]}): " + ("hosts short " + " ".join(s) if s else "no host short")
      + "; tightest " + " ".join(f"{h} +{x:.1f}" for x, h in t) + " GB after margin -> step 0 + the stage job's live gate decide (fail closed)")
EOF
if [ ${#FAILS[@]} -gt 0 ]; then say "NOT ARMED: ${#FAILS[@]} item(s)"; [ $MODE = go ] && die "not ARMED"; [ $MODE = armed ] && exit 3
else say "ARMED"; [ $MODE = armed ] && exit 0; fi

# ================================================================ the chain
RUNENV=$K/state/run.env                                  # = the env + (only with --max-steps) one override line; launch + watchdog use it
run() { if [ $MODE = dry ]; then echo "    [dry] $*"; else eval "$@"; fi; }
# audit 2 R2-6: every sbatch id must be numeric. Stage / ppm811 / watchdog / hold chain / roll supervisor: else cancel what was submitted, stop
# the roll, and die BEFORE anything else (CURRENT_JOB_811 is written only after ppm811's id passed). Eval logger / page refresher: ALERT only.
need_id() {   # <var name> <value> [soft]
  [[ "$2" =~ ^[0-9]+$ ]] && return 0
  if [ "${3:-}" = soft ]; then echo "- $(pt): [pose5 ALERT] switch: $1 submit returned '$2' (not a job id); run continues without it" | tee -a $MD >> /data/home/amartyavarma/setup-logs/watchdog/ALERTS.log; return 0; fi
  local ids="" v x; for v in SJ J WD HC EL RS_J; do x=${!v:-}; [[ "$x" =~ ^[0-9]+$ ]] && ids="$ids $x"; done
  [ -n "$ids" ] && scancel $ids; [[ "${RS_J:-}" =~ ^[0-9]+$ ]] && eval "$B_ROLL_STOP"
  die "submit of $1 returned '$2' (not a job id): cancelled [${ids# }]; CURRENT_JOB_811 $([ "$1" = SJ ] || [ "$1" = J ] && echo 'NOT written' || echo "= $J (cancelled)"); MoT is PAUSED: fix + re-run switch_to_pose.sh --after-pause"
}
run "cp $ENV0 $RUNENV"; [ -n "$MAXS" ] && run "echo 'PPM_MAX_STEPS=$MAXS   # switch --max-steps (run length only; constant LR)' >> $RUNENV"
if [ $AFTERPAUSE = 1 ]; then say "0+1. --after-pause: MoT already paused ($(head -1 $MSD/GPUS_FREE)); resuming the switch at step 2"
else
say "0. PRE-PAUSE disk gate (freshest motguard free figures + the switch's deltas; MoT untouched if any host is short)"
DNO=$P5/c/DISK_NEED.json; [ $MODE = dry ] && DNO=$P5/c/DISK_NEED.dryrun.json   # --dry-run never overwrites the live table (stage_host reads it)
HFB_FLOOR=$HFB_FLOOR python3 $P5/c/disk/build_disk_need.py --gate --out=$DNO | sed 's/^/   /'; GRC=${PIPESTATUS[0]}; say "   table -> $DNO"
if [ "$GRC" != 0 ]; then [ $MODE = go ] && die "pre-pause disk gate: a host would be short (table above); MoT NOT paused" || say "   (dry) the pre-pause gate WOULD ABORT here"; fi
say "1. MoT pause (job ${MOTJ:-?}): motguard's mot_pause.sh --export=none; waiting for $MSD/GPUS_FREE (as long as mot_pause.sh is alive; alert every 10 min)"
MPL=$MSD/pause-$(date +%m%d-%H%M).log
if [ $MODE = dry ]; then echo "    [dry] setsid nohup $MOTP --export=none > $MPL 2>&1 &   (not called in --dry-run: no MoT probe)"
else
  echo "- $(pt): [pose5 switch] MoT pause started (mot_pause.sh --export=none, log $MPL) on the user's word" >> $MD
  setsid nohup $MOTP --export=none > $MPL 2>&1 & MP=$!; T0=$(date +%s)
  while :; do                                             # audit 1 #14: no hard timeout while the pause is alive
    head -1 $MSD/GPUS_FREE 2>/dev/null | grep -q '^GPUS FREE for pose5' && break
    [ -e $MSD/PAUSE_ABORT ] && die "MoT pause ABORTED ($MSD/PAUSE_ABORT: $(head -3 $MSD/PAUSE_ABORT | tr '\n' ' ')); do NOT launch: ask main"
    kill -0 $MP 2>/dev/null || { head -1 $MSD/GPUS_FREE 2>/dev/null | grep -q '^GPUS FREE for pose5' && break; die "mot_pause.sh exited without GPUS_FREE (see $MPL)"; }
    E=$(( $(date +%s) - T0 )); if [ $E -ge 600 ] && [ $(( E % 600 )) -lt 5 ]; then
      echo "- $(pt): [pose5 ALERT] MoT pause still running after $((E / 60)) min (no GPUS_FREE yet; log $MPL); the switch keeps waiting" | tee -a $MD >> /data/home/amartyavarma/setup-logs/watchdog/ALERTS.log; fi
    sleep 5
  done
  say "   $(head -1 $MSD/GPUS_FREE)"
fi
fi
say "2. HF watcher: the running hfb-watch (FLOOR_GB $HFB_FLOOR) adopts the new ppm811 by name; not restarted, no second watcher"
if [ -z "$(squeue -h -u amartyavarma -n hfb-throttle -t R,PD -o %i)" ]; then   # main 10-27: uploads stall without the throttle controller
  run "HT=\$(csb --parsable $K/hfb_throttle.sbatch)"; run 'need_id HT "$HT"'; say "   started hfb-throttle (HF upload controller)"
else say "   hfb-throttle running: $(squeue -h -u amartyavarma -n hfb-throttle -o '%i %T %N' | head -1)"; fi
say "3. pose5-stage on the 16 hosts (17798 drop after a live HF re-check, else ABORT + caches only where a host is short, never MoT's + pins + resume links + B's hooks + slot-9 store + gate)"
SX="DROP17798=1 CLEANCACHE=1 AUTOSHRINK=${AUTOSHRINK:-1} HFB_FLOOR=$HFB_FLOOR MARGIN_GB=5 ROLL_GB_LEFT=${ROLL_GB_LEFT:-0} B_NPL5_PREP='${B_NPL5_PREP:-}'"   # -> sbatch env
run "SJ=\$($SX csb --parsable -N 16 -w $NODEF $K/pose5_stage.sbatch)"; run 'need_id SJ "$SJ"'
say "4. ppm811 (afterok stage): env $RUNENV, code $A, resume $RL, max $M"
run "J=\$(env \$UNSET PPM_ENV=$RUNENV PPM_RESUME_FROM=$RL sbatch --parsable --dependency=afterok:\$SJ -N 16 -w $NODEF --mem=560G $A/ppm811.sbatch)"; run 'need_id J "$J"'
run "echo \$J > $CTL/CURRENT_JOB_811"
say "5. companions on spares"
run "WD=\$(ENVF=$RUNENV NODEF=$NODEF TARGET=$M MIN_STEP=$RS RESUME_PIN=$RL PIN_STEP=$RS CODE=$A csb --parsable -w $SPARE $K/ppm811s2p_watchdog_v3.sbatch)"; run 'need_id WD "$WD"'   # v3 (10-27, audit 4)
run "HC=\$(TARGET=$M PIN_STEP=$RS csb --parsable -w $SPARE $K/hold_chain_v4.sbatch)"; run 'need_id HC "$HC"'   # v4 (10-27, audit 4)
run "EL=\$(MIN_JOB=\$J END=$M MIN_STEP=$RS csb --parsable -w $SPARE $K/eval_log_pose5.sbatch)"; run 'need_id EL "$EL" soft'
run "PG=\$(squeue -h -u amartyavarma -n klweb-ppm811-refresh -o %i | head -1); [ -n \"\$PG\" ] || PG=\$(csb --parsable -w $SPARE /data/home/amartyavarma/setup-logs/klweb/ppm811_refresh.sbatch)"; run 'need_id PG "$PG" soft'
run "RS_J=\$(eval \"\$B_ROLL_START\" | tail -1); echo \"   npl5 roll supervisor \$RS_J\""; run 'need_id RS_J "$RS_J"'
run "printf 'J=%s\nM=%s\nWD=%s\nHC=%s\nEL=%s\nPG=%s\nSJ=%s\nRS_J=%s\nVARIANT=%s\nB_ROLL_STOP=%q\n' \$J $M \$WD \$HC \$EL \$PG \$SJ \$RS_J $VN \"\$B_ROLL_STOP\" > $K/state/LAUNCHED_RUN"
run "setsid nohup bash $K/end_of_run.sh > $K/logs/end_of_run-\$(date +%m%d-%H%M).log 2>&1 & echo \$! > $K/state/END_OF_RUN.pid"
run "echo \"- \$(TZ=America/Los_Angeles date '+%a %m-%d %H:%M PT'): [pose5] 8,1,1 pose run launched: train \$J (after stage \$SJ) from dcp-17850 + npl5 + s2p $VN + g0cond, max $M; watchdog \$WD, chain \$HC, eval \$EL\" >> $MD"
say "   export /data drop (hold chain, each poll; MoT's policy): candidates now (dry):"; python3 -I $K/export_drop.py --dry-run --no-live 2>&1 | sed 's/^/     /'
say "6. verify live"
if [ $MODE = dry ]; then
  echo "    [dry] wait stage COMPLETED; then in ppm/logs/ppm-<J>.log: 'resumed $RL at step 17850', '[resume] FRESH POSE APPEND ... added at step [17352, 17641, 17850]',"
  echo "          the s2p ($VN) + g0cond ADDED lines, start eval vs ppmdecay-j27810 @17850 (kit/exact_check.py: trainer key rule, >= 184 keys; exact -> LIVE, <= rel 1e-6 -> LIVE + WARN, else NOT VERIFIED + ALERT), [811 rows] 17851/17852 == decay, gn_audio / gn_tower_p < 0.5, npl5 roll health"
  say "DRY RUN COMPLETE (s2p $VN): nothing called, nothing submitted; expected wall-clock --go -> POSE RUN LIVE ~25-30 min"; exit 0
fi
for i in $(seq 1 360); do st=$(sacct -n -X -j $SJ -o State%20 | head -1 | tr -d ' ')
  case $st in COMPLETED) break;; FAILED|CANCELLED*|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY)
    scancel $J $WD $HC $EL $RS_J 2>/dev/null; eval "$B_ROLL_STOP"; kill $(cat $K/state/END_OF_RUN.pid 2>/dev/null) 2>/dev/null   # audit 1 #14: stop everything
    die "stage job $SJ $st (see $K/logs/stage-$SJ.log): ppm811 $J never started; roll supervisor $RS_J, watchdog, chain, eval logger, end-of-run stopped (STOP_ROLL touched: remove it before a re-run); MoT is PAUSED (fix + re-run: switch_to_pose.sh --after-pause)";; esac; sleep 10; done
L=/data/home/amartyavarma/ppm/logs/ppm-$J.log
for i in $(seq 1 180); do grep -aq "step 17852/" $L 2>/dev/null && break; grep -aqE 'Traceback|Error|ABORT' $L 2>/dev/null && die "trainer error in $L"; sleep 10; done
grep -aq "resumed $RL at step 17850" $L || die "no 'resumed $RL at step 17850' in $L"
# audit 1 #8 / N2 / #7: exactness in the deployed layout (zero init => the start eval and the rows must equal the flags-off decay branch, job
# 27810, which resumed the same dcp-17850 on the same 16 slots with code-811 maths), the two ADDED lines (load_local asserts the missing keys
# are exactly these), and the clip-group grad norms. Not verified -> no "POSE RUN LIVE" (the run keeps going; main decides).
# audit 2 R2-1: start-eval exactness with the TRAINER's key rule (eval_ / codependence/ / xdep/, minus *time_s; wall-clock keys excluded):
# exact -> LIVE; every key within rel 1e-6 (slot 9 now on slinky-15) -> LIVE + WARN listing the keys; larger / key mismatch -> NOT VERIFIED + ALERT
python3 -I $K/exact_check.py /data/home/amartyavarma/ppm/runs/ppm-v1-j$J/metrics.jsonl /data/home/amartyavarma/ppm/runs/ppmdecay-j27810/metrics.jsonl 17850 \
  --label ppmdecay-j27810 --warn-rel ${EXACT_WARN_REL:-1e-6} > $K/state/exact_check.txt 2>&1; XRC=$?; cat $K/state/exact_check.txt
python3 - "$L" "$J" "$VN" <<'EOF'; VRC=$?
import json, re, sys
L, J, VN = sys.argv[1], sys.argv[2], sys.argv[3]
log = open(L, errors="replace").read(); bad = []
for pat in (r"\[resume\] POSE-S2P ADDED at step 17850: tower_p\.s_emb \(" + ("cb0" if VN == "cb0" else "full") + r"\)",
            r"\[resume\] G0COND ADDED at step 17850: inp\.g0\.weight", r"added at step \[17352, 17641, 17850\]"):
    if not re.search(pat, log): bad.append(f"missing log line /{pat}/")
rows = lambda t: {m.group(1): m.group(2) for m in re.finditer(r"\[811 rows\] (rank \d+ step 1785[12]): (.*)", t)}
rn, rd = rows(log), rows(open("/data/home/amartyavarma/ppm/logs/ppmdecay-27810.log", errors="replace").read())
if not rn or rn != {k: v for k, v in rd.items() if k in rn} or len(rn) < 4: bad.append(f"[811 rows] 17851/17852 != decay branch ({len(rn)} lines)")
else: print(f"   [811 rows] steps 17851/17852: {len(rn)} lines == ppmdecay-27810")
dlog = open("/data/home/amartyavarma/ppm/logs/ppmdecay-27810.log", errors="replace").read()
for st in ("17851", "17852"):                         # existing groups' grad norms vs the decay branch at the same step (report); new groups separately
    ln_ = [l for l in log.splitlines() if f"] step {st}/" in l]; ld_ = [l for l in dlog.splitlines() if f"] step {st}/" in l]
    if ln_ and ld_:
        gn = lambda l: dict((k, float(v)) for k, v in re.findall(r"train/gn_(\w+)=([0-9.]+)", l))
        a_, b_ = gn(ln_[-1]), gn(ld_[-1])
        print(f"   step {st} gn vs decay branch: " + " ".join(f"{k} {a_[k]:.4f}/{b_[k]:.4f}" for k in sorted(a_) if k in b_)
              + " | new groups: " + (" ".join(f"{k} {a_[k]:.4f}" for k in sorted(a_) if k not in b_) or "none"))
for l in [l for l in log.splitlines() if re.search(r"\] step 1785[12]/", l)]:
    for k in ("audio", "tower_p"):
        m = re.search(rf"train/gn_{k}=([0-9.]+)", l); st_ = re.search(r"step \d+", l).group(0)   # (py3.10: no backslash inside f-string braces)
        if not m or float(m.group(1)) >= 0.5: bad.append(f"gn_{k} {m.group(1) if m else 'missing'} at {st_}")
    print("  ", re.search(r"step \d+", l).group(0), " ".join(re.findall(r"train/gn_(?:audio|tower_p)=[0-9.]+", l)))
print("   NOT VERIFIED: " + "; ".join(bad) if bad else "   exactness + guards: all OK"); sys.exit(1 if bad else 0)
EOF
# main 10-27: the npl5 fetchers must be healthy in the LIVE job before "POSE RUN LIVE" (B's LOWMEM fires when job anon memory is within 6 GB of
# the job limit; 27653's tasks showed MaxRSS ~85 GB each under 560G): every host's roll/<host>.json fresh (< 120 s), ahead >= its configured ahead
# - 10, job_mem_headroom_GB > 6, no LOWMEM / NOSPACE / STEP_DIED markers, SUPER.json fresh. Waits up to 5 min for the first reports.
for i in $(seq 1 30); do
  python3 - "$NODEF" "$K" <<'EOF' > $K/state/roll_check.txt; RRC=$?
import json, os, sys, time
hosts = open(sys.argv[1]).read().strip().split(","); K = sys.argv[2]; R = os.environ.get("ROLL_DIR", "/data/home/amartyavarma/ppm/pose5/npl5/roll"); bad = []; now = time.time()
sp = f"{R}/SUPER.json"
if not os.path.exists(sp) or now - os.path.getmtime(sp) > 120: bad.append("SUPER.json missing/stale")
if os.path.exists(f"{R}/STEP_DIED"): bad.append("STEP_DIED marker")
for h in hosts:
    f = f"{R}/{h}.json"
    for mk in ("LOWMEM", "NOSPACE", "STARVED", "CRIT"):
        if os.path.exists(f"{R}/{h}.{mk}"): bad.append(f"{h} {mk}")
    if not os.path.exists(f) or now - os.path.getmtime(f) > 120: bad.append(f"{h} status missing/stale"); continue
    j = json.load(open(f))
    try: want = int(open(f"{K}/state/AHEAD-{h}").read().strip())
    except Exception: want = 100
    a, hr = j.get("ahead_steps"), j.get("job_mem_headroom_GB")
    if a is None or a < want - 10: bad.append(f"{h} ahead {a} < {want} - 10")
    if hr is None or hr <= 6: bad.append(f"{h} job mem headroom {hr} GB <= 6")
print("roll check: " + ("OK on all 16 hosts" if not bad else "; ".join(bad))); sys.exit(1 if bad else 0)
EOF
  [ $RRC = 0 ] && break; sleep 10
done
say "   $(cat $K/state/roll_check.txt)"
case $XRC in 0) ;; 10) echo "- $(pt): [pose5 WARN] pose run $J start eval within rel ${EXACT_WARN_REL:-1e-6} of ppmdecay-j27810, not bit-exact: $(tail -1 $K/state/exact_check.txt | cut -c1-400)" | tee -a $MD >> /data/home/amartyavarma/setup-logs/watchdog/ALERTS.log;;
  *) VRC=1;; esac                                        # 1 = NOT VERIFIED (differs > rel 1e-6, key mismatch, or no record): ALERT below, no LIVE
[ $RRC = 0 ] || { VRC=1; echo "- $(pt): [pose5 ALERT] pose run $J: npl5 fetcher check FAILED: $(cat $K/state/roll_check.txt | cut -c1-300)" | tee -a $MD >> /data/home/amartyavarma/setup-logs/watchdog/ALERTS.log; }
squeue -h -u amartyavarma -n hfb-throttle -t R | grep -q . && say "   hfb-throttle RUNNING (HF uploads controlled)" \
  || { VRC=1; echo "- $(pt): [pose5 ALERT] pose run $J: hfb-throttle NOT RUNNING (HF uploads stall: workers stay 0): sbatch $K/hfb_throttle.sbatch" | tee -a $MD >> /data/home/amartyavarma/setup-logs/watchdog/ALERTS.log; }
if [ $VRC != 0 ]; then echo "- $(pt): [pose5 ALERT] pose run $J is up but NOT VERIFIED (see $LOG): main decides" | tee -a $MD >> /data/home/amartyavarma/setup-logs/watchdog/ALERTS.log
  die "run $J is training but the exactness checks FAILED (above): NOT 'POSE RUN LIVE'; the run is left running for main's decision"; fi
XW=""; [ $XRC = 10 ] && XW=" [WARN start eval within rel ${EXACT_WARN_REL:-1e-6}, not bit-exact: see $K/state/exact_check.txt]"
say "POSE RUN LIVE $J (s2p $VN, g0cond on, from dcp-17850, max $M)$XW"; echo "- $(pt): [pose5] POSE RUN LIVE $J (s2p $VN, g0cond on)$XW" >> $MD
