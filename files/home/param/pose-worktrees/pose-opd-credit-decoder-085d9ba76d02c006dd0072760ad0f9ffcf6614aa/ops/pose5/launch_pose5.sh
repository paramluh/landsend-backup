#!/bin/bash
# pose5 LAUNCH KIT (prep 10-27; NOT run): resume the MAIN 8,1,1 run from dcp-17850 with npl5 (delivery 5) appended + s2p + g0cond, on MoT 30289's 16 hosts.
#   bash /data/home/amartyavarma/ppm/pose5/launch_pose5.sh --dry-run            # prints every command + read-only checks (default)
#   bash /data/home/amartyavarma/ppm/pose5/launch_pose5.sh --go --stop-mot      # the real thing, ONLY on the user's go via main
# Steps (each idempotent; a re-run skips what is done; every abort leaves things as they were):
#  0  preconditions (read only): env / code / B's dry run / dcp-17850 HF verify / no ppm811 job / control markers / resolved-cfg diff on a spare
#  a  stop MoT 30289 WITH a save (mot/control/STOP_AFTER_SAVE + mot/watchdog/WATCHDOG_OFF, = mot/guard/hostswap/swap_exec.sh stop; waits for
#     "STOPPED at step N" + the job gone + dcp-N COMPLETE on all 16; never scancel). Needs --stop-mot AND main's coordination with motguard.
#  a2 DISK (main's rule 10-27, launch day only): live free vs need per host (npl5 share + dcp-17850 where missing + slot-9 pose store + 122 gate);
#     prints the per-host HF-verified candidates (c/DISK_NEED.json); with --free-disk, after MoT's stop save is HF-verified 16/16
#     (mot/guard/state_backup/HF_STATE-dcp-N.json), deletes tier-1 candidates only (kit/free_disk_host.py: re-checks each item's PASS record,
#     never dcp-17850, never PROTECTED, logs pose5/kit/logs/DELETED-<date>.log) until the gap closes; re-measures; aborts with the remaining gap.
#  b  workstream B's npl5 launch-day job (npl5/bin/npl5_launchday.sbatch: npl5 share + plans, slot-9 plan-1/npl2/npl3 dirs + npl3 rows)
#  c  pose5-stage job (afterok:b; kit/pose5_stage.sbatch -> kit/stage_host.sh): gate, dcp-17850 (kit/stage_dcp17850.py), slot-9 pose store
#     (slot9/stage_slot9_extras.sh), trainer-side checks
#  d  ppm811 (code-811-s2p/ppm811.sbatch, env pose5/prod_811_fresh_append5_s2p.env, PPM_RESUME_FROM=dcp-17850 pinned, -w the 16 hosts, --mem)
#     queued --dependency=afterok:<stage job>
#  e  watchdog (kit/ppm811s2p_watchdog.sbatch, spare), eval logger (kit/eval_log_pose5.sbatch, spare), HF watcher check (hfb-watch adopts ppm811),
#     live page refresh (setup-logs/klweb/ppm811_refresh.sbatch on a spare), records (LAUNCHED5.txt, CURRENT_JOB_811, overnight_monitor_811f.md)
echo "SUPERSEDED (audit 1 #15): launch_pose5.sh is the old pre-10-27 kit (direct MoT stop, slinky-18 spare, resume inside CKPT, --free-disk). Use:"
echo "  bash /data/home/amartyavarma/ppm/pose5/switch_to_pose.sh --armed | --dry-run | --go | --after-pause"
exit 2
set -u
P5=/data/home/amartyavarma/ppm/pose5; K=$P5/kit; A=$P5/code-811-s2p
ENVF=$P5/prod_811_fresh_append5_s2p.env; NODEF=$P5/nodes16_s7_s15.txt
CTL=/data/home/amartyavarma/ppm/control/ppm-v1; MD=/data/home/amartyavarma/ppm/control/overnight_monitor_811f.md
M_DIR=/data/home/amartyavarma/ppm/mot; MOTJOB=${MOTJOB:-$(cat $M_DIR/JOB 2>/dev/null)}
RS=17850; RES=/scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-$RS
SPARE=${SPARE:-slinky-18}                              # watchdog / eval logger / page refresh / cfg diff (never a training host, never slinky-9)
B_DRY=${B_DRY:-$P5/npl5/dryrun}                        # workstream B's append_dryrun output for npl5 (RESULT.json + GUARD.json); TBD
A_READY=${A_READY:-$P5/code-811-s2p/S2P_READY}         # workstream A's "proofs passed" marker; TBD
B_JOB=${B_JOB:-$P5/npl5/bin/npl5_launchday.sbatch}     # workstream B's launch-day npl5 copy (16 hosts, CPU)
MOT_HF_WAIT_MIN=${MOT_HF_WAIT_MIN:-60}                 # --free-disk waits this long for MoT's stop save to be HF-verified 16/16
DRY=1; GO=0; STOPMOT=0; NOCFG=0; FREEDISK=0
for x in "$@"; do case $x in
  --dry-run) DRY=1;; --go) GO=1; DRY=0;; --stop-mot) STOPMOT=1;; --no-cfgdiff) NOCFG=1;; --free-disk) FREEDISK=1;;
  *) echo "unknown arg $x"; exit 1;; esac; done
LOG=$K/logs/launch-$(date +%m%d-%H%M%S)$([ $DRY = 1 ] && echo -dry).log; mkdir -p $K/logs $K/state
exec > >(tee -a "$LOG") 2>&1
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
say() { echo "[$(pt)] $*"; }
die() { say "ABORT: $*"; exit 1; }
warn() { say "WARN: $*"; [ $DRY = 1 ] || die "$*"; }        # a WARN in dry-run is an ABORT in --go
UNSET=$(env | grep -oE '^(SLURM|SBATCH|SRUN|SALLOC|PPM|ILP|MOT)_[A-Za-z0-9_]*=' | tr -d = | sed 's/^/-u /' | tr '\n' ' ')
csb() { env $UNSET sbatch "$@"; }
do_() { if [ $DRY = 1 ]; then echo "    [dry] $*"; else eval "$@"; fi; }
say "pose5 launch kit ($([ $DRY = 1 ] && echo DRY-RUN || echo GO)) log $LOG"
HOSTS=$(cat $NODEF); NH=$(echo $HOSTS | tr , '\n' | grep -c .)

# ================================================================ 0. preconditions (read only)
say "0. preconditions"
[ $NH = 16 ] || die "hosts file $NODEF has $NH hosts"
echo ",$HOSTS," | grep -q ',slinky-9,' && die "slinky-9 in $NODEF"
[ "$(tr , '\n' < $NODEF | sort -V | paste -sd,)" = "slinky-0,slinky-1,slinky-2,slinky-3,slinky-5,slinky-6,slinky-7,slinky-8,slinky-11,slinky-12,slinky-13,slinky-14,slinky-15,slinky-16,slinky-17,slinky-19" ] \
  || die "hosts != MoT 30289's 16 hosts"
ev() { grep -E "^$1=" $ENVF | tail -1 | cut -d= -f2-; }
[ "$(ev PPM_HOST_ALIAS)" = "slinky-7:slinky-10,slinky-15:slinky-9" ] || die "env PPM_HOST_ALIAS = $(ev PPM_HOST_ALIAS)"
[ "$(ev PPM_S2P)" = 1 ] || die "env PPM_S2P = $(ev PPM_S2P)"
[ "$(ev PPM_G0COND)" = 1 ] || die "env PPM_G0COND = $(ev PPM_G0COND)"
[ "$(ev PPM_FRESH_POSE_ROOT2)" = "/scratch/amartyavarma/ppm-fresh-pose-npl2,/scratch/amartyavarma/ppm-fresh-pose-npl3,/scratch/amartyavarma/ppm-fresh-pose-npl5" ] \
  || die "env ROOT2 = $(ev PPM_FRESH_POSE_ROOT2)"
[ "$(ev PPM_RESUME_FROM)" = auto ] || die "env PPM_RESUME_FROM must stay auto (the launch pins dcp-$RS; watchdog relaunches use auto)"
M=$(ev PPM_MAX_STEPS); [[ "$M" =~ ^[0-9]+$ ]] && [ "$M" -gt 18022 ] || { warn "env PPM_MAX_STEPS=$M (workstream B's dry run fills it)"; M=19425; say "   (dry: using the estimate $M)"; }
diff <(sed -n 1,308p $ENVF) /data/home/amartyavarma/ppm/code-811/prod_811_fresh_append3.env >/dev/null || die "env lines 1-308 != prod_811_fresh_append3.env (VERBATIM rule)"
grep -q "^C=$A;" $A/ppm811.sbatch || die "$A/ppm811.sbatch does not run $A (C= line)"
grep -q 'PPM_S2P' $A/train_ppm.py || die "$A/train_ppm.py has no PPM_S2P (s2p port missing: the knob would be ignored)"
grep -q 'PPM_G0COND' $A/train_ppm.py || warn "$A/train_ppm.py has no PPM_G0COND (workstream D's g0cond not merged yet: the knob would be ignored)"
grep -q 'g_proj' $A/train_ppm.py $A/model_ppm.py || warn "no g_proj in $A (resume-from-17850 missing-key handling must cover s_emb AND g_proj)"
[ -f "$A_READY" ] || warn "workstream A's READY marker $A_READY missing (s2p proofs)"
python3 - "$P5/c/hf17850/VERIFY_17850.json" <<'EOF' || die "dcp-17850 HF verify record not PASS"
import json, sys; v = json.load(open(sys.argv[1])); assert v["PASS"] and v["total_files_hf"] == 320, v.get("problems"); print(f"   dcp-17850 on HF {v['repo']}: 320 files {v['total_bytes_hf'] / 1e9:.1f} GB PASS ({v['time_utc']} UTC)")
EOF
if [ -f $B_DRY/RESULT.json ] && [ -f $B_DRY/GUARD.json ]; then
  python3 - "$B_DRY" "$M" <<'EOF' || die "workstream B's npl5 dry run does not match (see above)"
import json, sys
R, G = json.load(open(f"{sys.argv[1]}/RESULT.json")), json.load(open(f"{sys.argv[1]}/GUARD.json")); m = int(sys.argv[2]); bad = []
if G.get("guard") != "ok" or G.get("pose_append_at") != [17352, 17641, 17850]: bad.append(f"guard {G.get('guard')} {G.get('pose_append_at')}")
if R.get("resume_ckpt") != [17850]: bad.append(f"resume ckpt {R.get('resume_ckpt')}")
if R.get("max_steps") != m or R.get("ends", [])[:3] != [17352, 17641, 18022]: bad.append(f"max_steps {R.get('max_steps')} vs env {m}; ends {R.get('ends')}")
print("   B dry run:", "; ".join(bad) if bad else f"ok (ends {R['ends']}, max {R['max_steps']})"); sys.exit(1 if bad else 0)
EOF
else warn "workstream B's npl5 dry run ($B_DRY/RESULT.json + GUARD.json) not there yet"; fi
python3 -c "import json,sys; d=json.load(open('$P5/slot9/posestore/SLOT9_POSESTORE_READY.json')); sys.exit(0 if d.get('PASS') and d.get('node') == 'slinky-9' else 1)" 2>/dev/null \
  && say "   slot-9 PPM pose store staged on /data and verified (slot9/posestore/SLOT9_POSESTORE_READY.json)" \
  || warn "slot-9 PPM pose store not ready (slot9/build_posestore.sbatch -> slot9/posestore/SLOT9_POSESTORE_READY.json)"
[ -f $P5/npl5/state/dryrun.ok ] || warn "workstream B's npl5/state/dryrun.ok missing (its launch-day job refuses without it)"
Q=$(squeue -h -u amartyavarma -o '%i %j %T' | awk '$2=="ppm811" || $2 ~ /^ppm811.*watch/ || $2=="ppm811f-evallog" || $2=="ppm811s2p-evallog" || $2=="pose5-stage"')
if [ -n "$Q" ]; then
  if [ -f $K/state/LAUNCHED ] && [ $DRY = 0 ]; then die "already launched ($(cat $K/state/LAUNCHED)); queue: $Q"; fi
  warn "811 jobs already in the queue: $(echo $Q | tr '\n' ' ')"
fi
for f in STOP_AFTER_SAVE SAVE_NOW WATCHDOG811_OFF; do [ -e $CTL/$f ] && warn "$CTL/$f present (would stop / disable the new run)"; done
grep -q 'ppm/pose5/code-811-s2p' /data/home/amartyavarma/setup-logs/hfb/snap_code.sh || warn "hfb snap_code.sh does not know code-811-s2p (code snapshot would be ppm/code)"
squeue -h -u amartyavarma -n hfb-watch -o %i | grep -q . || warn "HF watcher hfb-watch not running (setup-logs/hfb/hfb_watch.sbatch; give -w a spare)"
if [ $NOCFG = 0 ]; then
  say "   resolved-cfg diff vs 27653 on $SPARE (CPU, ~1-3 min)"
  OUT=$K/state/CFG_DIFF_launch.txt
  timeout 900 env $UNSET srun -p all -w $SPARE -N1 -n1 -c 2 --mem=24G -t 15 -J pose5-cfgdiff nice -n 10 bash $P5/c/cfgdiff_pose5.sh $OUT $A \
      PPM_RESUME_FROM=$RES PPM_MAX_STEPS=$M > $K/logs/cfgdiff-launch.log 2>&1 || die "cfg diff run failed ($K/logs/cfgdiff-launch.log)"
  KEYS=$(grep -vE '^#|^ ' $OUT | sort | paste -sd' ')
  BASE="fresh.pose_root2 max_steps resume_from run_tag s2p s2p_from_step"; G0RE=${G0_KEY_RE:-^g0cond}   # g0cond key name(s): workstream D's code
  REST=$(for k in $KEYS; do echo " $BASE " | grep -q " $k " || echo $k; done | paste -sd' ')
  MISS=$(for k in $BASE; do echo " $KEYS " | grep -q " $k " || echo $k; done | paste -sd' ')
  [ -z "$MISS" ] || die "resolved cfg: expected differences missing: $MISS ($OUT)"
  [ -n "$REST" ] || die "resolved cfg has no g0cond key although PPM_G0COND=1 (D's change not in $A?) ($OUT)"
  for k in $REST; do echo "$k" | grep -qE "$G0RE" || die "resolved cfg differs from 27653 in an UNEXPECTED key: $k (all: $KEYS; see $OUT)"; done
  grep -A2 '^s2p$' $OUT | grep -q '8,1,1 : "full"' || die "cfg s2p != full ($OUT)"
  say "   cfg diff OK: only $KEYS ($OUT)"
fi

# ================================================================ a. MoT 30289: stop WITH a save
say "a. MoT ${MOTJOB:-?}"
MS=$(squeue -h -j "${MOTJOB:-0}" -o '%j %T' 2>/dev/null)
ML=$M_DIR/logs/ppm-mot-$MOTJOB.log
if [ -n "$MS" ]; then
  cat <<EOF
   !!! MoT ($MS) is running on the 16 hosts. Stopping it is NOT silent: MAIN must first coordinate with motguard (amartyavarma-3c [949fd3])
   !!! (its tmux guard + cluster watchdog would otherwise act), and confirm how MoT's stop save gets its HF backup (mot-state-bk overlaps into
   !!! a running job; after 30289 exits it needs an allocation: e.g. an --overlap step inside the new ppm811 job). Procedure used (documented in
   !!! mot/guard/hostswap/swap_exec.sh stop): touch $M_DIR/watchdog/WATCHDOG_OFF + $M_DIR/control/STOP_AFTER_SAVE, wait for "STOPPED at step N"
   !!! and the job to leave the queue (save ~20-45 s + exit ~10 s), abort on "STOP_AFTER_SAVE ... SKIPPED". Never scancel.
EOF
  [ $STOPMOT = 1 ] || { [ $DRY = 1 ] && say "   (dry) would refuse here without --stop-mot" || die "MoT running: re-run with --stop-mot once main has coordinated with motguard"; }
  do_ "touch $M_DIR/watchdog/WATCHDOG_OFF $M_DIR/control/STOP_AFTER_SAVE"
  do_ "echo '- \$(TZ=America/Los_Angeles date \"+%a %m-%d %H:%M PT\"): [pose5 launch] WATCHDOG_OFF + STOP_AFTER_SAVE touched for MoT job $MOTJOB (pose5 launch, main-coordinated)' >> $M_DIR/GUARD_LOG.md"
  if [ $DRY = 0 ]; then
    for i in $(seq 1 360); do                                   # <= 30 min
      grep -qE '!!! \[control\] STOP_AFTER_SAVE .*SKIPPED' $ML && die "MoT save SKIPPED (no /scratch space): MoT keeps running; fix space with main"
      grep -qE 'STOPPED at step [0-9]+' $ML && ! squeue -h -j $MOTJOB | grep -q . && break
      sleep 5
    done
    grep -qE 'STOPPED at step [0-9]+' $ML && ! squeue -h -j $MOTJOB | grep -q . || die "MoT did not stop within 30 min (STOP_AFTER_SAVE stays; check with motguard)"
  fi
else
  say "   MoT job ${MOTJOB:-?} not in the queue"
  [ -f "$ML" ] && ! grep -qE 'STOPPED at step [0-9]+' $ML && warn "MoT log $ML has no clean STOPPED line: was it stopped with a save? (ask motguard)"
fi
MN=$(grep -oE 'STOPPED at step [0-9]+' $ML 2>/dev/null | tail -1 | grep -oE '[0-9]+$')
say "   MoT stop save: dcp-${MN:-?} ($([ $DRY = 1 ] && echo 'dry: not stopped' || echo stopped))"
if [ $DRY = 0 ]; then
  N=$(timeout 300 env $UNSET srun -p all -N16 -n16 --ntasks-per-node=1 -w $HOSTS -c1 --mem=1G --gres=none -t 5 -J pose5-motchk \
        bash -c "[ -f /scratch/amartyavarma/ppm-mot/ckpt/ppm-mot/dcp-$MN/COMPLETE ] && echo C" 2>/dev/null | grep -c '^C$')
  [ "$N" = 16 ] || die "MoT dcp-$MN COMPLETE on $N/16 hosts"
  say "   MoT dcp-$MN COMPLETE on 16/16 (kept; protected path ppm-mot)"
fi

# ================================================================ a2. disk (gate per host; HF-verified deletions only with --free-disk)
say "a2. disk: need = npl5 share + dcp-17850 (where missing) + slot-9 pose (slinky-15) + 122 GB gate; candidates c/DISK_NEED.json"
python3 - $P5/c/DISK_NEED.json <<'EOF' || die "c/DISK_NEED.json unreadable"
import json, sys
D = json.load(open(sys.argv[1]))
print(f"   records ({D.get('free_src', '')}):  host   free  need  short | tier-1 HF-verified candidates (GB) | tier-2 protected (user OK) GB")
for h, v in D["hosts"].items():
    t1 = [c for c in v.get("candidates", []) if c.get("tier") == "1_hf_verified"]
    t2 = sum(c.get("bytes", 0) for c in v.get("candidates", []) if c.get("tier") == "2_protected_user_ok") / 1e9
    print(f"   {h:>10} {v['free_GB']:6.0f} {v['need']['total_GB']:5.0f} {v['short_GB']:6.0f} | "
          + ", ".join(f"{c['path'].replace('/scratch/amartyavarma/', '')} {c['bytes'] / 1e9:.0f}" for c in t1)[:150] + f" | {t2:.0f}")
EOF
if [ $DRY = 1 ]; then
  echo "    [dry] measure: srun -N16 -w <hosts> -c1 --mem=2G python3 $K/free_disk_host.py            (prints free / need / gap / WOULD-delete)"
  echo "    [dry] --free-disk: wait for $M_DIR/guard/state_backup/HF_STATE-dcp-<MoT stop step>.json (16 hosts, verified), then the same with --delete"
else
  FD=""
  if [ $FREEDISK = 1 ]; then
    say "   --free-disk: waiting (<= $MOT_HF_WAIT_MIN min) for MoT's stop save dcp-$MN to be HF-verified 16/16 (motguard's state backup)"
    for i in $(seq 1 $((MOT_HF_WAIT_MIN * 4))); do
      python3 -c "import json,sys; d=json.load(open('$M_DIR/guard/state_backup/HF_STATE-dcp-$MN.json')); sys.exit(0 if d.get('hosts') == 16 and d.get('files') == 320 and d.get('verified') else 1)" 2>/dev/null && break
      sleep 15
    done
    python3 -c "import json,sys; d=json.load(open('$M_DIR/guard/state_backup/HF_STATE-dcp-$MN.json')); sys.exit(0 if d.get('hosts') == 16 and d.get('files') == 320 and d.get('verified') else 1)" 2>/dev/null \
      || die "MoT stop save dcp-$MN is not HF-verified 16/16 ($M_DIR/guard/state_backup/HF_STATE-dcp-$MN.json): NOTHING deleted; motguard must back it up first"
    say "   MoT dcp-$MN HF-verified 16/16 -> HF-verified tier-1 deletions allowed"; FD=--delete
  fi
  OUT=$(timeout 3600 env $UNSET srun -p all -N16 -n16 --ntasks-per-node=1 -w $HOSTS -c1 --mem=2G --gres=none -t 60 -J pose5-disk \
        /data/home/amartyavarma/venvs/duplex/bin/python -I $K/free_disk_host.py $FD 2>&1); rc=$?
  echo "$OUT" | sed 's/^/   /'
  SHORT=$(echo "$OUT" | grep -E '^DISK \S+ (after|would be after).*STILL SHORT' | sed -E 's/^DISK (\S+) .*STILL SHORT ([0-9.]+) GB/\1:\2/' | paste -sd' ')
  [ "$(echo "$OUT" | grep -cE '^DISK \S+ (after|would be after)')" = 16 ] || die "disk step did not report from all 16 hosts (rc $rc)"
  if [ -n "$SHORT" ]; then
    [ $FREEDISK = 1 ] && die "still short after the HF-verified deletions (GB): $SHORT; remaining options need the user's path-specific OK (tier 2 in c/DISK_NEED.json)"
    die "short (GB): $SHORT -> re-run with --free-disk (main's rule; deletes HF-verified tier-1 items only) or free space with main"
  fi
  say "   disk OK on 16/16 hosts"
fi

# ================================================================ b. workstream B's npl5 launch-day job (16 hosts, CPU)
say "b. npl5 launch-day copy ($B_JOB)"
if [ $DRY = 1 ]; then echo "    [dry] NJ=\$(sbatch --parsable -w $NODEF $B_JOB)"
else
  [ -f $P5/npl5/state/launchday.ok ] && NJ="" && say "   npl5 already staged ($(cat $P5/npl5/state/launchday.ok))"
  if [ ! -f $P5/npl5/state/launchday.ok ]; then
    NJ=$(csb --parsable -w $NODEF $B_JOB) || die "npl5 sbatch failed"; [[ "$NJ" =~ ^[0-9]+$ ]] || die "npl5 sbatch: $NJ"; say "   npl5-copy job $NJ"
  fi
fi

# ================================================================ c. stage job on the 16 hosts
say "c. pose5-stage on the 16 hosts (afterok npl5-copy; gate; dcp-17850; slot-9 pose store; trainer-side checks)"
if [ $DRY = 1 ]; then
  echo "    [dry] SJ=\$(sbatch --parsable \${NJ:+--dependency=afterok:\$NJ} -N 16 -w $NODEF $K/pose5_stage.sbatch)"
  say "   (dry) per-host dry run is possible only once the hosts are free: sbatch -N 16 -w $NODEF $K/pose5_stage.sbatch --dry-run"
else
  SJ=$(csb --parsable ${NJ:+--dependency=afterok:$NJ} -N 16 -w $NODEF $K/pose5_stage.sbatch) || die "stage sbatch failed"
  [[ "$SJ" =~ ^[0-9]+$ ]] || die "stage sbatch: $SJ"
  say "   stage job $SJ"
fi

# ================================================================ d. ppm811 (queued behind the stage job)
say "d. ppm811 resume from dcp-$RS (max_steps $M)"
LCMD="PPM_ENV=$ENVF PPM_RESUME_FROM=$RES sbatch --parsable --dependency=afterok:\${SJ} -N 16 -w $NODEF --mem=560G $A/ppm811.sbatch"
if [ $DRY = 1 ]; then echo "    [dry] J=\$($LCMD)"
else
  mkdir $K/state/LAUNCH.lock 2>/dev/null || die "LAUNCH.lock exists ($K/state/LAUNCH.lock): launched before?"
  J=$(env $UNSET PPM_ENV=$ENVF PPM_RESUME_FROM=$RES sbatch --parsable --dependency=afterok:$SJ -N 16 -w $NODEF --mem=560G $A/ppm811.sbatch) || die "ppm811 sbatch failed"
  [[ "$J" =~ ^[0-9]+$ ]] || die "ppm811 sbatch: $J"
  echo $J > $CTL/CURRENT_JOB_811; echo "$J $SJ $(pt)" > $K/state/LAUNCHED
  say "   ppm811 $J (afterok:$SJ)"
fi

# ================================================================ e. companions
say "e. companions"
do_ "WD=\$(ENVF=$ENVF NODEF=$NODEF TARGET=$M MIN_STEP=$RS csb --parsable -w $SPARE $K/ppm811s2p_watchdog.sbatch)"
do_ "EL=\$(MIN_JOB=\$J END=$M MIN_STEP=$RS csb --parsable -w $SPARE $K/eval_log_pose5.sbatch)"
if squeue -h -u amartyavarma -n klweb-ppm811-refresh -o %i | grep -q .; then say "   live page refresh already running"
else do_ "csb -w $SPARE /data/home/amartyavarma/setup-logs/klweb/ppm811_refresh.sbatch"; fi
say "   HF watcher: hfb-watch adopts job name ppm811 (ppm_job.sh); codesnap from code-811-s2p (snap_code.sh); saves after 17850 upload as plain"
say "   dcp-<step>/ (no ppm-v1 record past 17850; the decay branch used ckpt/ppmdecay + its own repos); staged dcp-17850 keeps its original mtime"
say "   -> lineage check = the recorded save (no re-upload)."
if [ $DRY = 0 ]; then
  echo "$J $WD $EL $M $SJ" > $P5/LAUNCHED5.txt
  echo "- $(TZ=America/Los_Angeles date '+%a %m-%d %H:%M PT'): [pose5] main 8,1,1 RESUMED from dcp-$RS + npl5 + s2p: train $J (after stage $SJ), watchdog $WD, eval log $EL, max $M" >> $MD
  say "LAUNCHED: train $J (afterok stage $SJ), watchdog $WD, eval log $EL -> $P5/LAUNCHED5.txt"
  for X in $NJ $SJ; do
    say "   waiting for job $X ($([ "$X" = "${NJ:-}" ] && echo 'npl5-copy, ~10-20 min' || echo 'pose5-stage, ~3-10 min')) ..."
    for i in $(seq 1 600); do                                   # <= 2.5 h
      st=$(sacct -n -X -j $X -o State%20 2>/dev/null | head -1 | tr -d ' ')
      case $st in COMPLETED) say "   job $X COMPLETED"; break;;
        FAILED|CANCELLED*|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY)
          [ "$X" = "$SJ" ] && tail -40 $K/logs/stage-$SJ.log
          scancel $SJ $J $WD $EL 2>/dev/null                     # pending jobs that never started (afterok not met) + companions
          rmdir "${K:?}/state/LAUNCH.lock" 2>/dev/null; mv $K/state/LAUNCHED $K/state/LAUNCHED.aborted-$(date +%s) 2>/dev/null
          die "job $X $st: ppm811 $J (pending, never started) + companions cancelled; nodes free; fix + re-run";;
      esac; sleep 15; done
  done
  say "   staging done: ppm811 $J starts now"
fi
cat <<EOF
   Check in ppm/logs/ppm-<J>.log: "== host alias ... slinky-15:slinky-9 -> rank order ...,slinky-8,slinky-15,slinky-7,slinky-11,..";
   "[resume] FRESH POSE APPEND ... appended plans added at step [17352, 17641, 17850]"; "[fresh] POSE APPEND ... ends after each plan
   [17352, 17641, 18022, $M] ... digests == the checkpoint's (prefix)"; the s2p + g0cond resume lines (s_emb + g_proj added at 17850, zero-init, fresh AdamW); "resumed $RES at
   step $RS"; step 17851 ~68-70 s; start eval vs 27653's last eval (fresh/EVALS_append3.md) — pose evals equal at step 0 (s_emb = 0, g_proj = 0).
EOF
say "done ($([ $DRY = 1 ] && echo 'dry run: nothing submitted, nothing touched' || echo launched))"
