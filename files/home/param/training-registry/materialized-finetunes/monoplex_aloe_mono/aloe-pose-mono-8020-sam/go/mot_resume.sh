#!/bin/bash
# monosam go/mot_resume.sh (10-26): resume the MoT 14B (ppm-mot) after the planned pause, from the save recorded by go/mot_stop.sh, on the
# SAME 16 hosts with the SAME env / sbatch / code / overrides as the stopped job (= the submission motguard's swap_exec.sh and the MoT
# watchdog use; 30289 was launched this way). STAGED ONLY: run when main relays the go (after the monoplex finished + export verified).
#
#   mot_resume.sh --dry-run     validate everything read-only and print the exact command. Before the stop has happened it validates
#                               against the live job and its newest save (and says which checks only apply after the stop).
#   mot_resume.sh               submit + wait RUNNING + verify the first log lines; then remove the markers mot_stop.sh created.
#   options: --queue-behind (submit even if a host still has allocations; the job PENDs until all 16 are free)
#            --hosts-wait-min=10 (wait for the 16 hosts to be fully free)  --wait-run-min=30  --verify-min=20
#
# How MoT resumes (MOT_NOTES.md): mot.env PPM_RESUME_FROM=auto -> train_ilp.py:1160-1163 SF.newest_common = the newest step COMPLETE on
# EVERY node under /scratch/amartyavarma/ppm-mot/ckpt/ppm-mot (MIN over nodes, asserted present on all) -> load(): per-rank digests ==
# saved (asserted), world 128 + rank->host map == saved (asserted; needs the same host file + PPM_HOST_ALIAS), RESUME_KEYS cfg == saved.
# MOT_INIT is ignored on a resume. Exit: 0 resumed + verified | 1 refused (nothing submitted) | 4 not RUNNING in time (left PENDING,
# not cancelled) | 5 submitted but a verification failed (WATCHDOG_OFF kept; motguard to assess).
#   test: --verify-only=NEWJOB:STOPJOB:STEP  run ONLY the log verification of an existing resume (read-only; e.g. 30289:29514:5829)
set -uo pipefail
. /data/home/amartyavarma/ppm/aloe-pose-mono-8020-sam/go/mot_common.sh
DRY=0; QB=0; HW=10; WRUN=30; VMIN=20; VONLY=""
for a in "$@"; do case "$a" in
  --dry-run) DRY=1;; --queue-behind) QB=1;; --hosts-wait-min=*) HW=${a#*=};; --wait-run-min=*) WRUN=${a#*=};; --verify-min=*) VMIN=${a#*=};; --verify-only=*) VONLY=${a#*=};;
  *) echo "usage: mot_resume.sh [--dry-run] [--queue-behind] [--hosts-wait-min=10] [--wait-run-min=30] [--verify-min=20] [--verify-only=NEW:STOP:STEP]"; exit 2;; esac; done

# ------------------------------------------------------------------ the log verification of a resumed job (phase 6; also --verify-only)
verify_loop() {
  t0=$(date +%s); declare -A DONE=()
  chk() { # key, description, result(0/1)
    [ -n "${DONE[$1]:-}" ] && return; if [ "$3" = 0 ]; then DONE[$1]=ok; ok "$2"; fi; }
  while :; do
    if grep -aqE '^ABORT|COMPILE FAIL|Traceback|NCCL error|CUDA error|OutOfMemory|srun: error|resume digest mismatch|AssertionError' "$L" 2>/dev/null; then
      fail "error in $L:"; grep -aE '^ABORT|COMPILE FAIL|Traceback|NCCL error|CUDA error|OutOfMemory|srun: error|digest mismatch|AssertionError' "$L" | head -5 | sed 's/^/        /'; break; fi
    [ -f "$L" ] && { a=$(sub_ovr "$L"); [ -n "$a" ] && { [ "$a" = "$SUB_OVR" ] && chk ovr "submission overrides == job $STOP_JOB's" 0 || { fail "submission overrides '$a' != '$SUB_OVR'"; break; }; }; }
    [ -f "$L" ] && { a=$(rank_order "$L"); [ -n "$a" ] && { [ "$a" = "$RANK_ORDER" ] && chk rank "rank order == job $STOP_JOB's ($a)" 0 || { fail "rank order '$a' != '$RANK_ORDER'"; break; }; }; }
    r=$(grep -a -m1 -oE "\[resume\] auto: newest step COMPLETE on every node under [^ ]+: [0-9a-z ()-]+" "$L" 2>/dev/null)
    if [ -n "$r" ]; then [ "$(grep -oE '[0-9]+$' <<<"$r")" = "$S" ] && chk auto "$r" 0 || { fail "auto picked '${r##*: }', not $S"; break; }; fi
    grep -aqE "\[resume\] $CKPT/dcp-$S: step $S, .*per-rank digests == saved on every rank" "$L" 2>/dev/null && chk dig "[resume] dcp-$S: weights + 3 optimizers + sched restored, per-rank digests == saved on every rank" 0
    p=$(grep -a -m1 -E "\[plan\] steps [0-9]+\.\.[0-9]+:" "$L" 2>/dev/null)
    if [ -n "$p" ]; then grep -qE "\[plan\] steps $S\.\.$MAXSTEP:" <<<"$p" && chk plan "plan starts at $S: $(grep -oE 'steps .*' <<<"$p" | cut -c1-160)" 0 || { fail "plan line does not start at $S: $p"; break; }; fi
    if [ -z "${DONE[cfg]:-}" ] && grep -aqE '\] ILP ppm-mot-j[0-9]+: .* cfg \{' "$L" 2>/dev/null; then
      c=$(cfg_sha "$L")
      if [ "$c" = "$CFG_SHA" ]; then chk cfg "resolved trainer cfg == job $STOP_JOB's except run_tag ($c)" 0
      elif ! cfg_clean "$L" || ! cfg_clean "$L0"; then DONE[cfg]=warn; warn "cfg line of the new or the stopped job interleaved with another rank's stderr (as in 29514): text compare skipped; the trainer's own RESUME_KEYS assert (load(), train_ilp.py:856-857; world + host map :843) + the recipe sha cover it"
      else fail "resolved cfg differs from job $STOP_JOB's beyond run_tag ($c vs $CFG_SHA):"; diff <(cfg_keys "$L0") <(cfg_keys "$L") | head -20 | sed 's/^/        /'; break; fi
    fi
    e=$(evals "$L" | awk -v s="$S" '$1==s{print $2}' | head -1)
    if [ -n "$e" ] && [ -z "${DONE[eval]:-}" ]; then
      lastpre=$(tr ';' '\n' <<<"$PRE_EVALS" | tail -1 | awk '{print $2}')
      awk -v a="$e" -v b="${lastpre:-$e}" 'BEGIN{d=a-b; if(d<0)d=-d; exit !(d<=0.03)}' && chk eval "EVAL step $S word_ce_norm $e vs pre-stop ($PRE_EVALS)" 0 || { DONE[eval]=warn; warn "EVAL step $S word_ce_norm $e is > 0.03 from the last pre-stop EVAL ($PRE_EVALS)"; }
    fi
    ns=$(grep -aE "\] step [0-9]+/$MAXSTEP [0-9.]+s " "$L" 2>/dev/null | wc -l)
    if [ "$ns" -ge 4 ] && [ -z "${DONE[steps]:-}" ]; then
      grep -aE "\] step [0-9]+/$MAXSTEP [0-9.]+s " "$L" | head -4 | cut -c1-175 | sed 's/^/        /'
      f1=$(grep -aE "\] step [0-9]+/$MAXSTEP " "$L" | head -1 | sed -nE 's/.*\] step ([0-9]+)\/.*/\1/p')
      [ "$f1" = "$((S + 1))" ] && chk first "first step line = $((S + 1))" 0 || fail "first step line is $f1, expected $((S + 1))"
      read -r pm psd _ <<<"$PRE_WCE"; nm=$(grep -aE "\] step [0-9]+/$MAXSTEP " "$L" | head -4 | sed -nE 's/.*\| word_ce_norm=([0-9.]+) .*/\1/p' | awk '{s+=$1;n++} END{printf "%.4f", s/n}')
      awk -v a="$nm" -v m="$pm" -v s="$psd" 'BEGIN{d=a-m; if(d<0)d=-d; exit !(d <= 3*s/2 + 0.02)}' && chk steps "first 4 steps word_ce_norm mean $nm vs pre-stop $pm (sd $psd): continuous" 0 \
        || { DONE[steps]=warn; warn "first 4 steps word_ce_norm mean $nm vs pre-stop $pm sd $psd: check by eye"; }
      sp=$(grep -aE "\] step [0-9]+/$MAXSTEP [0-9.]+s " "$L" | sed -n '2,4p' | sed -nE 's/.*\] step [0-9]+\/[0-9]+ ([0-9.]+)s .*/\1/p' | awk '{s+=$1;n++} END{printf "%.1f", s/n}')
      awk -v s="$sp" 'BEGIN{exit !(s<=25)}' && chk sps "s/step steps 2-4 avg $sp (<= 25; first step ~27 s warm-up)" 0 || { DONE[sps]=warn; warn "s/step $sp > 25 after warm-up"; }
    fi
    [ -n "${DONE[ovr]:-}" ] && [ -n "${DONE[rank]:-}" ] && [ -n "${DONE[auto]:-}" ] && [ -n "${DONE[dig]:-}" ] && [ -n "${DONE[plan]:-}" ] && [ -n "${DONE[cfg]:-}" ] \
      && [ -n "${DONE[eval]:-}" ] && [ -n "${DONE[steps]:-}" ] && break
    if [ $(( $(date +%s) - t0 )) -ge $((VMIN * 60)) ]; then fail "verification incomplete after ${VMIN} min: have [${!DONE[*]}]"; tail -5 "$L" | cut -c1-200 | sed 's/^/        /'; break; fi
    [ -n "$VONLY" ] || [ "$(squeue -h -j "$JID" -o %T)" = RUNNING ] || { fail "job $JID no longer RUNNING during verification"; break; }
    sleep 10
  done
}
if [ -n "$VONLY" ]; then
  IFS=: read -r JID STOP_JOB S <<<"$VONLY"; L=$M/logs/ppm-mot-$JID.log; L0=$M/logs/ppm-mot-$STOP_JOB.log; VMIN=0
  CFG_SHA=$(cfg_sha "$L0"); SUB_OVR=$(sub_ovr "$L0"); RANK_ORDER=$(rank_order "$L0"); PRE_EVALS=$(evals "$L0" | tail -3 | paste -sd';')
  read -r a b c <<<"$(wce_stats "$L0" 50)"; PRE_WCE="$a $b $c"
  echo "== mot_resume.sh --verify-only: resumed job $JID vs stopped job $STOP_JOB at step $S (read-only)"; phase "6 verify (test)"; verify_loop
  echo; [ ${#FAIL[@]} -eq 0 ] && echo "== verify-only: PASS${WARNS:+ with ${#WARNS[@]} warning(s)}" || { echo "== verify-only: ${#FAIL[@]} FAIL"; printf '   - %s\n' "${FAIL[@]}"; }; exit 0
fi
echo "== mot_resume.sh $(pt)  dry-run=$DRY"
if [ "$DRY" = 0 ]; then exec 9>"$GO/.mot_resume.lock"; flock -n 9 || { echo "REFUSED: another mot_resume.sh is running"; exit 1; }; fi

# ------------------------------------------------------------------ PHASE 0: what to resume
phase "0 the recorded stop"
PRESTOP=0
if [ -f "$STEPF" ] && [ -f "$STATE" ]; then
  # shellcheck disable=SC1090
  . "$STATE"; S=$(tr -dc 0-9 < "$STEPF")
  [ "$S" = "${STOP_STEP:-}" ] && ok "recorded stop: job $STOP_JOB step $S (go/MOT_STOP_STEP = MOT_STOP_STATE)" || fail "MOT_STOP_STEP '$S' != MOT_STOP_STATE STOP_STEP '${STOP_STEP:-}'"
else
  [ "$DRY" = 1 ] || { echo "REFUSED: no recorded stop ($STEPF / $STATE): run go/mot_stop.sh first"; exit 1; }
  PRESTOP=1; STOP_JOB=$(tr -dc 0-9 < "$M/JOB"); NODEF=$(grep -E '^NODEF=' "$LREC" | tail -1 | cut -d= -f2-)
  HOSTS=$(tr -d ' \n' < "$NODEF"); MOT_INIT=$(grep -E '^MOT_INIT=' "$LREC" | tail -1 | cut -d= -f2-)
  ILP_MIN_SCRATCH_GB=$(grep -E '^ILP_MIN_SCRATCH_GB=' "$LREC" | tail -1 | cut -d= -f2-)
  L0=$M/logs/ppm-mot-$STOP_JOB.log; RECIPE_SHA=$(recipe_sha "$NODEF"); CFG_SHA=$(cfg_sha "$L0"); SUB_OVR=$(sub_ovr "$L0"); RANK_ORDER=$(rank_order "$L0")
  read -r a b c <<<"$(wce_stats "$L0" 50)"; PRE_WCE="$a $b $c"; PRE_EVALS=$(evals "$L0" | tail -3 | paste -sd';'); WD_OFF_PRE=0; PAUSED_PRE=0; SBOFF_PRE=0
  S=$(grep -aE '\[save\] /scratch/[^ ]+/dcp-[0-9]+ node-local' "$L0" | tail -1 | sed -nE 's/.*\/dcp-([0-9]+) node-local.*/\1/p')
  say "[after stop] dry-run stands in the live job's newest save dcp-$S for the recorded stop step"
  say "PRE-STOP DRY-RUN: no recorded stop yet (MoT job $STOP_JOB still running). Validating against the live job; checks marked [after stop] only pass once go/mot_stop.sh has run."
fi
L0=$M/logs/ppm-mot-$STOP_JOB.log
NH=$(tr , '\n' <<<"$HOSTS" | grep -c .)

# ------------------------------------------------------------------ PHASE 1: same recipe
phase "1 same env / sbatch / code / hosts / overrides as job $STOP_JOB"
L_INIT=$(grep -E '^MOT_INIT=' "$LREC" | tail -1 | cut -d= -f2-); L_MIN=$(grep -E '^ILP_MIN_SCRATCH_GB=' "$LREC" | tail -1 | cut -d= -f2-)
L_NODEF=$(grep -E '^NODEF=' "$LREC" | tail -1 | cut -d= -f2-)
[ "$L_INIT" = "$MOT_INIT" ] && [ "$L_MIN" = "$ILP_MIN_SCRATCH_GB" ] && [ "$L_NODEF" = "$NODEF" ] && ok "LAST_LAUNCH.env: MOT_INIT=$L_INIT ILP_MIN_SCRATCH_GB=$L_MIN NODEF=$L_NODEF (= at the stop)" \
  || fail "LAST_LAUNCH.env changed since the stop: $L_INIT/$L_MIN/$L_NODEF vs $MOT_INIT/$ILP_MIN_SCRATCH_GB/$NODEF"
[ "$(tr -d ' \n' < "$NODEF")" = "$HOSTS" ] && [ "$NH" -eq 16 ] && ok "host file $NODEF: $HOSTS" || fail "host file differs from the stop's / not 16 hosts"
grep -qw slinky-9 <<<"${HOSTS//,/ }" && fail "slinky-9 in the host list"
RS=$(recipe_sha "$NODEF"); [ "$RS" = "$RECIPE_SHA" ] && ok "recipe sha $RS (mot.env + mot.sbatch + host file + code/**) == at the stop" || fail "recipe sha $RS != at the stop $RECIPE_SHA"
JS=$(sacct -n -X -j "$STOP_JOB" -o Start -P 2>/dev/null | head -1); JSE=$(date -d "$JS" +%s 2>/dev/null || echo 0); RM=$(recipe_newest_mtime "$NODEF")
[ "$RM" -lt "$JSE" ] && ok "no recipe file modified since job $STOP_JOB started ($JS UTC; newest recipe mtime $(date -d @"$RM" '+%F %T') UTC)" || fail "a recipe file was modified after job $STOP_JOB started"
grep -qx 'PPM_RESUME_FROM=auto' "$ENVF" && ok "mot.env PPM_RESUME_FROM=auto" || fail "mot.env has no PPM_RESUME_FROM=auto line"
ALIAS=$(grep -E '^PPM_HOST_ALIAS=' "$ENVF" | tail -1 | cut -d= -f2-); say "mot.env PPM_HOST_ALIAS=$ALIAS (part of the recipe sha; rank order at the stop: $RANK_ORDER)"
[ "$(grep -E '^PPM_SAVE_ROOT=' "$ENVF" | tail -1 | cut -d= -f2-)" = /scratch/amartyavarma/ppm-mot/ckpt ] && ok "mot.env PPM_SAVE_ROOT=/scratch/amartyavarma/ppm-mot/ckpt" || fail "PPM_SAVE_ROOT changed"
OVR="PPM_ENV=$ENVF MOT_INIT=$L_INIT PPM_BACKBONE=qwen3-14b PPM_LAYERS=40 ILP_MIN_SCRATCH_GB=$L_MIN PPM_RESUME_FROM=auto"
EXP_SUB=$(tr ' ' '\n' <<<"${OVR#PPM_ENV=$ENVF }" | sort | paste -sd' ')
[ "$EXP_SUB" = "$SUB_OVR" ] && ok "submission overrides == job $STOP_JOB's logged ones: $SUB_OVR" || fail "overrides '$EXP_SUB' != job $STOP_JOB's '$SUB_OVR'"
CTL=$(grep -E '^PPM_CONTROL_DIR=' "$ENVF" | tail -1 | cut -d= -f2-)
for x in STOP_AFTER_SAVE SAVE_NOW; do [ -e "$CTL/$x" ] && fail "$CTL/$x present: the resumed job would act on it at its first step boundary" || ok "no pending $CTL/$x"; done
[ -f "$SB" ] && grep -q -- '--mem=560G' "$SB" && ok "mot.sbatch header --mem=560G (passed explicitly too: same value)" || fail "mot.sbatch --mem header changed"

# ------------------------------------------------------------------ PHASE 2: queue + hosts
phase "2 queue and hosts"
NMOT=$(squeue -h -u "$(id -un)" -n ppm-mot -t R,PD -o '%i:%T' | paste -sd' ')
if [ -n "$NMOT" ]; then
  [ $PRESTOP = 1 ] && say "[after stop] ppm-mot in the queue now: $NMOT (expected before the stop; a real resume refuses while any ppm-mot job exists)" || fail "ppm-mot job(s) in the queue: $NMOT"
else ok "no ppm-mot job in the queue"; fi
if [ $PRESTOP = 0 ]; then
  grep -aqE "STOPPED at step $S \(clean exit\)" "$L0" && ok "job $STOP_JOB log: STOPPED at step $S" || fail "job $STOP_JOB log has no 'STOPPED at step $S'"
fi
hosts_free() {   # prints busy hosts (CPUs / GPUs allocated or not schedulable); empty = all 16 free
  local h l st ca at
  for h in ${HOSTS//,/ }; do
    l=$(scontrol show node "$h" 2>/dev/null); st=$(grep -oE 'State=[^ ]+' <<<"$l" | cut -d= -f2); ca=$(grep -oE 'CPUAlloc=[0-9]+' <<<"$l" | cut -d= -f2)
    at=$(grep -oE 'AllocTRES=[^ ]*' <<<"$l" | cut -d= -f2-)
    if [[ "$st" =~ DRAIN|DOWN|FAIL|NOT_RESPONDING|MAINT|RESERVED ]] || [ "${ca:-1}" -gt 0 ]; then echo "$h state=$st cpu_alloc=${ca:-?} alloc=[${at}]"; fi
  done
}
BUSY=$(hosts_free)
if [ -z "$BUSY" ]; then ok "all 16 hosts idle (MoT needs all 64 CPUs, 560G, 8 GPUs per host)"
else
  if [ $PRESTOP = 1 ]; then say "[after stop] hosts not free now (expected: MoT holds them): $(wc -l <<<"$BUSY") busy"
  else warn "hosts not fully free:"; echo "$BUSY" | sed 's/^/        /'; fi
  squeue -h -w "$HOSTS" -o '        %i %u %j %T %N' 2>/dev/null | head -20
fi

# ------------------------------------------------------------------ PHASE 3: saves + /scratch on every host (read-only)
phase "3 dcp-${S:-<newest>} and /scratch on every host (read-only probe)"
RUNJ=$(squeue -h -u "$(id -un)" -n ppm-mot -t R -o %i | head -1)
if [ -n "$RUNJ" ]; then say "probe = srun --overlap into running ppm-mot $RUNJ (-c1 --mem=256M --cpu-bind=none, read-only)"; SCAN=$(hostscan "$RUNJ" "$HOSTS" "$S")
elif [ -n "$BUSY" ] && [ "$DRY" = 1 ]; then say "probe skipped in dry-run: hosts busy and no MoT job to overlap (a real run probes once the hosts are free)"; SCAN=""
else
  if [ -n "$BUSY" ]; then say "waiting <= ${HW} min for the 16 hosts to be free before probing"; t0=$(date +%s)
    while BUSY=$(hosts_free); [ -n "$BUSY" ] && [ $(( $(date +%s) - t0 )) -lt $((HW * 60)) ]; do sleep 30; done; fi
  say "probe = tiny allocation -J mot-savecheck on the 16 hosts (--gres=none -c1 --mem=256M -t 3, read-only)"; SCAN=$(hostscan "" "$HOSTS" "$S")
fi
if [ -n "$SCAN" ]; then
  echo "$SCAN" | sed 's/^/        /'
  read -r SN SMIN SMINH SNEW SOK STMP SHOLD <<<"$(scan_summary "$SCAN")"
  [ "$SN" -eq 16 ] && ok "16/16 hosts answered" || fail "only $SN/16 hosts answered the probe"
  [ "$SOK" -eq 16 ] && ok "dcp-$S COMPLETE (=$S, 20 files) on 16/16 hosts" || fail "dcp-$S intact on only $SOK/16 hosts"
  [ "$SNEW" = "$S" ] && ok "newest COMPLETE step on every host = $S -> PPM_RESUME_FROM=auto picks dcp-$S" || fail "newest common COMPLETE step = $SNEW, not $S: auto would resume elsewhere"
  [ "$STMP" -eq 0 ] && ok "no dcp-*.tmp" || fail "$STMP host(s) hold a dcp-*.tmp"
  [ "$SHOLD" -eq 0 ] && ok "no hfhold/ snapshot on any host" || fail "$SHOLD host(s) still hold /scratch/amartyavarma/ppm-mot/hfhold/* (leftover state-backup hold; motguard review 4b): ask motguard to release it first"
  awk -v m="$SMIN" -v n="$L_MIN" 'BEGIN{exit !(m+0 >= n+0)}' && ok "min /scratch free ${SMIN} GiB ($SMINH) >= ${L_MIN} (mot.sbatch pre-flight ILP_MIN_SCRATCH_GB; first save needs ${NEED_SAVE_GIB})" \
    || fail "min /scratch free ${SMIN} GiB on $SMINH < ${L_MIN}: mot.sbatch pre-flight would ABORT (free the monoplex's /scratch files; never dcp-$S)"
  awk -v m="$SMIN" -v n="$GUARD_NEED_GIB" 'BEGIN{exit !(m+0 >= n+0)}' || warn "min /scratch ${SMIN} GiB < motguard's ${GUARD_NEED_GIB} GiB alert line"
  [ $PRESTOP = 0 ] && [ -n "${SCRATCH_FREE_GIB:-}" ] && say "free at the stop: $SCRATCH_FREE_GIB"
fi
DF=$(df -BG --output=avail /data | tail -1 | tr -dc 0-9); [ "$DF" -ge 386 ] && ok "/data free ${DF} GB (next 6-h export ~86 GB keeps >= 300)" || warn "/data free ${DF} GB: the next 6-h export (86 GB) would be REFUSED by the trainer (< 300 after) - not a launch blocker"

CMD="env $UNSET $OVR sbatch --mem=560G -N $NH -w $NODEF $SB"
echo; say "COMMAND: $OVR sbatch --mem=560G -N $NH -w $NODEF $SB"
say "EXPECT in the new log (in order, ~2.5 min to the first step on 10-25): '== submission overrides: $SUB_OVR' | rank order $RANK_ORDER | 16 '== scratch' lines, no ABORT | '[resume] auto: newest step COMPLETE on every node under $CKPT: $S' | '[resume] $CKPT/dcp-$S: step $S, ... per-rank digests == saved on every rank' | '[plan] steps $S..$MAXSTEP: ...' | cfg == job $STOP_JOB's except run_tag | 'EVAL step $S' ~ pre-stop EVALs ($PRE_EVALS) | 'step $((S + 1))/$MAXSTEP' ~27 s then ~21 s, word_ce_norm ~ $PRE_WCE (mean sd n)"

if [ ${#FAIL[@]} -gt 0 ]; then
  echo; echo "######## REFUSED: ${#FAIL[@]} failed check(s); nothing submitted ########"; printf '   - %s\n' "${FAIL[@]}"; exit 1
fi
if [ "$DRY" = 1 ]; then
  phase "dry-run: actions NOT taken"
  would "cp -p $M/logs/plan_dump_first4.json $GO/plan_dump_first4.j$STOP_JOB.json (the new job overwrites it)"
  would "rm $M/guard/PAUSED (only if go/mot_stop.sh created it; motguard precedent: removed at submit) ; keep watchdog/WATCHDOG_OFF until verified"
  would "submit: $OVR sbatch --mem=560G -N $NH -w $NODEF $SB   (stray SLURM_/PPM_/ILP_/MOT_ vars unset)"
  would "write the new id to $M/JOB + $M/watchdog/CURRENT_JOB; rewrite $LREC (same MOT_INIT / ILP_MIN_SCRATCH_GB / NODEF, new JOB)"
  would "wait RUNNING (<= ${WRUN} min, never cancel), then verify the EXPECT lines above (<= ${VMIN} min)"
  would "on PASS: rm watchdog/WATCHDOG_OFF and guard/state_backup/OFF (each only if mot_stop.sh set it) + $INPROG; write $GO/MOT_RESUMED; one line to $GL"
  echo; [ $PRESTOP = 1 ] && echo "== dry-run (pre-stop): every check that can run before the stop passed; nothing written" || echo "== dry-run: all checks passed; nothing written"; exit 0
fi

# ------------------------------------------------------------------ real resume
phase "4 submit"
BUSY=$(hosts_free)
if [ -n "$BUSY" ] && [ $QB = 0 ]; then
  say "hosts busy; waiting <= ${HW} min"; t0=$(date +%s)
  while BUSY=$(hosts_free); [ -n "$BUSY" ] && [ $(( $(date +%s) - t0 )) -lt $((HW * 60)) ]; do sleep 30; done
  [ -n "$BUSY" ] && { echo "REFUSED: hosts still busy after ${HW} min (use --queue-behind to submit and PEND):"; echo "$BUSY"; exit 1; }
fi
n=$(squeue -h -u "$(id -un)" -n ppm-mot -t R,PD | wc -l); [ "$n" -eq 0 ] || { echo "REFUSED: $n ppm-mot job(s) appeared in the queue"; exit 1; }
cp -p "$M/logs/plan_dump_first4.json" "$GO/plan_dump_first4.j$STOP_JOB.json" 2>/dev/null && ok "kept job $STOP_JOB's plan dump copy"
[ "${PAUSED_PRE:-0}" = 0 ] && rm -f "$M/guard/PAUSED" && ok "guard/PAUSED removed (motguard tick back to normal; WATCHDOG_OFF kept until verified)"
OUT=$($CMD 2>&1); JID=$(grep -oE 'Submitted batch job [0-9]+' <<<"$OUT" | grep -oE '[0-9]+$')
[ -n "$JID" ] || { echo "!!! sbatch FAILED: $OUT"; [ "${PAUSED_PRE:-0}" = 0 ] && echo "PLANNED PAUSE: resume sbatch failed $(pt); see go/" > "$M/guard/PAUSED"; exit 1; }
echo "$JID" > "$M/JOB"; echo "$JID" > "$M/watchdog/CURRENT_JOB"
cat > "$LREC" <<EOF
# written by monosam go/mot_resume.sh $(ptm) PT (planned-pause resume of job $STOP_JOB from dcp-$S; = launch.sh's record); read by watchdog/mot_watchdog.sh
JOB=$JID
MOT_INIT=$L_INIT
ILP_MIN_SCRATCH_GB=$L_MIN
NODEF=$NODEF
EOF
say "SUBMITTED job $JID"; glog "MoT RESUME submitted as job $JID from dcp-$S (go/mot_resume.sh; identical overrides to $STOP_JOB, $NODEF, resume auto); guard/PAUSED removed, WATCHDOG_OFF kept until verified"
L=$M/logs/ppm-mot-$JID.log

phase "5 wait RUNNING (<= ${WRUN} min)"
t0=$(date +%s); ln=0
until [ "$(squeue -h -j "$JID" -o %T 2>/dev/null)" = RUNNING ]; do
  st=$(squeue -h -j "$JID" -o '%T %r' 2>/dev/null)
  [ -z "$st" ] && { echo "!!! job $JID left the queue before RUNNING: $(sacct -n -X -j "$JID" -o State,ExitCode -P | head -1); log tail:"; tail -5 "$L" 2>/dev/null | sed 's/^/        /'; exit 5; }
  [ $(( $(date +%s) - t0 )) -ge $((WRUN * 60)) ] && { echo "!!! job $JID not RUNNING after ${WRUN} min ($st): left PENDING (not cancelled); WATCHDOG_OFF kept"; glog "resume job $JID still $st after ${WRUN} min"; exit 4; }
  [ $(( $(date +%s) - ln )) -ge 60 ] && { say "job $JID $st"; ln=$(date +%s); }; sleep 10
done
say "job $JID RUNNING on $(squeue -h -j "$JID" -o %N)"

phase "6 verify the resume (<= ${VMIN} min)"
verify_loop

if [ ${#FAIL[@]} -gt 0 ]; then
  echo; echo "######## RESUME job $JID SUBMITTED but ${#FAIL[@]} verification problem(s); WATCHDOG_OFF KEPT (no auto-relaunch): hand to motguard ########"
  printf '   - %s\n' "${FAIL[@]}"; glog "resume job $JID: verification FAILED (${FAIL[*]:0:3}); WATCHDOG_OFF kept; motguard to assess"; exit 5
fi
[ "${WD_OFF_PRE:-0}" = 0 ] && rm -f "$M/watchdog/WATCHDOG_OFF" && ok "watchdog/WATCHDOG_OFF removed (ppm-mot-watchdog adopts job $JID)"
[ "${SBOFF_PRE:-0}" = 0 ] && rm -f "$M/guard/state_backup/OFF" && ok "guard/state_backup/OFF removed (2-hourly HF state backup resumes with job $JID)"
rm -f "$INPROG"
cat > "$GO/MOT_RESUMED" <<EOF
RESUME_JOB=$JID
FROM_STEP=$S
STOP_JOB=$STOP_JOB
RESUMED_PT="$(pt)"
WARNINGS="${WARNS[*]:-none}"
EOF
glog "VERIFIED resume $JID: dcp-$S digests == saved every rank, plan $S..$MAXSTEP, cfg == $STOP_JOB except run_tag, EVAL/loss continuous; WATCHDOG_OFF + state_backup/OFF removed (guard fully armed)"
say "MoT RESUMED as job $JID from dcp-$S and verified${WARNS:+ (warnings: ${WARNS[*]})}"
exit 0
