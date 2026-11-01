#!/bin/bash
# monosam go/mot_stop.sh (10-26): PLANNED PAUSE of the MoT 14B (ppm-mot) with a clean save-and-exit, for the monoplex SAM FT.
# STAGED ONLY: run it ONLY when main relays the user's explicit go. The user said "do not pause without my explicit permission".
#
#   mot_stop.sh --dry-run    every read-only check (job, control dir, log cadence, guard / watchdog / state-backup, per-host save layout
#                            + /scratch via ONE tiny srun --overlap step) and prints exactly what it WOULD do. Writes nothing.
#   mot_stop.sh              the stop. Re-entrant: if a planned stop already finished (MOT_PAUSE_IN_PROGRESS + "STOPPED at step"), it
#                            only re-verifies and records.
#   options: --save-timeout-min=15 (request -> "[control] ... saved" line)  --exit-timeout-min=10 (save -> job gone)
#            --sb-wait-min=25 (wait for a running 2-hourly HF state-backup cycle to finish before asking for the stop)
#
# Mechanism (file:line in MOT_NOTES.md): touch <control>/STOP_AFTER_SAVE (control = mot.env PPM_CONTROL_DIR = ppm/mot/control); rank 0 polls
# it at every step boundary (safety_ppm.Control.poll, all-reduced MAX), the trainer saves dcp-<step> to node-local /scratch (barrier ->
# per-node COMPLETE -> barrier -> "[save] ..." line), consumes the file (-> STOP_AFTER_SAVE.done-<step>), logs "[control] STOP_AFTER_SAVE
# at step S: saved .../dcp-S -> exiting cleanly" + "STOPPED at step S", exits 0. Never scancel (memory stop-runs-with-save).
# Planned-pause markers (so nothing relaunches MoT onto the hosts or alarms): watchdog/WATCHDOG_OFF (ppm-mot-watchdog 28257: no actions),
# guard/PAUSED (motguard tick.sh paused mode), guard/state_backup/OFF (no HF state-backup cycle on the stop save; motguard review 4a),
# go/MOT_PAUSE_IN_PROGRESS. mot_resume.sh removes the ones this script created (after its verification; PAUSED at submit).
# Exit: 0 stopped + recorded | 1 pre-check refused | 3 save SKIPPED (stop retracted, run continues) | 4 no save within timeout (NO action
# taken: hand to motguard, stacks first) | 5 job ended without a clean STOPPED | 6 saved but job still in the queue after the timeout.
set -uo pipefail
. /data/home/amartyavarma/ppm/aloe-pose-mono-8020-sam/go/mot_common.sh
DRY=0; SAVE_TO=15; EXIT_TO=10; SBW=25
for a in "$@"; do case "$a" in
  --dry-run) DRY=1;; --save-timeout-min=*) SAVE_TO=${a#*=};; --exit-timeout-min=*) EXIT_TO=${a#*=};; --sb-wait-min=*) SBW=${a#*=};;
  *) echo "usage: mot_stop.sh [--dry-run] [--save-timeout-min=15] [--exit-timeout-min=10] [--sb-wait-min=25]"; exit 2;; esac; done
echo "== mot_stop.sh $(pt)  dry-run=$DRY  (planned MoT pause for the monoplex SAM FT; main relays the user's go)"
if [ "$DRY" = 0 ]; then exec 9>"$GO/.mot_stop.lock"; flock -n 9 || { echo "REFUSED: another mot_stop.sh is running"; exit 1; }; fi

# ------------------------------------------------------------------ PHASE 0: read-only pre-checks
phase "0 pre-checks (read-only)"
for f in "$M/JOB" "$M/watchdog/CURRENT_JOB" "$LREC" "$ENVF" "$SB" "$HC" "$M/guard/tick.sh" "$M/watchdog/mot_watchdog.sh"; do
  [ -f "$f" ] && ok "exists $f" || fail "missing $f"; done
J=$(tr -dc 0-9 < "$M/JOB"); CJ=$(tr -dc 0-9 < "$M/watchdog/CURRENT_JOB")
LJ=$(grep -E '^JOB=' "$LREC" | tail -1 | cut -d= -f2 | tr -dc 0-9)
NODEF=$(grep -E '^NODEF=' "$LREC" | tail -1 | cut -d= -f2-); L_INIT=$(grep -E '^MOT_INIT=' "$LREC" | tail -1 | cut -d= -f2-)
L_MIN=$(grep -E '^ILP_MIN_SCRATCH_GB=' "$LREC" | tail -1 | cut -d= -f2-)
[ "$J" = "$CJ" ] && [ "$J" = "$LJ" ] && ok "job id $J (JOB = watchdog/CURRENT_JOB = LAST_LAUNCH.env)" || fail "job ids disagree: JOB=$J CURRENT_JOB=$CJ LAST_LAUNCH=$LJ"
[ -s "$NODEF" ] || fail "host file '$NODEF' (LAST_LAUNCH.env NODEF) missing"
HOSTS=$(tr -d ' \n' < "$NODEF" 2>/dev/null); NH=$(tr , '\n' <<<"$HOSTS" | grep -c .)
[ "$NH" -eq 16 ] && ok "host file $NODEF: 16 hosts $HOSTS" || fail "host file has $NH hosts"
grep -qw slinky-9 <<<"${HOSTS//,/ }" && fail "slinky-9 in the host list (never slinky-9)"
L=$M/logs/ppm-mot-$J.log
[ -f "$L" ] && ok "log $L" || fail "log $L missing"

# re-entrant: a planned stop already ran to the end?
REENTER=0
if [ -f "$INPROG" ] && ! squeue -h -j "$J" 2>/dev/null | grep -q . && grep -aqE 'STOPPED at step [0-9]+' "$L" 2>/dev/null; then
  REENTER=1; say "MOT_PAUSE_IN_PROGRESS present, job $J gone and its log has STOPPED: skipping to verification / record"
fi

if [ $REENTER = 0 ]; then
  read -r JN JST JNL <<<"$(squeue -h -j "$J" -o '%j %T %N' 2>/dev/null)"
  [ "$JN" = ppm-mot ] && [ "$JST" = RUNNING ] && ok "job $J ppm-mot RUNNING on $JNL" || fail "job $J is not a RUNNING ppm-mot (name '${JN:-}', state '${JST:-gone}')"
  NMOT=$(squeue -h -u "$(id -un)" -n ppm-mot -o %i | wc -l); [ "$NMOT" -eq 1 ] && ok "exactly 1 ppm-mot job in the queue" || fail "$NMOT ppm-mot jobs in the queue"
  if [ -n "${JNL:-}" ]; then
    A=$(scontrol show hostnames "$JNL" | sort -V | paste -sd,); B=$(tr , '\n' <<<"$HOSTS" | sort -V | paste -sd,)
    [ "$A" = "$B" ] && ok "job nodelist == host file (16 hosts)" || fail "job nodelist $A != host file $B"
  fi
  HET=$(scontrol show job "$J" 2>/dev/null | grep -c 'HetJob'); [ "$HET" -eq 0 ] && ok "job $J is NOT heterogeneous (srun --overlap allowed)" || fail "job $J is heterogeneous: no overlap steps allowed"
fi

# control dir: mot.env PPM_CONTROL_DIR == the trainer's logged control dir == mot.sbatch default
CTL_ENV=$(grep -E '^PPM_CONTROL_DIR=' "$ENVF" | tail -1 | cut -d= -f2-)
CTL_LOG=$(grep -a -m1 -oE '\[safety\] checkpoints [^ ]+/dcp-N .*control [^ ]+/SAVE_NOW' "$L" 2>/dev/null | sed -E 's/.*control ([^ ]+)\/SAVE_NOW/\1/')
CK_LOG=$(grep -a -m1 -oE '\[safety\] checkpoints [^ ]+/dcp-N' "$L" 2>/dev/null | sed -E 's/\[safety\] checkpoints ([^ ]+)\/dcp-N/\1/')
CTL=${CTL_LOG:-$CTL_ENV}
[ "$CTL_ENV" = "$CTL_DEFAULT" ] && [ "$CTL_LOG" = "$CTL_DEFAULT" ] && ok "control dir $CTL (mot.env = trainer log = mot.sbatch default)" || fail "control dir mismatch: env '$CTL_ENV' log '$CTL_LOG' default '$CTL_DEFAULT'"
[ "$CK_LOG" = "$CKPT" ] && ok "node-local save root $CKPT (trainer log)" || fail "save root mismatch: log '$CK_LOG' vs $CKPT"
[ -d "$CTL" ] && [ -w "$CTL" ] && ok "control dir writable (test -w, nothing written)" || fail "control dir $CTL not a writable directory"
if [ $REENTER = 0 ]; then
  for x in STOP_AFTER_SAVE SAVE_NOW; do [ -e "$CTL/$x" ] && fail "$CTL/$x already present (someone else's request in flight: $(head -c 80 "$CTL/$x"))" || ok "no pending $x"; done
fi
ls "$CTL" 2>/dev/null | sed 's/^/        control: /'

# log cadence: step freshness, s/step, last save / eval / export
if [ $REENTER = 0 ]; then
  now=$(date +%s); LS=$(last_step_line "$L"); STEP=$(step_of "$LS"); AGE=$(( now - $(line_epoch "$LS") ))
  SPS=$(sps_avg "$L" 28)
  [ -n "$STEP" ] && [ "$AGE" -lt 240 ] && ok "last step $STEP/$MAXSTEP ${AGE}s ago, s/step avg(28) $SPS" || fail "no fresh step line (last '${STEP:-none}' ${AGE}s ago): a stalled job is motguard's incident (stacks first), not a planned stop"
  SL=$(grep -aE '\[save\] /scratch/[^ ]+/dcp-[0-9]+ node-local' "$L" | tail -1); SV=$(sed -nE 's/.*\/dcp-([0-9]+) node-local.*/\1/p' <<<"$SL")
  SA=$(( (now - $(line_epoch "$SL")) / 60 )); SD=$(sed -nE 's/.*\(([0-9.]+)s\).*/\1/p' <<<"$SL")
  say "last save dcp-$SV ${SA} min ago (${SD}s write); hourly cadence (PPM_SAVE_EVERY_MIN=60) -> next ~$((60 - SA)) min; a STOP writes its own save regardless"
  EL=$(grep -aE '\] EVAL step [0-9]+:' "$L" | tail -1); EA=$(( (now - $(line_epoch "$EL")) / 60 ))
  say "last EVAL $(sed -nE 's/.*EVAL step ([0-9]+):.*/\1/p' <<<"$EL") ${EA} min ago (every 30 min, ~45-70 s) -> next ~$((30 - EA)) min"
  XL=$(grep -aE '\[export\] .*/w-[0-9]+: consolidated' "$L" | tail -1)
  if [ -n "$XL" ]; then XA=$(( (now - $(line_epoch "$XL")) / 60 )); say "last export $(grep -oE 'w-[0-9]+' <<<"$XL" | head -1) ${XA} min ago (every 360 min, ~15 s) -> next ~$((360 - XA)) min"; fi
  [ $((30 - EA)) -le 2 ] && say "note: an EVAL is due at about the stop boundary (+45-70 s before the save)"
  read -r WM WS WN <<<"$(wce_stats "$L" 50)"; say "pre-stop train word_ce_norm (speech) last $WN steps: mean $WM sd $WS"
  say "pre-stop EVALs (step word_ce_norm): $(evals "$L" | tail -3 | paste -sd';')"
fi

# guard / watchdog / companions / state backup
[ -f "$M/watchdog/WATCHDOG_OFF" ] && warn "watchdog/WATCHDOG_OFF already present (kept; resume will NOT remove it)" || ok "watchdog/WATCHDOG_OFF absent (watchdog armed)"
[ -f "$M/guard/PAUSED" ] && warn "guard/PAUSED already present: $(head -c 100 "$M/guard/PAUSED")" || ok "guard/PAUSED absent (motguard tick in normal mode)"
HB=$M/GUARD_HEARTBEAT; [ -f "$HB" ] && say "motguard heartbeat ($(( ($(date +%s) - $(stat -c %Y "$HB")) / 60 )) min old): $(cat "$HB")"
for nm in ppm-mot-watchdog mot-state-bk mot-hf-watch motlive-gpu; do
  r=$(squeue -h -u "$(id -un)" -n $nm -o '%i %T %N' | head -1); [ -n "$r" ] && ok "companion $nm: $r" || warn "companion $nm not in the queue"; done
SBS=$M/guard/state_backup/state.json
if grep -q '"cycle": null' "$SBS" 2>/dev/null; then ok "HF full-state backup (mot-state-bk): no cycle in flight (state.json cycle null); status: $(head -c 120 "$M/guard/state/fullstate_status" 2>/dev/null)"
else warn "HF full-state backup cycle IN FLIGHT (hardlink hold + upload inside the job): the real stop waits up to ${SBW} min for it (a stop now would prune its dcp and the hold would then cost ~34 GiB/host)"; fi
[ -f "$M/guard/state_backup/OFF" ] && warn "state_backup/OFF present"
DF=$(df -BG --output=avail /data | tail -1 | tr -dc 0-9); say "/data free ${DF} GB"

# per-host save layout + /scratch (ONE tiny read-only srun --overlap step into the MoT job)
if [ $REENTER = 0 ] && [ "${HET:-1}" -eq 0 ] && [ "${JST:-}" = RUNNING ]; then
  say "per-host scan: srun --overlap --jobid $J -N16 --ntasks-per-node=1 -c1 --mem=256M --cpu-bind=none -t 3 nice -n 19 bash $HC (read-only)"
  SCAN=$(hostscan "$J" "$HOSTS"); echo "$SCAN" | sed 's/^/        /'
  read -r SN SMIN SMINH SNEW SOK STMP SHOLD <<<"$(scan_summary "$SCAN")"
  [ "$SN" -eq 16 ] && ok "16/16 hosts answered" || fail "only $SN/16 hosts answered the scan"
  [ "$SNEW" -ge 0 ] && ok "newest save COMPLETE on all 16 hosts: dcp-$SNEW" || fail "no save COMPLETE on all 16 hosts"
  awk -v m="$SMIN" -v n="$NEED_SAVE_GIB" 'BEGIN{exit !(m+0 >= n+0)}' && ok "min /scratch free ${SMIN} GiB ($SMINH) >= ${NEED_SAVE_GIB} GiB needed for the stop save (save x1.002 + 30 floor; PPM_SAVE_PRUNE_FIRST=0)" \
    || fail "min /scratch free ${SMIN} GiB on $SMINH < ${NEED_SAVE_GIB}: the STOP save would be SKIPPED (the run keeps training)"
  [ "$STMP" -eq 0 ] || warn "$STMP host(s) hold a dcp-*.tmp (save in progress at scan time?)"
  [ "$SHOLD" -eq 0 ] && ok "no hfhold/ snapshot on any host" || warn "$SHOLD host(s) have hfhold/ entries (state-backup hold)"
fi

echo; say "EXPECTED STOP TIME: next step boundary (<= ~21 s; +45-70 s if the 30-min EVAL falls there; +~15 s if the 6-h export does) + save ~15-18 s + prune ~4 s + exit ~10 s = ~45-120 s (10-24: request 23:57:43 -> exit 0 23:58:23)"

if [ ${#FAIL[@]} -gt 0 ]; then
  echo; echo "######## REFUSED: ${#FAIL[@]} failed check(s); nothing written ########"; printf '   - %s\n' "${FAIL[@]}"; exit 1
fi
if [ "$DRY" = 1 ]; then
  phase "dry-run: actions NOT taken"
  would "wait (<= ${SBW} min, 30 s polls) while guard/state_backup/state.json has a cycle in flight, then touch $M/guard/state_backup/OFF (no new HF state-backup cycle on the stop save; motguard review 4a) and re-check no cycle started"
  would "write $INPROG and $STATE (job $J, hosts, recipe sha $(recipe_sha "$NODEF"), cfg sha $(cfg_sha "$L"), submission overrides, rank order, pre-stop EVALs / loss stats)"
  would "touch $M/watchdog/WATCHDOG_OFF  (ppm-mot-watchdog 28257: no actions while set; a deliberate STOP is not relaunched anyway: mot_watchdog.sh:193-194)"
  would "write $M/guard/PAUSED = 'PLANNED PAUSE (user GO via main): monoplex SAM FT on 12 MoT hosts; resume = go/mot_resume.sh; set by monosam'  (motguard tick.sh:11-17 paused mode)"
  would "append one line to $GL"
  would "echo 'monosam go/mot_stop.sh <PT time>: planned pause' > $CTL/STOP_AFTER_SAVE"
  would "poll $L every 5 s for '[control] STOP_AFTER_SAVE at step S: saved $CKPT/dcp-S -> exiting cleanly' (<= ${SAVE_TO} min), 'STOPPED at step S', '== exit 0', job gone (<= ${EXIT_TO} min)"
  would "on '!!! [control] STOP_AFTER_SAVE ... SKIPPED': retract the request (mv STOP_AFTER_SAVE -> .retracted-<ts>), remove the markers it set, exit 3"
  would "on timeout: print diagnostics, take NO action (no scancel), keep the markers, exit 4 (motguard: stacks first)"
  would "verify: sacct COMPLETED 0:0, $CTL/STOP_AFTER_SAVE.done-S, and a read-only per-host check of dcp-S (20 files, COMPLETE=S, newest=S, /scratch free) via srun -p all -N16 -w <hosts> -c1 --mem=256M --gres=none -t 3 --immediate=300 -J mot-savecheck"
  would "record S -> $STEPF ; append STOP_STEP / STOPPED_PT / per-host free to $STATE ; one line to $GL"
  echo; echo "== dry-run: all checks passed; nothing written, nothing signalled"; exit 0
fi

# ------------------------------------------------------------------ real stop
if [ $REENTER = 0 ]; then
  phase "1 wait for any in-flight HF state-backup cycle, then state_backup/OFF (motguard review 4a)"
  # OFF = no new cycle (sb_driver.py docstring): stops the driver from starting a cycle on the stop save's fresh [save] line while the job
  # exits. Set only when no cycle is in flight (OFF would abort a running upload); re-checked after the touch to close the race window.
  SBOFF=$M/guard/state_backup/OFF; SBOFF_PRE=0; [ -f "$SBOFF" ] && SBOFF_PRE=1
  t0=$(date +%s)
  while :; do
    until grep -q '"cycle": null' "$SBS" 2>/dev/null; do
      [ $(( $(date +%s) - t0 )) -ge $((SBW * 60)) ] && { [ $SBOFF_PRE = 0 ] && rm -f "$SBOFF"; echo "!!! state-backup cycle still in flight after ${SBW} min: NOT stopping (ask motguard)"; exit 1; }
      say "state-backup cycle in flight: $(head -c 100 "$M/guard/state/fullstate_status" 2>/dev/null); waiting"; sleep 30
    done
    [ $SBOFF_PRE = 0 ] && touch "$SBOFF"
    grep -q '"cycle": null' "$SBS" 2>/dev/null && break
    [ $SBOFF_PRE = 0 ] && rm -f "$SBOFF"; say "a cycle started in the touch window: OFF lifted again, waiting for it"
  done
  ok "no state-backup cycle in flight; state_backup/OFF in place (pre-existing: $SBOFF_PRE)"

  phase "2 planned-pause markers"
  WD_PRE=0; [ -f "$M/watchdog/WATCHDOG_OFF" ] && WD_PRE=1
  PA_PRE=0; [ -f "$M/guard/PAUSED" ] && PA_PRE=1
  cat > "$STATE" <<EOF
# written by monosam go/mot_stop.sh $(pt); read by go/mot_resume.sh
STOP_JOB=$J
NODEF=$NODEF
HOSTS=$HOSTS
MOT_INIT=$L_INIT
ILP_MIN_SCRATCH_GB=$L_MIN
RECIPE_SHA=$(recipe_sha "$NODEF")
RECIPE_NEWEST_MTIME=$(recipe_newest_mtime "$NODEF")
CFG_SHA=$(cfg_sha "$L")
SUB_OVR="$(sub_ovr "$L")"
RANK_ORDER=$(rank_order "$L")
PRE_WCE="$WM $WS $WN"
PRE_EVALS="$(evals "$L" | tail -3 | paste -sd';')"
PRE_SPS=$SPS
WD_OFF_PRE=$WD_PRE
PAUSED_PRE=$PA_PRE
SBOFF_PRE=$SBOFF_PRE
STOP_REQ_PT="$(pt)"
EOF
  echo "planned MoT pause: job $J, requested $(pt) by monosam go/mot_stop.sh (main relayed the user's go)" > "$INPROG"
  touch "$M/watchdog/WATCHDOG_OFF"; ok "touched watchdog/WATCHDOG_OFF (pre-existing: $WD_PRE)"
  [ $PA_PRE = 0 ] && echo "PLANNED PAUSE (user GO via main) $(pt): monoplex SAM FT on 12 MoT hosts; resume = /data/home/amartyavarma/ppm/aloe-pose-mono-8020-sam/go/mot_resume.sh; set by monosam go/mot_stop.sh" > "$M/guard/PAUSED"
  ok "guard/PAUSED in place (pre-existing: $PA_PRE)"
  glog "STOP requested for job $J (step ~$STEP): WATCHDOG_OFF + guard/PAUSED + state_backup/OFF set, STOP_AFTER_SAVE next (go/mot_stop.sh; user GO via main)"

  phase "3 request save-and-exit"
  read -r JN JST _ <<<"$(squeue -h -j "$J" -o '%j %T %N' 2>/dev/null)"
  [ "$JN" = ppm-mot ] && [ "$JST" = RUNNING ] || { echo "!!! job $J no longer RUNNING ppm-mot ($JN $JST): NOT requesting; markers kept"; exit 5; }
  [ -e "$CTL/STOP_AFTER_SAVE" ] && { echo "!!! $CTL/STOP_AFTER_SAVE appeared meanwhile: not overwriting; markers kept"; exit 1; }
  OFF=$(stat -c %s "$L"); T_REQ=$(date +%s)
  echo "monosam go/mot_stop.sh $(pt): planned pause (user GO via main)" > "$CTL/STOP_AFTER_SAVE"
  say "STOP_AFTER_SAVE written ($CTL/STOP_AFTER_SAVE); log offset $OFF"

  phase "4 wait for the save (<= ${SAVE_TO} min) and the exit (<= ${EXIT_TO} min)"
  S=""; T_SAVED=""; ERRSEEN=0; last_note=0
  while :; do
    NEW=$(tail -c +$((OFF + 1)) "$L" 2>/dev/null)
    if grep -aqE '!!! \[control\] STOP_AFTER_SAVE at step [0-9]+: the save was SKIPPED' <<<"$NEW" && [ -z "$S" ]; then
      echo "!!! the STOP save was SKIPPED (no /scratch space):"; grep -aE '!!! \[save\]|SKIPPED' <<<"$NEW" | tail -20 | sed 's/^/        /'
      mv "$CTL/STOP_AFTER_SAVE" "$CTL/STOP_AFTER_SAVE.retracted-$(date +%s)" 2>/dev/null && say "request retracted (run keeps training)"
      sleep 60; NEW=$(tail -c +$((OFF + 1)) "$L")
      if ! grep -aqE '\[control\] STOP_AFTER_SAVE at step [0-9]+: saved' <<<"$NEW"; then
        [ "$WD_PRE" = 0 ] && rm -f "$M/watchdog/WATCHDOG_OFF"; [ "$PA_PRE" = 0 ] && rm -f "$M/guard/PAUSED"; [ "$SBOFF_PRE" = 0 ] && rm -f "$SBOFF"; rm -f "$INPROG"
        glog "STOP save SKIPPED (no /scratch space) on job $J: request retracted, markers removed, run keeps training"
        echo "== exit 3: no stop; free /scratch on the hosts listed above, then rerun"; exit 3
      fi
    fi
    if [ -z "$S" ]; then
      S=$(grep -aoE '\[control\] STOP_AFTER_SAVE at step [0-9]+: saved [^ ]+/dcp-[0-9]+ -> exiting cleanly' <<<"$NEW" | tail -1 | sed -nE 's/.*at step ([0-9]+):.*/\1/p')
      if [ -n "$S" ]; then T_SAVED=$(date +%s); say "SAVE COMPLETE: dcp-$S on all 16 hosts (the '[save]' line follows the post-COMPLETE barrier: safety_ppm.save_local) after $((T_SAVED - T_REQ)) s"
        grep -aE "\[save\] $CKPT/dcp-$S node-local|keep-last-1|\[control\]" <<<"$NEW" | sed 's/^/        /'; fi
    fi
    if [ $ERRSEEN = 0 ] && grep -aqE 'Traceback|NCCL error|CUDA error|OutOfMemory|srun: error|^== exit [1-9]' <<<"$NEW"; then
      ERRSEEN=1; echo "!!! error lines in the log since the request:"; grep -aE 'Traceback|NCCL error|CUDA error|OutOfMemory|srun: error|^== exit' <<<"$NEW" | head -5 | sed 's/^/        /'; fi
    INQ=$(squeue -h -j "$J" -o %T 2>/dev/null)
    if [ -z "$INQ" ]; then
      for _ in 1 2 3 4 5 6; do grep -aqE "STOPPED at step [0-9]+ " <<<"$NEW" && break; sleep 5; NEW=$(tail -c +$((OFF + 1)) "$L" 2>/dev/null); done   # log flush grace
      [ -z "$S" ] && S=$(grep -aoE '\[control\] STOP_AFTER_SAVE at step [0-9]+: saved [^ ]+/dcp-[0-9]+ -> exiting cleanly' <<<"$NEW" | tail -1 | sed -nE 's/.*at step ([0-9]+):.*/\1/p')
      if [ -n "$S" ] && grep -aqE "STOPPED at step $S " <<<"$NEW"; then say "job $J left the queue: '$(grep -aE '^== exit' <<<"$NEW" | tail -1)'"; break; fi
      echo "!!! job $J left the queue WITHOUT a clean 'STOPPED at step' after a save (S='${S}'). No relaunch (WATCHDOG_OFF kept). Last log lines:"
      tail -8 "$L" | cut -c1-250 | sed 's/^/        /'; glog "job $J ended WITHOUT a clean STOPPED during the planned stop (S='${S}'); markers kept; motguard to assess"; exit 5
    fi
    now=$(date +%s)
    if [ -z "$S" ] && [ $((now - T_REQ)) -ge $((SAVE_TO * 60)) ]; then
      echo "!!! no '[control] ... saved' line ${SAVE_TO} min after the request. NO action taken (no scancel). Diagnostics:"
      echo "        squeue: $(squeue -h -j "$J" -o '%i %T %M %N')"; echo "        control: $(ls "$CTL" | tr '\n' ' ')"
      echo "        last step: $(last_step_line "$L" | cut -c1-120)"; tail -5 "$L" | cut -c1-250 | sed 's/^/        /'
      echo "        -> hand to motguard: stacks + NCCL flight recorder FIRST (touch $M/logs/DEBUG_DUMP; per-rank files in $M/logs/debug-$J), then its hang procedure."
      echo "        STOP_AFTER_SAVE, WATCHDOG_OFF and guard/PAUSED are left in place (the stop still completes if the job recovers)."
      glog "STOP request on job $J: no save after ${SAVE_TO} min; no action taken, handed to motguard"; exit 4
    fi
    if [ -n "$S" ] && [ $((now - T_SAVED)) -ge $((EXIT_TO * 60)) ]; then
      echo "!!! dcp-$S is COMPLETE but job $J is still '$INQ' ${EXIT_TO} min later. NO scancel. Last lines:"; tail -5 "$L" | cut -c1-250 | sed 's/^/        /'
      echo "$S" > "$STEPF"; echo "STOP_STEP=$S" >> "$STATE"
      glog "job $J saved dcp-$S (COMPLETE) but did not leave the queue within ${EXIT_TO} min; recorded; no scancel"; exit 6
    fi
    [ $((now - last_note)) -ge 30 ] && { say "waiting: state=$INQ saved=${S:-no} last: $(tail -1 "$L" | cut -c1-110)"; last_note=$now; }
    sleep 5
  done
else
  S=$(grep -aoE 'STOPPED at step [0-9]+' "$L" | tail -1 | grep -oE '[0-9]+$'); WD_PRE=$(grep -E '^WD_OFF_PRE=' "$STATE" 2>/dev/null | cut -d= -f2)
fi

# ------------------------------------------------------------------ PHASE 5: verify + record
phase "5 verify dcp-$S and record"
grep -aqE "\[save\] $CKPT/dcp-$S node-local per-rank state" "$L" && ok "log: [save] $CKPT/dcp-$S (all ranks past the post-COMPLETE barrier)" || fail "log has no [save] line for dcp-$S"
grep -aqE "STOPPED at step $S \(clean exit\)" "$L" && ok "log: STOPPED at step $S (clean exit)" || fail "log has no 'STOPPED at step $S'"
grep -aqE '^== exit 0' "$L" && ok "log: == exit 0" || warn "no '== exit 0' line"
[ -e "$CTL/STOP_AFTER_SAVE.done-$S" ] && ok "control file consumed: STOP_AFTER_SAVE.done-$S" || warn "no $CTL/STOP_AFTER_SAVE.done-$S"
[ -e "$CTL/STOP_AFTER_SAVE" ] && fail "$CTL/STOP_AFTER_SAVE still present (would stop the resumed job at once)"
SA=$(sacct -n -X -j "$J" -o State,ExitCode -P 2>/dev/null | head -1); [[ "$SA" == COMPLETED\|0:0* ]] && ok "sacct $J: $SA" || warn "sacct $J: '$SA'"
say "per-host read-only check (job gone -> tiny allocation -J mot-savecheck, --gres=none -c1 --mem=256M, exits by itself)"
SCAN=$(hostscan "" "$HOSTS" "$S"); echo "$SCAN" | sed 's/^/        /'
read -r SN SMIN SMINH SNEW SOK STMP SHOLD <<<"$(scan_summary "$SCAN")"
if [ "$SN" -eq 16 ]; then
  [ "$SOK" -eq 16 ] && ok "dcp-$S COMPLETE (=$S, 20 files) on 16/16 hosts" || fail "dcp-$S verified on only $SOK/16 hosts"
  [ "$SNEW" = "$S" ] && ok "newest COMPLETE on every host = $S (PPM_RESUME_FROM=auto will pick it)" || fail "newest common COMPLETE step is $SNEW, not $S"
  [ "$STMP" -eq 0 ] || fail "$STMP host(s) still hold a dcp-*.tmp"
else
  warn "per-host check answered by $SN/16 hosts (allocation not granted in time?): relying on the barrier-proven log line; mot_resume.sh re-checks every host"
fi
echo "$S" > "$STEPF"
{ echo "STOP_STEP=$S"; echo "STOPPED_PT=\"$(pt)\""; echo "SCRATCH_FREE_GIB=\"$(awk '{f=$3; sub("free_GiB=","",f); printf "%s:%s ", $2, f}' <<<"$SCAN")\""; } >> "$STATE"
glog "job $J STOPPED cleanly at step $S (dcp-$S COMPLETE on 16/16, node-local, keep; HF state backup keeps the last 2-hourly dcp); WATCHDOG_OFF + guard/PAUSED + state_backup/OFF stay until go/mot_resume.sh"
if [ ${#FAIL[@]} -gt 0 ]; then echo "######## STOPPED at $S but ${#FAIL[@]} verification problem(s):"; printf '   - %s\n' "${FAIL[@]}"; exit 5; fi
echo; say "MoT STOPPED at step $S; recorded in $STEPF. Hosts are free. dcp-$S must stay on every host (never delete); each host needs >= ${L_MIN} GiB free again before go/mot_resume.sh (first new save: ${NEED_SAVE_GIB} GiB)."
exit 0
