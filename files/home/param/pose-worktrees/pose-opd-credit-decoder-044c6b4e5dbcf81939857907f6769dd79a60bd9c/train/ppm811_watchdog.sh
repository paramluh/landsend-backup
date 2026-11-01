#!/bin/bash
# 8,1,1 STAGE watchdog (code-811) = ppm/code/ppm_watchdog.sh with the 8,1,1 code / env / sbatch / job name / target and its OWN state files
# (run base ppm-v1 shares the control dir with PPM: STOP_AFTER_SAVE / SAVE_NOW are the same files). Disable: touch control/ppm-v1/WATCHDOG811_OFF.
# PPM cluster-side watchdog (independent of any Claude session). Runs as a CPU-only job on slinky-15 (ppm811_watchdog.sbatch).
# Every 60 s: state of the job in $CTL/CURRENT_JOB.
#  - gone without a clean finish ("done $TARGET steps", no STOPPED) and without a deliberate STOP_AFTER_SAVE -> find the newest dcp-N
#    COMPLETE on all 16 hosts, clear any stop marker, resubmit the IDENTICAL sbatch (same env file, same host list), record the new id.
#  - RUNNING but no new step for HANG_MIN -> touch STOP_AFTER_SAVE (watchdog-owned); still no progress HANG_MIN later -> confirm a
#    COMPLETE save, scancel, resubmit.
#  - at most MAX_RL relaunches per RL_WIN_H hours; then it stops relaunching and writes an ALERT.
#  - (10-19) at most 2 relaunches after failures at the SAME logged step (a deterministic hang/crash); then it stops and writes an ALERT.
#  - $CTL/WATCHDOG_OFF present -> observe only.  DRY=1 -> log the actions, never sbatch / scancel / touch / mv.
# Never changes the recipe; resubmission = the exact 8,1,1 launch command (code-811/LAUNCH_811.md).
set -u
CTL=${CTL:-/data/home/amartyavarma/ppm/control/ppm-v1}
MD=${MD:-/data/home/amartyavarma/ppm/control/overnight_monitor_811.md}
LOGDIR=${LOGDIR:-/data/home/amartyavarma/ppm/logs}
CODE=/data/home/amartyavarma/ppm/pose5/code-811-s2p
ENVF=$CODE/prod_811.env
NODEF=/data/home/amartyavarma/ppm/code/nodes16.txt
SB=$CODE/ppm811.sbatch
JNAME=ppm811
CKPT=/scratch/amartyavarma/ppm/ckpt/ppm-v1
TARGET=${TARGET:-19063}                  # = prod_811.env PPM_MAX_STEPS (switch at 16861)
MIN_STEP=${MIN_STEP:-16861}          # never relaunch onto anything older than the 8,1,1 switch step (set MIN_STEP=<S0> if switched earlier)
HANG_MIN=${HANG_MIN:-15}
MAX_RL=${MAX_RL:-3}; RL_WIN_H=${RL_WIN_H:-6}
POLL=${POLL:-60}
DRY=${DRY:-0}
ONESHOT=${ONESHOT:-0}                # test: one decision pass, then exit
NOADOPT=${NOADOPT:-0}                # test: do not adopt newer ppm jobs
SCAN_JOB=${SCAN_JOB:-}               # test: overlap this running job for the relaunch scan instead of a new tiny job
ST=${ST:-$CTL/watchdog811_state}; mkdir -p "$ST"
JOBF=${JOBF:-$CTL/CURRENT_JOB_811}
HOSTS=$(cat "$NODEF")
NHOST=$(echo "$HOSTS" | tr , '\n' | wc -l)

pt() { TZ=America/Los_Angeles date "+%a %m-%d %H:%M PT"; }
note() { local tag=watchdog; [ "$DRY" = 1 ] && tag="watchdog DRY"; local m="- $(pt): [$tag] $*"; echo "$m"; [ "$DRY" = 1 ] || echo "$m" >> "$MD"; }
UNSET=$(env | grep -oE '^(SLURM|SBATCH|SRUN|SALLOC|PPM)_[A-Za-z0-9_]*=' | tr -d = | sed 's/^/-u /' | tr '\n' ' ')   # no Slurm job vars, no stray PPM_* overrides
jstate() { squeue -h -j "$1" -o %T 2>/dev/null | head -1; }
newest_step() { tail -c 3000000 "$1" 2>/dev/null | grep -aoE "\] step [0-9]+/[0-9]+ " | tail -1 | grep -oE "[0-9]+/" | tr -d /; }

# newest step that is COMPLETE on every one of the 16 hosts ("" if the scan failed / none). $1 = running job id to overlap, or "" (new tiny job)
scan_complete() {
  local out
  if [ -n "$1" ]; then
    out=$(timeout 180 env $UNSET srun --jobid="$1" --overlap -N"$NHOST" -w "$HOSTS" --ntasks-per-node=1 -c1 --mem=1G --cpu-bind=none \
          bash -c "for d in $CKPT/dcp-*; do [ -f \$d/COMPLETE ] && echo \$(hostname) \${d##*-}; done; echo \$(hostname) END" 2>/dev/null)
  else
    out=$(timeout 600 env $UNSET srun -p all -N"$NHOST" -w "$HOSTS" --ntasks-per-node=1 -c1 --mem=1G -t 5 --immediate=300 -J ppm-wd-scan \
          bash -c "for d in $CKPT/dcp-*; do [ -f \$d/COMPLETE ] && echo \$(hostname) \${d##*-}; done; echo \$(hostname) END" 2>/dev/null)
  fi
  local nend; nend=$(echo "$out" | grep -c " END$")
  [ "$nend" -eq "$NHOST" ] || { echo ""; return; }
  # steps present on every host
  echo "$out" | grep -v " END$" | awk '{print $2}' | sort -n | uniq -c | awk -v n="$NHOST" '$1==n{print $2}' | tail -1
}

relaunches_recent() { local now; now=$(date +%s); [ -f "$ST/relaunches" ] || { echo 0; return; }; awk -v c=$((now - RL_WIN_H * 3600)) '$1>=c' "$ST/relaunches" | wc -l; }

relaunch() {   # $1 = reason, $2 = running job id to overlap for the scan (or "")
  local why=$1 n s
  n=$(relaunches_recent)
  if [ "$n" -ge "$MAX_RL" ]; then
    [ -f "$ST/capped" ] || note "ALERT: relaunch NOT done ($why): cap $MAX_RL relaunches per ${RL_WIN_H} h reached; needs a human/agent."
    [ "$DRY" = 1 ] || touch "$ST/capped"; return 1
  fi
  # 10-19: never loop on a deterministic failure: after 2 relaunches following a failure at the SAME logged step, stop and alert
  local at; at=$(echo "$why" | grep -oE 'at step [0-9]+' | grep -oE '[0-9]+$')
  if [ -n "$at" ] && [ -f "$ST/relaunch_at" ] && [ "$(awk -v a="$at" '$2 == a' "$ST/relaunch_at" | wc -l)" -ge 2 ]; then
    [ -f "$ST/samestep-$at" ] || note "ALERT: relaunch NOT done ($why): already relaunched twice after a failure at step $at (deterministic?); needs a human/agent."
    [ "$DRY" = 1 ] || touch "$ST/samestep-$at"; return 1
  fi
  s=$(scan_complete "${2:-$SCAN_JOB}")
  if [ -z "$s" ] || [ "$s" -lt "$MIN_STEP" ]; then
    note "ALERT: relaunch NOT done ($why): no save COMPLETE on all $NHOST hosts >= $MIN_STEP (scan result '${s}'); would risk a fresh start."
    return 1
  fi
  for f in "$CTL/STOP_AFTER_SAVE" "$CTL/SAVE_NOW"; do
    [ -e "$f" ] && { note "clearing stale $(basename "$f") before relaunch"; [ "$DRY" = 1 ] || mv "$f" "$f.wd-cleared-$(date +%s)"; }
  done
  local cmd="PPM_ENV=$ENVF sbatch -N 16 -w $NODEF $SB"
  if [ "$DRY" = 1 ]; then note "would relaunch ($why) from dcp-$s on all $NHOST hosts: $cmd"; return 0; fi
  local out jid
  out=$(env $UNSET PPM_ENV=$ENVF sbatch -N 16 -w "$NODEF" "$SB" 2>&1); jid=$(echo "$out" | grep -oE '[0-9]+$')
  if [ -z "$jid" ]; then note "ALERT: sbatch FAILED ($why): $out"; return 1; fi
  echo "$(date +%s) $jid $why" >> "$ST/relaunches"
  [ -n "$at" ] && echo "$(date +%s) $at $jid" >> "$ST/relaunch_at"
  echo "$jid" > "$JOBF"; rm -f "$ST/hang_stop_at" "$ST/wd_stop"
  note "RELAUNCHED ($why): new job $jid, resumes (auto) from dcp-$s COMPLETE on all $NHOST hosts; relaunch $((n + 1))/$MAX_RL in ${RL_WIN_H} h"
  return 0
}

[ -s "$JOBF" ] || { echo "ABORT: $JOBF (the 8,1,1 job id) missing: write it at launch"; exit 1; }
note "armed (8,1,1): job $(cat "$JOBF"), poll ${POLL}s, hang ${HANG_MIN} min, cap $MAX_RL/${RL_WIN_H}h, target $TARGET, min step $MIN_STEP"
J=""; last_step=""; last_prog=$(date +%s)
while :; do
  sleep_s=$POLL
  NJ=$(cat "$JOBF" 2>/dev/null)
  # adopt a newer "ppm811" job someone else launched (they may not update CURRENT_JOB_811)
  ADOPT=$(squeue -h -u "$(id -un)" -n $JNAME -o %i 2>/dev/null | sort -n | tail -1)
  if [ "$NOADOPT" != 1 ] && [ -n "$ADOPT" ] && [ "$ADOPT" -gt "$NJ" ]; then note "adopting newer ppm811 job $ADOPT (was $NJ)"; NJ=$ADOPT; [ "$DRY" = 1 ] || echo "$NJ" > "$JOBF"; fi
  if [ "$NJ" != "$J" ]; then [ -n "$J" ] && rm -f "$ST/hang_stop_at" "$ST/capped"; J=$NJ; last_step=""; last_prog=$(date +%s); fi
  LOG=$LOGDIR/ppm-$J.log
  S=$(jstate "$J")
  if [ -f "$CTL/WATCHDOG811_OFF" ]; then last_prog=$(date +%s);   # 10-21 fix: time while disabled never counts as a hang
  elif [ -z "$S" ] || [[ "$S" =~ ^(COMPLETED|FAILED|CANCELLED|TIMEOUT|NODE_FAIL|PREEMPTED|OUT_OF_MEMORY|BOOT_FAIL|DEADLINE)$ ]]; then
    stp=$(newest_step "$LOG")
    if grep -aq "done $TARGET steps" "$LOG" 2>/dev/null && ! grep -aq "STOPPED at step" "$LOG"; then
      note "FINISHED: job $J reached step $TARGET cleanly. Watchdog exiting."; [ "$DRY" = 1 ] || touch "$ST/FINISHED"; exit 0
    elif grep -aq "STOPPED at step" "$LOG" 2>/dev/null && [ ! -f "$ST/wd_stop" ]; then
      [ -f "$ST/deliberate-$J" ] || { note "job $J stopped deliberately via STOP_AFTER_SAVE (not by the watchdog) at step ${stp:-?}: NOT relaunching; waiting for a new job"; [ "$DRY" = 1 ] || touch "$ST/deliberate-$J"; }
    else
      [ -f "$ST/handled-$J" ] || {
        note "job $J gone (state '${S:-not in squeue}', last step ${stp:-?}) without a clean finish"
        if relaunch "job $J died at step ${stp:-?}" ""; then [ "$DRY" = 1 ] || touch "$ST/handled-$J"; else sleep_s=300; fi
      }
    fi
  elif [ "$S" != RUNNING ]; then last_prog=$(date +%s); last_step=""   # 10-21 fix: PENDING / held / CONFIGURING time never counts as a hang (old: a held job got STOP_AFTER_SAVE right at its start)
  elif [ "$S" = RUNNING ]; then
    stp=$(newest_step "$LOG"); now=$(date +%s)
    if [ -n "$stp" ] && [ "$stp" != "$last_step" ]; then [ -n "$last_step" ] && rm -f "$ST/hang_stop_at"; last_step=$stp; last_prog=$now; fi
    idle=$(( (now - last_prog) / 60 ))
    if [ "$idle" -ge "$HANG_MIN" ]; then
      if [ ! -f "$ST/hang_stop_at" ]; then
        note "HANG suspected: job $J no new step for ${idle} min (last ${last_step:-none}); touching STOP_AFTER_SAVE"
        [ "$DRY" = 1 ] || { echo "watchdog $(pt)" > "$CTL/STOP_AFTER_SAVE"; touch "$ST/wd_stop"; echo "$now" > "$ST/hang_stop_at"; }
        [ "$DRY" = 1 ] && [ "$ONESHOT" = 1 ] && note "would wait ${HANG_MIN} min, then scancel + relaunch if still no progress"
      elif [ $(( (now - $(cat "$ST/hang_stop_at")) / 60 )) -ge "$HANG_MIN" ]; then
        s=$(scan_complete "$J")
        if [ -n "$s" ] && [ "$s" -ge "$MIN_STEP" ]; then
          note "HANG confirmed: no progress ${HANG_MIN} min after STOP_AFTER_SAVE; newest COMPLETE on all hosts dcp-$s; scancel $J"
          if [ "$(relaunches_recent)" -ge "$MAX_RL" ]; then note "ALERT: hang on $J but relaunch cap reached; NOT cancelling"; sleep_s=600;
          else
            [ "$DRY" = 1 ] || { scancel "$J"; for i in $(seq 60); do [ -z "$(jstate "$J")" ] && break; sleep 5; done; }
            relaunch "job $J hung at step ${last_step:-?}" "" && { [ "$DRY" = 1 ] || touch "$ST/handled-$J"; }
          fi
        else
          note "ALERT: hang on $J but no COMPLETE save on all hosts >= $MIN_STEP (scan '${s}'); NOT cancelling"; sleep_s=600
        fi
      fi
    fi
  fi
  [ "$ONESHOT" = 1 ] && exit 0
  sleep "$sleep_s"
done
