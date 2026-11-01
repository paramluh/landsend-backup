#!/bin/bash
# pose5 (10-27): = code-811/ppm811f_watchdog.sh with ONLY CODE (-> pose5/code-811-s2p, overridable) and the default state dir
# (-> control/ppm-v1/watchdog_pose5_state: no stale relaunch / FINISHED records of 27114..27653) changed. Set ENVF, NODEF, TARGET, MIN_STEP.
# 8,1,1 FRESH-DATA watchdog (10-21) = ppm811_watchdog.sh with prod_811_fresh.env, nodes16_s7.txt, target 17503, own state files, SIGUSR2
# stack dumps on a hang before acting, at most 1 relaunch per failure step (never loop blind). 8,1,1 STAGE watchdog (code-811) = ppm/code/ppm_watchdog.sh with the 8,1,1 code / env / sbatch / job name / target and its OWN state files
# (run base ppm-v1 shares the control dir with PPM: STOP_AFTER_SAVE / SAVE_NOW are the same files). Disable: touch control/ppm-v1/WATCHDOG811_OFF (shared with the 8,1,1 watchdog).
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
MD=${MD:-/data/home/amartyavarma/ppm/control/overnight_monitor_811f.md}
LOGDIR=${LOGDIR:-/data/home/amartyavarma/ppm/logs}
CODE=${CODE:-/data/home/amartyavarma/ppm/pose5/code-811-s2p}   # pose5: the s2p copy (its ppm811.sbatch runs code-811-s2p)
ENVF=${ENVF:-$CODE/prod_811_fresh.env}
NODEF=${NODEF:-$CODE/nodes16_s7.txt}   # 10-21 fresh: slinky-7 plays slinky-10 (PPM_HOST_ALIAS in the env)
SB=$CODE/ppm811.sbatch
JNAME=ppm811
CKPT=/scratch/amartyavarma/ppm/ckpt/ppm-v1
TARGET=${TARGET:-17503}                  # = prod_811_fresh.env PPM_MAX_STEPS (end of the fresh pose)
MIN_STEP=${MIN_STEP:-17012}          # never relaunch onto anything older than the 8,1,1 switch step (set MIN_STEP=<S0> if switched earlier)
# pose5 RESUME PIN (10-27): until a save NEWER than PIN_STEP is COMPLETE on all hosts under CKPT, a relaunch resumes from RESUME_PIN (an explicit
# dir outside CKPT, e.g. /scratch/amartyavarma/pose5-resume/dcp-17850 = per-host symlink) instead of PPM_RESUME_FROM=auto: auto =
# newest_common(CKPT) = MIN over nodes, and slinky-15 holds no ppm-v1 save -> -1 -> a FRESH START. Never older than PIN_STEP.
RESUME_PIN=${RESUME_PIN:-}; PIN_STEP=${PIN_STEP:-$MIN_STEP}
HANG_MIN=${HANG_MIN:-15}
MAX_RL=${MAX_RL:-3}; RL_WIN_H=${RL_WIN_H:-6}
POLL=${POLL:-60}
DRY=${DRY:-0}
ONESHOT=${ONESHOT:-0}                # test: one decision pass, then exit
NOADOPT=${NOADOPT:-0}                # test: do not adopt newer ppm jobs
SCAN_JOB=${SCAN_JOB:-}               # test: overlap this running job for the relaunch scan instead of a new tiny job
ST=${ST:-$CTL/watchdog_pose5_state}; mkdir -p "$ST"
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

# pose5 (audit 1 B1): the resume source for a relaunch, from ONE scan of CKPT + the chained-hold dir on every host:
#   "FROM <step> <dir>" = the newest step > PIN_STEP COMPLETE on all hosts in CKPT (dir CKPT/dcp-N) or, if keep-1 / space_check dropped its CKPT
#                         names, in the hold on all hosts (dir HOLDD/dcp-N; the hold keeps the data)
#   "FROM <PIN_STEP> <RESUME_PIN>" only when the scan answered on ALL hosts AND no save > PIN_STEP exists anywhere (CKPT or hold, any host)
#   "ALERT <why>"       = the scan did not answer on every host, or a save > PIN_STEP exists somewhere but no step is complete on all
#                         hosts in one place (mixed CKPT/hold: needs a human re-link) -> NO relaunch (never back to 17850 once a newer save exists)
HOLDD=${HOLDD:-/scratch/amartyavarma/ppm/ckpt-keep-pose5/hold}
resume_decision() {   # $1 = running job id to overlap, or "" (new tiny job)
  local out sc="for d in $CKPT/dcp-*; do n=\${d##*-}; [[ \$n =~ ^[0-9]+\$ ]] && [ -f \$d/COMPLETE ] && echo \$(hostname) L \$n; done; \
for d in $HOLDD/dcp-*; do n=\${d##*-}; [[ \$n =~ ^[0-9]+\$ ]] && [ -f \$d/COMPLETE ] && echo \$(hostname) H \$n; done; echo \$(hostname) END"
  if [ -n "$1" ]; then out=$(timeout 180 env $UNSET srun --jobid="$1" --overlap -N"$NHOST" -w "$HOSTS" --ntasks-per-node=1 -c1 --mem=1G --cpu-bind=none bash -c "$sc" 2>/dev/null)
  else out=$(timeout 600 env $UNSET srun -p all -N"$NHOST" -w "$HOSTS" --ntasks-per-node=1 -c1 --mem=1G -t 5 --immediate=300 -J ppm-wd-scan bash -c "$sc" 2>/dev/null); fi
  echo "$out" | python3 -c "
import sys
nh, pin, rp, ck, hd = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
L, H, E = {}, {}, set()
for ln in sys.stdin:
    p = ln.split()
    if len(p) == 2 and p[1] == 'END': E.add(p[0])
    elif len(p) == 3 and p[1] in 'LH' and p[2].isdigit(): (L if p[1] == 'L' else H).setdefault(p[0], set()).add(int(p[2]))
if len(E) != nh: print(f'ALERT scan answered on {len(E)}/{nh} hosts (failed / partial scan)'); sys.exit()
hosts = sorted(E); newer = sorted({s for d in (L, H) for v in d.values() for s in v if s > pin})
both = [s for s in newer if all(s in L.get(h, set()) | H.get(h, set()) for h in hosts)]
if both:
    b = both[-1]
    if all(b in L.get(h, set()) for h in hosts): print(f'FROM {b} {ck}/dcp-{b}')
    elif all(b in H.get(h, set()) for h in hosts): print(f'FROM {b} {hd}/dcp-{b}')
    else: print(f'ALERT dcp-{b} is complete on all hosts but split between CKPT and the hold: needs a human re-link')
    if b != newer[-1]: print(f'NOTE newer save(s) {newer[-1]} exist on some hosts only (partial)', file=sys.stderr)
elif newer: print(f'ALERT save(s) > {pin} exist ({newer[-3:]}) but none is complete on all hosts: NOT rewinding to {pin}')
elif rp: print(f'FROM {pin} {rp}')
else: print('ALERT no save > PIN_STEP and no RESUME_PIN')
" "$NHOST" "$PIN_STEP" "$RESUME_PIN" "$CKPT" "$HOLDD"
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
  if [ -n "$at" ] && [ -f "$ST/relaunch_at" ] && [ "$(awk -v a="$at" '$2 == a' "$ST/relaunch_at" | wc -l)" -ge 1 ]; then
    [ -f "$ST/samestep-$at" ] || note "ALERT: relaunch NOT done ($why): already relaunched twice after a failure at step $at (deterministic?); needs a human/agent."
    [ "$DRY" = 1 ] || touch "$ST/samestep-$at"; return 1
  fi
  local dec pin=""; dec=$(resume_decision "${2:-$SCAN_JOB}")       # pose5 (audit 1 B1): explicit resume dir, never a blind pin
  case "$dec" in
    FROM\ *) s=$(echo "$dec" | awk '{print $2}'); pin=$(echo "$dec" | awk '{print $3}');;
    *) note "ALERT: relaunch NOT done ($why): $dec"; [ "$DRY" = 1 ] || echo "$(pt) ppm811s2p watchdog: $why: $dec" >> /data/home/amartyavarma/setup-logs/watchdog/ALERTS.log; return 1;;
  esac
  if [ -z "$s" ] || [ "$s" -lt "$MIN_STEP" ]; then note "ALERT: relaunch NOT done ($why): resume step '$s' < $MIN_STEP"; return 1; fi
  for f in "$CTL/STOP_AFTER_SAVE" "$CTL/SAVE_NOW"; do
    [ -e "$f" ] && { note "clearing stale $(basename "$f") before relaunch"; [ "$DRY" = 1 ] || mv "$f" "$f.wd-cleared-$(date +%s)"; }
  done
  local cmd="PPM_ENV=$ENVF ${pin:+PPM_RESUME_FROM=$pin }sbatch -N 16 -w $NODEF $SB"
  if [ "$DRY" = 1 ]; then note "would relaunch ($why) from dcp-$s on all $NHOST hosts: $cmd"; return 0; fi
  local out jid
  out=$(env $UNSET PPM_ENV=$ENVF ${pin:+PPM_RESUME_FROM=$pin} sbatch -N 16 -w "$NODEF" "$SB" 2>&1); jid=$(echo "$out" | grep -oE '[0-9]+$')
  if [ -z "$jid" ]; then note "ALERT: sbatch FAILED ($why): $out"; return 1; fi
  echo "$(date +%s) $jid $why" >> "$ST/relaunches"
  [ -n "$at" ] && echo "$(date +%s) $at $jid" >> "$ST/relaunch_at"
  echo "$jid" > "$JOBF"; rm -f "${ST:?}/hang_stop_at" "${ST:?}/wd_stop"
  note "RELAUNCHED ($why): new job $jid, resumes from $pin (dcp-$s); relaunch $((n + 1))/$MAX_RL in ${RL_WIN_H} h"
  return 0
}

[ -s "$JOBF" ] || { echo "ABORT: $JOBF (the 8,1,1 job id) missing: write it at launch"; exit 1; }
[[ "$(cat "$JOBF")" =~ ^[0-9]+$ ]] || { echo "ABORT: $JOBF holds '$(head -c 40 "$JOBF")', not a numeric job id (audit 2 R2-6: fail closed, no relaunch)"; exit 1; }
note "armed (8,1,1): job $(cat "$JOBF"), poll ${POLL}s, hang ${HANG_MIN} min, cap $MAX_RL/${RL_WIN_H}h, target $TARGET, min step $MIN_STEP"
J=""; last_step=""; last_prog=$(date +%s)
while :; do
  sleep_s=$POLL
  NJ=$(cat "$JOBF" 2>/dev/null)
  if ! [[ "$NJ" =~ ^[0-9]+$ ]]; then                    # audit 2 R2-6: a non-numeric job file never means "job gone" (no relaunch); keep the last good id
    note "ALERT: $JOBF holds '$(printf %s "$NJ" | head -c 40)', not a numeric job id: ignored (still watching ${J:-nothing})"
    if [ "$DRY" != 1 ] && { [ ! -f "$ST/alert-jobf" ] || [ $(( $(date +%s) - $(stat -c %Y "$ST/alert-jobf") )) -ge 3600 ]; }; then
      echo "$(pt) ppm811s2p watchdog: $JOBF not a numeric job id" >> /data/home/amartyavarma/setup-logs/watchdog/ALERTS.log; touch "$ST/alert-jobf"; fi
    [ -n "$J" ] && NJ=$J || { sleep $POLL; continue; }
  fi
  # pose5 (audit 1 #6): the chained hold is the condition of KEEP=1 -> alert (once per hour) when its job is not RUNNING
  HCJ=$(grep -oE '^HC=[0-9]+' /data/home/amartyavarma/ppm/pose5/kit/state/LAUNCHED_RUN 2>/dev/null | cut -d= -f2)
  if [ -n "$HCJ" ] && [ ! -f /data/home/amartyavarma/ppm/pose5/kit/state/hold_chain/CHAIN_DONE ] && ! squeue -h -j "$HCJ" -t R 2>/dev/null | grep -q .; then
    if [ ! -f "$ST/alert-hc" ] || [ $(( $(date +%s) - $(stat -c %Y "$ST/alert-hc") )) -ge 3600 ]; then
      note "ALERT: pose5 chained-hold job $HCJ is NOT RUNNING (KEEP=1 runs without its safety net); restart: TARGET=$TARGET PIN_STEP=$PIN_STEP sbatch /data/home/amartyavarma/ppm/pose5/kit/hold_chain.sbatch"
      [ "$DRY" = 1 ] || { echo "$(pt) ppm811s2p watchdog: chained-hold job $HCJ NOT RUNNING" >> /data/home/amartyavarma/setup-logs/watchdog/ALERTS.log; touch "$ST/alert-hc"; }
    fi
  fi
  # adopt a newer "ppm811" job someone else launched (they may not update CURRENT_JOB_811)
  ADOPT=$(squeue -h -u "$(id -un)" -n $JNAME -o %i 2>/dev/null | sort -n | tail -1)
  if [ "$NOADOPT" != 1 ] && [ -n "$ADOPT" ] && [ "$ADOPT" -gt "$NJ" ]; then note "adopting newer ppm811 job $ADOPT (was $NJ)"; NJ=$ADOPT; [ "$DRY" = 1 ] || echo "$NJ" > "$JOBF"; fi
  if [ "$NJ" != "$J" ]; then [ -n "$J" ] && rm -f "${ST:?}/hang_stop_at" "${ST:?}/capped"; J=$NJ; last_step=""; last_prog=$(date +%s); fi
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
    if [ -n "$stp" ] && [ "$stp" != "$last_step" ]; then [ -n "$last_step" ] && rm -f "${ST:?}/hang_stop_at"; last_step=$stp; last_prog=$now; fi
    idle=$(( (now - last_prog) / 60 ))
    if [ "$idle" -ge "$HANG_MIN" ]; then
      if [ ! -f "$ST/hang_stop_at" ]; then
        note "HANG suspected: job $J no new step for ${idle} min (last ${last_step:-none}); stack dumps (SIGUSR2 -> faulthandler, all ranks, into the job log) + NCCL flight recorder, then STOP_AFTER_SAVE"
        [ "$DRY" = 1 ] || { timeout 120 env $UNSET srun --jobid="$J" --overlap -N"$NHOST" -w "$HOSTS" --ntasks-per-node=1 -c1 --mem=1G --cpu-bind=none \
            bash -c 'pkill -USR2 -f "[p]ython.*[t]rain_ppm.py"; echo $(hostname -s) usr2 sent' >> "$ST/stackdump-$J.log" 2>&1
            touch "$LOGDIR/DEBUG_DUMP" 2>/dev/null; sleep 20; cp -f "$LOG" "$ST/hanglog-$J-$(date +%s).log" 2>/dev/null; }
        [ "$DRY" = 1 ] || { echo "watchdog $(pt)" > "$CTL/STOP_AFTER_SAVE"; touch "$ST/wd_stop"; echo "$now" > "$ST/hang_stop_at"; }
        [ "$DRY" = 1 ] && [ "$ONESHOT" = 1 ] && note "would wait ${HANG_MIN} min, then scancel + relaunch if still no progress"
      elif [ $(( (now - $(cat "$ST/hang_stop_at")) / 60 )) -ge "$HANG_MIN" ]; then
        dec=$(resume_decision "$J"); s=$(echo "$dec" | awk '$1=="FROM"{print $2}')
        if [ -n "$s" ] && [ "$s" -ge "$MIN_STEP" ]; then
          note "HANG confirmed: no progress ${HANG_MIN} min after STOP_AFTER_SAVE; resume source $dec; scancel $J"
          if [ "$(relaunches_recent)" -ge "$MAX_RL" ]; then note "ALERT: hang on $J but relaunch cap reached; NOT cancelling"; sleep_s=600;
          else
            [ "$DRY" = 1 ] || { scancel "$J"; for i in $(seq 60); do [ -z "$(jstate "$J")" ] && break; sleep 5; done; }
            relaunch "job $J hung at step ${last_step:-?}" "" && { [ "$DRY" = 1 ] || touch "$ST/handled-$J"; }
          fi
        else
          note "ALERT: hang on $J but no usable resume source ($dec); NOT cancelling"; sleep_s=600
        fi
      fi
    fi
  fi
  [ "$ONESHOT" = 1 ] && exit 0
  sleep "$sleep_s"
done
