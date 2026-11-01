#!/bin/bash
# MONOFT: pause the MAIN 8,1,1 run (ppm811, job in control/ppm-v1/CURRENT_JOB_811) with STOP_AFTER_SAVE (never scancel) and print S = the step of
# its stop save dcp-S (= the FT init and the main run's resume point). For the MAIN session, on the user's go; run it in the background:
#
#   bash /data/home/amartyavarma/ppm/monoft/pause_main.sh [save|now|eval] [--launch <8 FT hosts|default> [final save 0|1]]
#
# Mechanism (code-811/train_ppm.py): at every step boundary rank 0 polls control/ppm-v1/STOP_AFTER_SAVE; when present the trainer saves
# dcp-<step> node-local on all 16 hosts (~45 s; keep-last-2 prunes the oldest), logs "[control] STOP_AFTER_SAVE at step N: saved ...", exits
# cleanly; the watchdog sees a deliberate stop and does NOT relaunch. If /scratch is short the save is SKIPPED and the run continues (this
# script then reports it and exits 1). Modes:
#  save (default; "pause right after an hourly save"): touch it in the step whose boundary is the hourly save timer's (PPM_SAVE_EVERY_MIN 60),
#       so the stop save IS the hourly save: no extra 43 GB/host save, no extra HF backup. Waits <= ~60 min for that boundary.
#  now: touch it at once (S = the next step boundary, ~1 min).
#  eval: touch it in the step whose boundary is the 30-min eval timer's, so the main run logs "EVAL step S" right before the stop save: the
#       samdata_ft scaling page needs the pretraining eval at EXACTLY the init step (else it waits for the eval at the main run's resume).
# PAUSE_DRY=1 (save / eval modes): find the window and report it, touch nothing.
# Writes ppm/monoft/PAUSED.txt ("<S> <job> <PT time>"). --launch: then runs launch.sh S <hosts> [final save].
set -uo pipefail
MODE=${1:-save}; shift || true
LAUNCH=""; if [ "${1:-}" = --launch ]; then LAUNCH="${2:-default} ${3:-0}"; fi
CTL=/data/home/amartyavarma/ppm/control/ppm-v1; M=/data/home/amartyavarma/ppm/monoft; LOG=$M/logs/pause.log; mkdir -p $M/logs
J=$(cat $CTL/CURRENT_JOB_811); L=/data/home/amartyavarma/ppm/logs/ppm-$J.log
pt() { TZ=America/Los_Angeles date '+%m-%d %T PT'; }
say() { echo "[$(pt)] $*" | tee -a $LOG; }
lt() { sed -nE "s/^\[([0-9]{2})-([0-9]{2}) ([0-9:]{8}) PT\].*/$(date +%Y)-\1-\2 \3/p" | TZ=America/Los_Angeles xargs -I{} date -d {} +%s; }   # log line -> epoch
case $MODE in save|now|eval) ;; *) say "usage: pause_main.sh [save|now|eval] [--launch <hosts|default> [0|1]]"; exit 2;; esac
squeue -h -j $J -t R -o %i | grep -q . || { say "ABORT: main job $J is not RUNNING"; exit 1; }
[ -e $CTL/STOP_AFTER_SAVE ] && { say "ABORT: $CTL/STOP_AFTER_SAVE already exists (someone else is stopping the run)"; exit 1; }
say "pause main job $J, mode $MODE"
if [ $MODE != now ]; then
  if [ $MODE = save ]; then PAT='\] \[save\] '; P=3600; else PAT='\] EVAL step '; P=1800; fi
  for i in $(seq 1 1200); do                                     # <= ~100 min of 5 s polls (2 timer periods)
    tref=$(grep -a "$PAT" $L | tail -1 | lt); last=$(grep -aE '\] step [0-9]+/[0-9]+ ' $L | tail -1)
    T=$(echo "$last" | lt); exp=$((tref + P))
    if [ -n "$T" ] && [ "$T" -lt $((exp - 3)) ] && [ "$T" -gt $((exp - 60)) ] && [ $(( $(date +%s) - T )) -lt 20 ]; then
      say "window: last step line $(echo "$last" | cut -c1-40) is the last before the $MODE timer ($(TZ=America/Los_Angeles date -d @$exp +%T) PT): stopping at the next boundary"
      [ "${PAUSE_DRY:-0}" = 1 ] && { say "PAUSE_DRY=1: would touch STOP_AFTER_SAVE now (nothing touched)"; exit 0; }
      break
    fi
    squeue -h -j $J -t R -o %i | grep -q . || { say "ABORT: main job $J left the queue while waiting"; exit 1; }
    sleep 5
  done
fi
echo "monoft pause_main $MODE $(pt)" > $CTL/STOP_AFTER_SAVE
say "touched $CTL/STOP_AFTER_SAVE"
S=""
for i in $(seq 1 180); do                                         # <= 15 min for the save + clean exit
  S=$(grep -aoE 'STOP_AFTER_SAVE at step [0-9]+: saved' $L | tail -1 | grep -oE '[0-9]+')
  grep -aq "STOP_AFTER_SAVE at step .*: the save was SKIPPED" $L && { say "ABORT: the stop save was SKIPPED (no /scratch space); the run CONTINUES; control file kept by the trainer ($(grep -a 'was SKIPPED' $L | tail -1 | cut -c1-200))"; exit 1; }
  [ -n "$S" ] && ! squeue -h -j $J -o %i | grep -q . && break
  sleep 5
done
[ -n "$S" ] || { say "ABORT: no 'STOP_AFTER_SAVE at step N: saved' line in $L after 15 min"; exit 1; }
squeue -h -j $J -o %i | grep -q . && { say "WARNING: dcp-$S saved but job $J is still in the queue"; }
grep -a "EVAL step $S:" $L > /dev/null && ev="EVAL at step $S logged" || ev="no EVAL at step $S (scaling page x comes from the resume eval)"
say "PAUSED main job $J at step $S: dcp-$S saved node-local ($ev)"
echo "$S $J $(pt)" > $M/PAUSED.txt
echo "$S"
[ -n "$LAUNCH" ] && exec bash $M/launch.sh $S $LAUNCH
exit 0
