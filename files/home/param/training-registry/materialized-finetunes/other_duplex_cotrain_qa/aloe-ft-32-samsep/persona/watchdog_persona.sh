#!/bin/bash
# aloe-ft-32-samsep stall watchdog (minimal). Run as a tiny CPU job:  sbatch -w slinky-18 -c 1 --mem=1G --gres=none -t 12:00:00 \
#   -J aloesamsep-wd -o /data/home/amartyavarma/ppm/aloe-ft-32-samsep/logs/watchdog-%j.log /data/home/amartyavarma/ppm/aloe-ft-32-samsep/watchdog.sh <train job>
# Acts ONLY on the given job, and only if its name is aloepersona. A stall = no new progress line (step / EVAL / export / calib / init) in the
# run log for 15 min while the job is RUNNING (after a 25 min startup grace). On a stall: (1) py-spy stacks of every trainer process on both
# hosts (overlap step, no CPU binding) -> logs/stacks-<job>-<time>.txt (NCCL flight-recorder dumps also land in
# /scratch/amartyavarma/aloe-ft-32-samsep/nccl_trace_r* on a NCCL timeout); (2) touch control/STOP_AFTER_SAVE (save-and-exit at the next step
# boundary); (3) ALERT file control/ALERT-<job>; (4) 20 min later, still running with no progress -> scancel that job. No automatic relaunch:
# STOP-only saves rarely fit /scratch at world 14 (see recipe/STATUS.md), so the relaunch (resume or fresh) is the lead's call.
set -u
Q=/data/home/amartyavarma/ppm/aloe-ft-32-samsep; J=$1; LOG=$Q/persona/logs/aloepersona-$J.log; CD=$Q/persona/control
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
PAT='\] (step [0-9]+/|EVAL step|\[export|\[calib\]|\[init\]|\[pack\]|\[offload\]|\[resume)'
[ "$(squeue -h -j $J -o %j 2>/dev/null)" = aloepersona ] || { echo "[$(pt)] job $J is not an aloesamsep job in squeue: exit"; exit 0; }
echo "[$(pt)] watching job $J ($LOG)"
stopped=0
while true; do
  st=$(squeue -h -j $J -o %T 2>/dev/null)
  [ -z "$st" ] && { echo "[$(pt)] job $J left squeue: exit"; exit 0; }
  if [ "$st" = RUNNING ] && [ -f $LOG ]; then
    t0=$(date -d "$(squeue -h -j $J -o %S)" +%s); now=$(date +%s)
    last=$(stat -c %Y $LOG); lp=$(grep -E "$PAT" $LOG | tail -1)
    # progress time = the log's mtime only when the last progress line is also the log's newest content of that kind
    pm=$(grep -nE "$PAT" $LOG | tail -1 | cut -d: -f1); [ -n "$pm" ] || pm=0
    [ "${pm_prev:-x}" != "$pm" ] && { pm_prev=$pm; prog_t=$now; }
    if [ $((now - t0)) -gt 1500 ] && [ $((now - ${prog_t:-$now})) -gt 900 ]; then
      if [ $stopped = 0 ]; then
        echo "[$(pt)] STALL: no progress line for $(( (now - prog_t) / 60 )) min; last: ${lp:0:160}"
        srun --jobid=$J --overlap -N2 --ntasks-per-node=1 -c 1 --cpu-bind=none --mem=2G bash -c \
          'for p in $(pgrep -u $USER -f train_samsep.py); do echo "== $(hostname -s) pid $p"; timeout 60 /data/home/amartyavarma/venvs/qwen-s2/bin/py-spy dump --pid $p 2>&1 | head -80; done' \
          > $Q/logs/stacks-$J-$(date +%H%M%S).txt 2>&1
        touch $CD/STOP_AFTER_SAVE
        echo "[$(pt)] job $J stalled (no progress 15 min); stacks in logs/stacks-$J-*; STOP_AFTER_SAVE touched" > $CD/ALERT-$J
        stopped=$now
      elif [ $((now - stopped)) -gt 1200 ]; then
        [ "$(squeue -h -j $J -o %j)" = aloepersona ] && { echo "[$(pt)] still stalled 20 min after STOP_AFTER_SAVE: scancel $J"; scancel $J; }
        echo "[$(pt)] job $J cancelled by the watchdog after a stall (see logs/stacks-$J-*)" >> $CD/ALERT-$J
        exit 1
      fi
    fi
  fi
  sleep 60
done
