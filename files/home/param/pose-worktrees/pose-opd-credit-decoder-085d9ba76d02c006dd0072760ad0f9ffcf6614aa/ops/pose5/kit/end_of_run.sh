#!/bin/bash
# pose5 END OF RUN / HAND-BACK (user 06:16 PT item 4). Waits for the pose run's end, checks the final save is HF-verified 16/16, stops the pose5
# companions and prints (never runs) motguard's MoT resume command + the pose-run files left on the hosts (removed only after HF verify and only
# if MoT needs the space: main's call; this script deletes NOTHING).
# usage: bash end_of_run.sh [--dry-run] [--no-wait]     (switch_to_pose.sh starts it detached after the launch; idempotent)
set -u
DRY=0; WAIT=1; for a in "$@"; do case $a in --dry-run) DRY=1;; --no-wait) WAIT=0;; esac; done
P5=/data/home/amartyavarma/ppm/pose5; K=$P5/kit; CTL=/data/home/amartyavarma/ppm/control/ppm-v1; HFB=/data/home/amartyavarma/setup-logs/hfb
HOSTS=$(cat $P5/nodes16_s7_s15.txt); LS=$K/state/LAUNCHED_RUN
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
say() { echo "[$(pt)] $*"; }
eval "$(sed -n '/^verified() {/,/^}/p' $K/hold_chain.sh)"
[ -f $LS ] || { [ $DRY = 1 ] && { say "(dry) no $LS yet: showing the steps with placeholders"; J=JOB; M=MAX; } || { say "ABORT: $LS missing"; exit 1; }; }
[ -f $LS ] && . $LS                                      # J (first job), M (max steps), WD, EL, HC, PG (job ids) written by switch_to_pose.sh
J=$(cat $CTL/CURRENT_JOB_811 2>/dev/null || echo ${J:-JOB})
say "end-of-run watch: run $J (CURRENT_JOB_811), target $M"
if [ $WAIT = 1 ] && [ $DRY = 0 ]; then
  while :; do
    J=$(cat $CTL/CURRENT_JOB_811); L=/data/home/amartyavarma/ppm/logs/ppm-$J.log
    if grep -aq "done $M steps" $L 2>/dev/null && ! grep -aq "STOPPED at step" $L && ! squeue -h -j $J | grep -q .; then
      verified $M && break
      say "final save dcp-$M not HF-verified 16/16 yet (hfb); waiting"
    fi
    sleep 120
  done
fi
say "1. final save dcp-$M: $( [ $DRY = 1 ] && echo '(dry)' || echo 'COMPLETE + HF-verified 16/16 (setup-logs/hfb/done + verify records)')"
say "2. stop companions"
for x in "touch $P5/EVAL_LOG_STOP" "scancel ${PG:-<page refresh job>}   # klweb-ppm811-refresh (CPU, spare)" \
         "# watchdog ${WD:-?} exits by itself on 'done $M steps' (state FINISHED); hold chain ${HC:-?} exits once dcp-$M is HF-verified" \
         "${B_ROLL_STOP:-# B: npl5 roll daemons stop at END by themselves (npl5_roll.py daemon --end $M)}"; do
  if [ $DRY = 1 ] || [[ $x == \#* ]]; then echo "    $x"; else eval "$x" && echo "    done: $x"; fi
done
say "3. the 16 hosts: pose5 jobs left in the queue on them:"
squeue -h -u amartyavarma -w $HOSTS -o '%i %j %N' | sed 's/^/    /' || true
cat <<EOF
3b. HF watcher back to FLOOR_GB=30 (main 10-27; only AFTER dcp-$M is HF-verified and nothing is uploading; main runs it):
    scancel \$(squeue -h -u amartyavarma -n hfb-watch -o %i); sleep 5; FLOOR_GB=30 sbatch -w slinky-10 /data/home/amartyavarma/setup-logs/hfb/hfb_watch.sbatch
4. MoT resume (motguard owns it; run ONLY on main's GO, after the hosts are free):
    setsid nohup /data/home/amartyavarma/ppm/mot/guard/stop/mot_resume_pause.sh > /data/home/amartyavarma/ppm/mot/guard/stop/resume-\$(date +%m%d-%H%M).log 2>&1 &
5. pose-run files LEFT on the 16 hosts (NOT deleted; main decides, only if MoT needs the space; all HF-backed once dcp-$M is verified):
    /scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-$M                      final save (46.1 GB/host; HF ppm-v1-*)
    /scratch/amartyavarma/ppm/ckpt-keep-pose5/hold/dcp-*              chained holds still present ($(ls $K/state/hold_chain/held-* 2>/dev/null | grep -oE '[0-9]+$' | paste -sd' ') ; 46.1 GB each)
    /scratch/amartyavarma/ppm/ckpt-keep-pose5/dcp-17850               hard-link pins of node-local dcp-17850 (s1/2/7/14; data = the original copy)
    /scratch/amartyavarma/pose5-resume/dcp-17850                      symlink only
    /scratch/amartyavarma/npl-append/npl5 + ppm-fresh-pose-npl5       npl5 roll window + plan (B; HF bucket delivery 5)
    slinky-15: /scratch/amartyavarma/pose-s3-samdata/rows (9.2 GB slot-9 pose store), npl3 rows + slot-9 plans (B, 10.3 GB)
    Weka: /data/home/amartyavarma/ppm/pose5/dcp17850-weka (507 GB of dcp-17850 copies; HF ppm-v1-c holds the original) -> /data floor
EOF
[ $DRY = 1 ] && say "dry run: nothing stopped, nothing deleted" || { say "HAND-BACK READY: 16 hosts free for the MoT resume (main's GO)"; echo "$(pt)" > $K/state/HANDBACK_READY; }
