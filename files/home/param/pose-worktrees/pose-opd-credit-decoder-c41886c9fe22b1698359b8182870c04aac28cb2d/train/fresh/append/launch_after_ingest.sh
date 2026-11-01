#!/bin/bash
# 10-22 (user: "we want to now continue train on this data" = delivery2): after the ingest job 27122 succeeds, launch the append run per APPEND_READY.md §4
set -u
C=/data/home/amartyavarma/ppm/code-811; CTL=/data/home/amartyavarma/ppm/control/ppm-v1; L27=/data/home/amartyavarma/ppm/logs/ppm-27114.log
say(){ echo "[$(TZ=America/Los_Angeles date '+%F %T') PT] $*"; }
grep -aq "done 17352 steps" $L27 || { say "ABORT: 27114 did not finish at 17352"; exit 1; }
grep -aq "STOPPED at step" $L27 && { say "ABORT: 27114 stopped early"; exit 1; }
for j in 27114 27116; do s=$(squeue -h -j $j -o %T 2>/dev/null); [ -n "$s" ] && { say "job $j still $s -> scancel (watchdog must not adopt the new job)"; [ $j = 27116 ] && scancel 27116; }; done
grep -q '"guard": *"ok"' $C/fresh/append/npl2/dryrun/GUARD.json || { say "ABORT: GUARD not ok"; cat $C/fresh/append/npl2/dryrun/GUARD.json; exit 1; }
grep -q '"resume_is_final_save": *true' $C/fresh/append/npl2/dryrun/RESULT.json || { say "ABORT: dry run not against the final save"; exit 1; }
M=$(grep '^PPM_MAX_STEPS=' $C/prod_811_fresh_append.env | tail -1 | cut -d= -f2); [[ "$M" =~ ^[0-9]+$ ]] || { say "ABORT: PPM_MAX_STEPS=$M"; exit 1; }
for h in $(tr , ' ' < $C/nodes16_s7.txt); do :; done
J=$(PPM_ENV=$C/prod_811_fresh_append.env PPM_RESUME_FROM=/scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-17352 sbatch --parsable -N 16 -w $(cat $C/nodes16_s7.txt) $C/ppm811.sbatch); say "TRAIN job $J max_steps $M"
echo $J > $CTL/CURRENT_JOB_811
W=$(ENVF=$C/prod_811_fresh_append.env TARGET=$M MIN_STEP=17352 sbatch --parsable $C/ppm811f_watchdog.sbatch); say "WATCHDOG $W"
E=$(MIN_JOB=$J END=$M sbatch --parsable $C/fresh/append/eval_log_append.sbatch); say "EVALLOG $E"
echo "$J $W $E $M" > $C/fresh/append/LAUNCHED.txt
