#!/bin/bash
#SBATCH -J npl5-rehearse -p all -w slinky-4 -x slinky-9 -N1 -n1 -c 2 --mem=12G --gres=none -t 0:30:00 --time-min=0:15:00 --deadline=2026-10-27T17:50:00
#SBATCH -o /data/home/amartyavarma/ppm/pose5/npl5/logs/rehearse-%j.out
# LAUNCH-DAY REHEARSAL of B's per-host steps exactly as C's kit/stage_host.sh calls them, on the spare slinky-4 acting as slinky-15 (slot 9):
#   1 bin/leg1_slot9_hook.sh (HOOK_AS=slinky-15): slot-9 plans 1-3 + npl3 window, digests == dcp-17850, 16 micros
#   2 bin/npl5_prep.sh (PREP_AS=slinky-15): npl5 plan (sha), first window from HF, fresh_pose check rows 0-2, host dry run (17850 + first npl5 micros)
# Writes the canonical /scratch/amartyavarma/{ppm-fresh-pose*,npl-append} on slinky-4, removed at the end. -> TIMES.tsv 'rehearsal'
set -uo pipefail
W=/data/home/amartyavarma/ppm/pose5/npl5; S=/scratch/amartyavarma
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
for d in $S/ppm-fresh-pose $S/ppm-fresh-pose-npl2 $S/ppm-fresh-pose-npl3 $S/ppm-fresh-pose-npl5 $S/npl-append; do
  [ -e $d ] && { echo "ABORT: $d exists on $(hostname -s)"; exit 1; }; done
t0=$(date +%s)
HOOK_AS=slinky-15 nice -n19 bash $W/bin/leg1_slot9_hook.sh; r1=$?
t1=$(date +%s)
PREP_AS=slinky-15 nice -n19 bash $W/bin/npl5_prep.sh; r2=$?
t2=$(date +%s)
printf "rehearsal\tslot9@slinky-4\thook %ss rc %s; prep %ss rc %s\t-\t%s\n" $((t1 - t0)) $r1 $((t2 - t1)) $r2 "$(pt)" >> $W/TIMES.tsv
echo "== $(pt) rehearsal: hook rc $r1 ($((t1 - t0)) s), prep rc $r2 ($((t2 - t1)) s)"
mkdir -p $W/launch/rehearsal-$SLURM_JOB_ID; mv $W/leg1/slot9/results/slinky-15.* $W/leg1/slot9/results/missing-slinky-15.tsv $W/launch/prep/slinky-15* $W/launch/rehearsal-$SLURM_JOB_ID/ 2>/dev/null
mv $W/leg1/slot9/logs/hook-slinky-15.log $W/launch/rehearsal-$SLURM_JOB_ID/ 2>/dev/null; rm -rf "${W:?}/leg1/slot9/verify-run-slinky-15"
rm -rf "${S:?}/ppm-fresh-pose" "${S:?}/ppm-fresh-pose-npl2" "${S:?}/ppm-fresh-pose-npl3" "${S:?}/ppm-fresh-pose-npl5" "${S:?}/npl-append/npl3" "${S:?}/npl-append/npl5"
rmdir "${S:?}/npl-append" 2>/dev/null
echo "== $(pt) canonical dirs removed from $(hostname -s); records in $W/launch/rehearsal-$SLURM_JOB_ID"
[ $r1 = 0 ] && [ $r2 = 0 ]
