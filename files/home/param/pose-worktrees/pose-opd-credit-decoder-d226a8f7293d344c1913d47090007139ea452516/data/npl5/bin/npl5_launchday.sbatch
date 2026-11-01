#!/bin/bash
#SBATCH -J npl5-copy -p all -N16 -n16 --ntasks-per-node=1 -c 48 --mem=180G --gres=none -x slinky-9 -t 1:30:00
#SBATCH -o /data/home/amartyavarma/ppm/pose5/npl5/logs/launchday-%j.out
# npl5 LAUNCH-DAY COPY (NOT run during prep). Run ONLY after MoT 30289 has stopped WITH its save, before the 8,1,1 launch:
#   sbatch -w $(cat /data/home/amartyavarma/ppm/pose5/nodes16_s7_s15.txt) /data/home/amartyavarma/ppm/pose5/npl5/bin/npl5_launchday.sbatch
# Needs: W/state/dryrun.ok (staged plans + off-host dry run passed). Every host in parallel (bin/npl5_host.sh): plan dirs, npl5 parquet from HF
# (HF_RATE 90/host x 16 = 1.44k/s < the hub's ~1.6k/s resolver quota), slot-9 extras on slinky-15, npl3 remainder, verify, check, host dry run.
# Knobs (sbatch --export=ALL,...): HF_RATE (90), THREADS_HF (96), WITH_JSON=1 (also release.json; not read by any launch step), GATE_GB (122),
# FORCE_DISK=1 (skip the free-space precheck). Re-running is safe: present files are skipped, plan dirs are compared, not overwritten.
set -uo pipefail
echo "SUPERSEDED (10-27 audit #15): the launch-day path is C's kit/stage_host.sh -> B's bin/leg1_slot9_hook.sh + bin/npl5_prep.sh, and the roll supervisor bin/npl5_roll_super.sbatch. This full-share copy no longer fits the hosts." >&2; exit 1
W=/data/home/amartyavarma/ppm/pose5/npl5; L=$W/launch/run-$SLURM_JOB_ID; export L
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
mkdir -p $L/logs
echo "== $(pt) npl5 launch-day copy job $SLURM_JOB_ID on $SLURM_JOB_NODELIST -> $L"
[ -f $W/state/dryrun.ok ] || { echo "ABORT: $W/state/dryrun.ok missing (off-host plans / dry run not done)"; exit 1; }
[[ ",$(scontrol show hostnames $SLURM_JOB_NODELIST | paste -sd,)," == *",slinky-9,"* ]] && { echo "ABORT: slinky-9 in the allocation"; exit 1; }
[ "$(scontrol show hostnames $SLURM_JOB_NODELIST | wc -l)" = 16 ] || { echo "ABORT: need the 16 hosts"; exit 1; }
O=$(squeue -h -w $SLURM_JOB_NODELIST -t R -o "%i %j %b" | grep -v "^$SLURM_JOB_ID " | grep -iE "gpu|mot")   # (CPU-only side jobs, e.g. a ckpt restore, may run)
[ -z "$O" ] || { echo "ABORT: GPU / MoT jobs still RUNNING on the hosts (MoT must be stopped with its save first): $O"; exit 1; }
t0=$(date +%s)
srun -N16 -n16 --ntasks-per-node=1 -c 48 --cpu-bind=none --kill-on-bad-exit=0 --export=ALL bash $W/bin/npl5_host.sh
ok=$(ls $L/OK-* 2>/dev/null | wc -l); el=$(( $(date +%s) - t0 ))
printf "launchday_all\t-\t%ss\t%s\t%s\n" $el "$ok/16 ok" "$(pt)" >> $W/TIMES.tsv
if [ "$ok" = 16 ]; then
  echo "== $(pt) READY: npl5 on all 16 hosts in $el s ($(cat $L/OK-* | awk '{print $5,$6,$7}' | paste -sd';' | cut -c1-600)); next: the 8,1,1 launch (pose5 launch kit)"
  echo "$(pt) job $SLURM_JOB_ID $el s" > $W/state/launchday.ok
else
  echo "== $(pt) NOT READY: $ok/16 hosts ok; failures: $(cat $L/FAIL-* 2>/dev/null | paste -sd'|')"; exit 1
fi
