#!/bin/bash
#SBATCH -J pose5-stage -p all --ntasks-per-node=1 -c 32 --mem=64G --gres=none -t 02:00:00 -o /data/home/amartyavarma/ppm/pose5/kit/logs/stage-%j.log
# pose5 launch kit steps (b) + (c) on the 16 hosts, CPU only, after MoT has stopped: one stage_host.sh per host (gate, dcp-17850, B's npl5 hook,
# slot-9 extras on slinky-15, trainer-side checks). Exit 0 only if all hosts print OK; ppm811 is queued behind it with --dependency=afterok.
# usage: sbatch -N 16 -w <hosts file> pose5_stage.sbatch [--dry-run]
set -u
K=/data/home/amartyavarma/ppm/pose5/kit
echo "== pose5-stage $SLURM_JOB_ID on $SLURM_JOB_NODELIST $(TZ=America/Los_Angeles date '+%F %T PT') args ${*:-}"
srun --kill-on-bad-exit=0 -N $SLURM_NNODES -n $SLURM_NNODES --ntasks-per-node=1 -c 32 --mem=64G --cpu-bind=none bash $K/stage_host.sh "$@" \
  > $K/logs/stage-$SLURM_JOB_ID.hosts.log 2>&1
cat $K/logs/stage-$SLURM_JOB_ID.hosts.log
n=$(grep -cE '^STAGEHOST \S+ \S+ (OK|DRY) ' $K/logs/stage-$SLURM_JOB_ID.hosts.log)
echo "== $n/$SLURM_NNODES hosts OK $(TZ=America/Los_Angeles date '+%T PT')"
[ "$n" -eq "$SLURM_NNODES" ] && [ "$SLURM_NNODES" -eq 16 ] || exit 1
