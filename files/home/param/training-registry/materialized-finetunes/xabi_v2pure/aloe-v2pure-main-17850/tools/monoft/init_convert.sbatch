#!/bin/bash
#SBATCH -J mf-init -p all --ntasks-per-node=1 -c 16 --mem=64G --gres=none -t 2:00:00
# MONOFT init distribution job (submitted by ppm/monoft/launch.sh; -o / -J / -N / -w given there). One init_dist.sh task per host:
#  * all 16 PPM hosts in THIS job (-N16) when they are free (main run paused, nothing else on them), or
#  * the 8 FT hosts here (-N8) + the 8 other (source-only) hosts as an --overlap step of the job already running there (MF_SRC_JOB, e.g. the
#    LR-decay branch; CPU only, --cpu-bind=none, nice) when that job took them first.
# env (from launch.sh): MF_S MF_FT_HOSTS MF_ALL_HOSTS MF_CTL MF_INIT MF_STAGE MF_PORT MF_FINAL_SAVE MF_TRAINER MF_TOOLS (the run dir's frozen copy of
#   ppm/monoft/tools) [MF_SRC_JOB MF_SRC_HOSTS]
set -uo pipefail
: "${MF_S:?}" "${MF_FT_HOSTS:?}" "${MF_ALL_HOSTS:?}" "${MF_CTL:?}"
export MF_MBPS=${MF_MBPS:-0} MF_WORKERS=${MF_WORKERS:-4} MF_DRY=0
W=${MF_TOOLS:-/data/home/amartyavarma/ppm/monoft/tools}/init_dist.sh
echo "== mf-init job $SLURM_JOB_ID $(TZ=America/Los_Angeles date '+%F %T PT') dcp-$MF_S -> node-local $MF_INIT on $MF_FT_HOSTS; job hosts $SLURM_JOB_NODELIST; src overlap job ${MF_SRC_JOB:-none}"
rm -f $MF_CTL/manifest-*.json $MF_CTL/FETCHED-* $MF_CTL/FAILED-* $MF_CTL/ABORT-* $MF_CTL/need-*.json
if [ -n "${MF_SRC_JOB:-}" ]; then
  : "${MF_SRC_HOSTS:?}"
  env $(env | grep -oE '^(SLURM|SBATCH|SRUN|SALLOC)_[A-Za-z0-9_]*=' | tr -d = | sed 's/^/-u /') \
    srun --jobid $MF_SRC_JOB --overlap -N8 -n8 --ntasks-per-node=1 -w $MF_SRC_HOSTS --cpu-bind=none -c 4 --kill-on-bad-exit=0 bash $W &
  P2=$!
  srun -N$SLURM_NNODES -n$SLURM_NNODES --ntasks-per-node=1 -c 16 --cpu-bind=none --kill-on-bad-exit=0 bash $W; rc=$?
  wait $P2; rc2=$?; [ $rc = 0 ] && rc=$rc2
else
  srun -N$SLURM_NNODES -n$SLURM_NNODES --ntasks-per-node=1 -c 16 --cpu-bind=none --kill-on-bad-exit=0 bash $W; rc=$?
fi
nf=$(ls $MF_CTL/FETCHED-* 2>/dev/null | wc -l)
echo "== mf-init rc $rc: $nf of 8 FT hosts hold a load_init-checked COMPLETE init $(TZ=America/Los_Angeles date '+%T PT')"
[ "$nf" = 8 ] || rc=1
exit $rc
