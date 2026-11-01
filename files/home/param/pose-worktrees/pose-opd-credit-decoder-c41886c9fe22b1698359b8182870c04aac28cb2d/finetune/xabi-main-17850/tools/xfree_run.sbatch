#!/bin/bash
#SBATCH -J v2pm17850-xfree -p all --ntasks-per-node=1 -c 4 --mem=16G --gres=none -t 1-00:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-v2pure-main-17850/logs/xfree-%j.log
# one xfree_run.sh task per host (doc there). env: XF_RUN XF_REPO XF_STEPS XF_FINAL XF_HOSTS XF_NTASKS [XF_TLOG XF_TJOB]
# usage: XF_...=... sbatch -N<n> -w <XF_HOSTS> [--dependency=after:<train job>] xfree_run.sbatch
: "${XF_RUN:?}" "${XF_REPO:?}" "${XF_STEPS:?}" "${XF_FINAL:?}" "${XF_HOSTS:?}" "${XF_NTASKS:?}"
echo "== xfree $XF_RUN job $SLURM_JOB_ID $(TZ=America/Los_Angeles date '+%F %T PT') hosts $XF_HOSTS ranks $XF_NTASKS steps [$XF_STEPS] -> $XF_REPO log ${XF_TLOG:-} train ${XF_TJOB:-}"
srun -N$SLURM_NNODES -n$SLURM_NNODES --ntasks-per-node=1 -c 4 --cpu-bind=none --kill-on-bad-exit=0 \
  bash /data/home/amartyavarma/ppm/aloe-v2pure-main-17850/tools/xfree_run.sh "$XF_RUN" "$XF_REPO" "$XF_STEPS" "$XF_FINAL" "$XF_HOSTS" "$XF_NTASKS" "${XF_TLOG:-}" "${XF_TJOB:-}"
rc=$?; echo "== exit $rc $(TZ=America/Los_Angeles date '+%T PT')"; grep -F "job $SLURM_JOB_ID" /data/home/amartyavarma/ppm/$XF_RUN/logs/BACKUP_LOG.txt | tail -n 60; exit $rc
