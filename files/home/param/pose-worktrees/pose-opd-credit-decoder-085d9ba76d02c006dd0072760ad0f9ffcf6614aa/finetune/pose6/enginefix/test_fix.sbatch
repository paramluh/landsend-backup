#!/bin/bash
#SBATCH -J pose6-enginefix -p all -x slinky-9 -N1 -n1 -c 2 --mem=8G -t 0-00:15:00 --nice=100
#SBATCH -o /data/home/amartyavarma/ppm/pose6/enginefix/logs/test_fix-%j.log
# pose6 ENGINEFIX CPU proof (test_fix.py). Tiny synthetic records (KB) on this node's /scratch/amartyavarma/pose6-enginefix/<job> only.
echo "host $(hostname -s) start $(TZ=America/Los_Angeles date '+%F %T PT') job ${SLURM_JOB_ID}"
source /data/home/amartyavarma/ppm/aloe-posedup-live/env_live.sh
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2
$PY -u /data/home/amartyavarma/ppm/pose6/enginefix/test_fix.py
echo "rc $? end $(TZ=America/Los_Angeles date '+%F %T PT')"
