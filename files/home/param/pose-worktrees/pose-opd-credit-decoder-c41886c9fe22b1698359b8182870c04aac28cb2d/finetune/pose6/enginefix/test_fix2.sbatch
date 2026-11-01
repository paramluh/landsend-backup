#!/bin/bash
#SBATCH -J pose6-enginefix2 -p all -x slinky-9 -N1 -n1 -c 2 --mem=12G -t 0-00:15:00 --nice=100
#SBATCH -o /data/home/amartyavarma/ppm/pose6/enginefix/logs/test_fix2-%j.log
# pose6 ENGINEFIX CPU proofs: fast_ppm (deployed + pose-s2p) and the duplex live_ppm_pose.py. A1_FIX_MODE=readonly: only the duplex R part
# (nothing written to this node's /scratch). Otherwise synthetic records (KB) go to this node's /scratch/amartyavarma/pose6-enginefix/<job>.
echo "host $(hostname -s) start $(TZ=America/Los_Angeles date '+%F %T PT') job ${SLURM_JOB_ID}"
source /data/home/amartyavarma/ppm/aloe-posedup-live/env_live.sh
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2
if [ "${A1_FIX_MODE:-full}" = readonly ]; then
  $PY -u /data/home/amartyavarma/ppm/pose6/enginefix/test_fix_duplex.py R; echo "duplex R rc $?"
else
  $PY -u /data/home/amartyavarma/ppm/pose6/enginefix/test_fix.py; echo "fast_ppm rc $?"
  $PY -u /data/home/amartyavarma/ppm/pose6/enginefix/test_fix_duplex.py R,S,D; echo "duplex rc $?"
fi
echo "end $(TZ=America/Los_Angeles date '+%F %T PT')"
