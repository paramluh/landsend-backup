#!/bin/bash
#SBATCH -J pose6-a1-cpu2 -p all -x slinky-9 -N1 -n1 -c 2 --mem=16G -t 0-00:20:00 --nice=100
#SBATCH -o /data/home/amartyavarma/ppm/pose6/infer/tests/logs/cpu2-%j.log
# pose6 A1 small CPU re-run of cpu_load.py (S, L1, LK, L2) after the link-off (decp_bb_off) load-path change. No GPU. Synthetic test
# checkpoints only on node /scratch/amartyavarma/pose6-a1/<job> (never /data).
echo "host $(hostname -s) start $(TZ=America/Los_Angeles date '+%F %T PT') job ${SLURM_JOB_ID}"
source /data/home/amartyavarma/ppm/aloe-posedup-live/env_live.sh
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 A1_SCRATCH=/scratch/amartyavarma/pose6-a1/${SLURM_JOB_ID}
mkdir -p "${A1_SCRATCH:?}"
$PY -u /data/home/amartyavarma/ppm/pose6/infer/tests/cpu_load.py ${A1_STAGES:-S,L1,LK,L2}
echo "rc $? end $(TZ=America/Los_Angeles date '+%F %T PT')"
