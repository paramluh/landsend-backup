#!/bin/bash
# usage: dup_run.sh <physical GPU pair> <ck> <runs.json> <out dir>   (inside a posefix job via srun --overlap)
source /data/home/amartyavarma/ppm/aloe-posedup-live/env_live.sh
export TRITON_CACHE_DIR=/scratch/amartyavarma/posefix/triton-cache CUDA_VISIBLE_DEVICES=$1
exec $PY -u /data/home/amartyavarma/ppm/aloe-posedup-live/posefix/sim/replay.py $2 $3 $4 > $4/inst-$(basename $3 .json)-$SLURM_JOB_ID.log 2>&1
