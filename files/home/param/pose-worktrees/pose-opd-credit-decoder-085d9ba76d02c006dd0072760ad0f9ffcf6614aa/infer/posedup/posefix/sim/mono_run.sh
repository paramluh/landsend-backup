#!/bin/bash
# usage: mono_run.sh <physical GPU pair e.g. 6,7> <runs.json> <out dir>   (inside a posefix job via srun --overlap)
source /data/home/amartyavarma/ppm/aloe-posedup-live/env_live.sh
export TRITON_CACHE_DIR=/scratch/amartyavarma/posefix/triton-cache MIMI_DIR=/data/home/amartyavarma/models/mimi FA3_DIR=/data/home/amartyavarma/models/qwen-s2/vllm-flash-attn3
export CUDA_VISIBLE_DEVICES=$1
exec $PY -u /data/home/amartyavarma/ppm/aloe-posedup-live/posefix/sim/mono.py /data/home/amartyavarma/ppm/aloe-pose-mono-8020/exports/export-357 $2 $3 > $3/inst-$(basename $2 .json)-$SLURM_JOB_ID-r.log 2>&1
