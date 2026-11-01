#!/bin/bash
# 10-27 user ("put up rosie"): ONE hot monoplex worker serving the Rosie full-FT weights (ppm/aloe-pose-rosie exports-local/export-18,
# ppm-local bf16 shards = export-357's format) on 2 GPUs: worker.py with GENWEB_WEIGHTS=rosie claims only runs whose request carries
# mono_weights "rosie" (the page's monoplex weights picker); it never rewrites SCHEMA / OPTIONS. Same env as hot_unit.sh's worker, no lane.
G=/data/home/amartyavarma/ppm/aloe-posedup-live/genweb
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES host $(hostname) role ROSIE monoplex worker $(date)"
source /data/home/amartyavarma/ppm/aloe-posedup-live/env_live.sh
export TRITON_CACHE_DIR=/scratch/amartyavarma/genweb/triton-cache MIMI_DIR=/data/home/amartyavarma/models/mimi FA3_DIR=/data/home/amartyavarma/models/qwen-s2/vllm-flash-attn3
export GENWEB_CK=/data/home/amartyavarma/ppm/aloe-pose-rosie/exports-local/export-18 GENWEB_WEIGHTS=rosie
mkdir -p $TRITON_CACHE_DIR
while true; do $PY -u $G/worker.py rosie$SLURM_JOB_ID >> $G/logs/worker-rosie-$SLURM_JOB_ID.log 2>&1; echo "exited rc=$? $(date)" >> $G/logs/worker-rosie-$SLURM_JOB_ID.log; sleep 10; done
