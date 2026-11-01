#!/bin/bash
#SBATCH -J posedyn -p all -w slinky-10 -x slinky-9 -N1 -n1 -c 8 --mem=120G --gres=gpu:2 -t 0-02:00:00
#SBATCH -o /data/home/amartyavarma/ppm/posedyn/onestep-%j.log
source /data/home/amartyavarma/ppm/aloe-posedup-live/env_live.sh
export TRITON_CACHE_DIR=/scratch/amartyavarma/genweb/triton-cache MIMI_DIR=/data/home/amartyavarma/models/mimi FA3_DIR=/data/home/amartyavarma/models/qwen-s2/vllm-flash-attn3; mkdir -p $TRITON_CACHE_DIR
cd /data/home/amartyavarma/ppm/posedyn
$PY -u /data/home/amartyavarma/ppm/posedyn/onestep.py
