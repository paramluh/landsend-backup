#!/bin/bash
#SBATCH -J posefix-align
#SBATCH -x slinky-9
#SBATCH --gres=gpu:2
#SBATCH -c 8
#SBATCH --mem=120G
#SBATCH -t 1:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-posedup-live/posefix/sim/logs/%x-%j.out
source /data/home/amartyavarma/ppm/aloe-posedup-live/env_live.sh
export TRITON_CACHE_DIR=/scratch/amartyavarma/posefix/triton-cache MIMI_DIR=/data/home/amartyavarma/models/mimi FA3_DIR=/data/home/amartyavarma/models/qwen-s2/vllm-flash-attn3
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
$PY -u /data/home/amartyavarma/ppm/aloe-posedup-live/posefix/sim/align.py /data/home/amartyavarma/ppm/aloe-pose-mono-8020/exports/export-357 01_ab63d745 /data/home/amartyavarma/ppm/aloe-posedup-live/posefix/sim/prefill
