#!/bin/bash
#SBATCH -J posefix-sim
#SBATCH -x slinky-9
#SBATCH --gres=gpu:2
#SBATCH -c 12
#SBATCH --mem=200G
#SBATCH -t 4:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-posedup-live/posefix/sim/logs/%x-%j.out
# usage: sbatch -w <node> sim.sbatch <ck> <runs.json> <out dir>
source /data/home/amartyavarma/ppm/aloe-posedup-live/env_live.sh
export TRITON_CACHE_DIR=/scratch/amartyavarma/posefix/triton-cache; mkdir -p $TRITON_CACHE_DIR
nvidia-smi --query-gpu=index,name,clocks.sm,memory.used --format=csv,noheader
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
$PY -u /data/home/amartyavarma/ppm/aloe-posedup-live/posefix/sim/replay.py "$@"
