#!/bin/bash
#SBATCH -J posefix-exmono
#SBATCH -w slinky-4
#SBATCH -x slinky-9
#SBATCH --gres=gpu:2
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=120G
#SBATCH -t 1:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-posedup-live/posefix/exact/mono-%j.log
source /data/home/amartyavarma/ppm/aloe-posedup-live/env_live.sh
export TRITON_CACHE_DIR=/scratch/amartyavarma/posefix/triton-cache MIMI_DIR=/data/home/amartyavarma/models/mimi FA3_DIR=/data/home/amartyavarma/models/qwen-s2/vllm-flash-attn3
nvidia-smi --query-gpu=index,clocks.sm,memory.used --format=csv,noheader
echo CVD=$CUDA_VISIBLE_DEVICES
$PY -u /data/home/amartyavarma/ppm/aloe-posedup-live/posefix/sim/test_light_mono.py /data/home/amartyavarma/ppm/aloe-pose-mono-8020/exports/export-357 /data/home/amartyavarma/ppm/aloe-posedup-live/posefix/exact/test_light_mono.json 120 /data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/samples/convs/c4.npz /data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/samples/convs/c5.npz
