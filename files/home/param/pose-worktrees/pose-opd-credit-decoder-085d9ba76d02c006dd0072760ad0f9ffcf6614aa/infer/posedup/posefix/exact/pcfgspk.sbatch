#!/bin/bash
#SBATCH -J posefix-pcfgspk
#SBATCH -w slinky-4
#SBATCH -x slinky-9
#SBATCH --gres=gpu:2
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=120G
#SBATCH -t 0:50:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-posedup-live/posefix/exact/pcfgspk-%j.log
source /data/home/amartyavarma/ppm/aloe-posedup-live/env_live.sh
export TRITON_CACHE_DIR=/scratch/amartyavarma/posefix/triton-cache MIMI_DIR=/data/home/amartyavarma/models/mimi FA3_DIR=/data/home/amartyavarma/models/qwen-s2/vllm-flash-attn3
cd /tmp
$PY -u /data/home/amartyavarma/ppm/aloe-posedup-live/posefix/sim/test_pcfgspk.py live /scratch/amartyavarma/pdsamples/export-498 /data/home/amartyavarma/ppm/aloe-posedup-live/posefix/exact/test_pcfgspk_live.json 150
$PY -u /data/home/amartyavarma/ppm/aloe-posedup-live/posefix/sim/test_pcfgspk.py mono /data/home/amartyavarma/ppm/aloe-pose-mono-8020/exports/export-357 /data/home/amartyavarma/ppm/aloe-posedup-live/posefix/exact/test_pcfgspk_mono.json 150
