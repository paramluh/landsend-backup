#!/bin/bash
#SBATCH -J posefix-tring
#SBATCH -x slinky-9
#SBATCH --gres=gpu:2
#SBATCH -c 8
#SBATCH --mem=120G
#SBATCH -t 1:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-posedup-live/posefix/sim/logs/%x-%j.out
source /data/home/amartyavarma/ppm/aloe-posedup-live/env_live.sh
export TRITON_CACHE_DIR=/scratch/amartyavarma/posefix/triton-cache
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
cd /tmp
$PY -u /data/home/amartyavarma/ppm/aloe-posedup-live/posefix/sim/test_ring.py /scratch/amartyavarma/pdsamples/export-498 01_ab63d745:1:11.68 10_27286bdd:2:4.96 01_ab63d745:1:2.0
