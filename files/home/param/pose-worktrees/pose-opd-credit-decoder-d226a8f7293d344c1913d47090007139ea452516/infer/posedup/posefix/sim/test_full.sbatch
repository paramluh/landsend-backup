#!/bin/bash
#SBATCH -J posefix-tfull
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
$PY -u /data/home/amartyavarma/ppm/aloe-posedup-live/posefix/sim/test_full.py /scratch/amartyavarma/aloe-posedup-live/export-498-dcp /data/home/amartyavarma/ppm/aloe-posedup-live/posefix/out/stagefull2 01_ab63d745:0:2.0 10_27286bdd:0:2.0 01_ab63d745:0:11.68
