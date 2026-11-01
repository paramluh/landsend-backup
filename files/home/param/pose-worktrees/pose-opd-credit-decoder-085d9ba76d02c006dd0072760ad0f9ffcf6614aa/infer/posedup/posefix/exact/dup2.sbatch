#!/bin/bash
#SBATCH -J posefix-exdup2
#SBATCH -w slinky-4
#SBATCH -x slinky-9
#SBATCH --gres=gpu:2
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=120G
#SBATCH -t 1:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-posedup-live/posefix/exact/dup2-%j.log
source /data/home/amartyavarma/ppm/aloe-posedup-live/env_live.sh
export TRITON_CACHE_DIR=/scratch/amartyavarma/posefix/triton-cache
nvidia-smi --query-gpu=index,clocks.sm,memory.used --format=csv,noheader
echo CVD=$CUDA_VISIBLE_DEVICES
cd /tmp
$PY -u /data/home/amartyavarma/ppm/aloe-posedup-live/posefix/sim/test_light.py /scratch/amartyavarma/pdsamples/export-498 /data/home/amartyavarma/ppm/aloe-posedup-live/posefix/exact/test_light2.json 220 /data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/samples/convs/c5.npz /data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/samples/convs/c6.npz /data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/samples/convs/c7.npz
