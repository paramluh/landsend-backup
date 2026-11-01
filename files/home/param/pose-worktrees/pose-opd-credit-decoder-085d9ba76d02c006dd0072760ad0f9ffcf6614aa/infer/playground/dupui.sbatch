#!/bin/bash
#SBATCH -J genweb-dup
#SBATCH -w slinky-18
#SBATCH -x slinky-9
#SBATCH --gres=gpu:2
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=110G
#SBATCH -t 24:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-posedup-live/genweb/logs/%x-%j.out
# 10-26 dupui: the playground's DUPLEX worker pair (= dup_hold.sbatch env). worker_dup.py runs in a restart loop only while DUP_ENABLED AND
# DUPUI_LIVE exist; without DUPUI_LIVE the job just holds the pair for dupui's development (srun --overlap -n 1 into this job).
source /data/home/amartyavarma/ppm/aloe-posedup-live/env_live.sh
export TRITON_CACHE_DIR=/scratch/amartyavarma/genweb/triton-cache MIMI_DIR=/data/home/amartyavarma/models/mimi FA3_DIR=/data/home/amartyavarma/models/qwen-s2/vllm-flash-attn3
G=/data/home/amartyavarma/ppm/aloe-posedup-live/genweb
export TRITON_CACHE_AUTOTUNING=1 TRITON_CACHE_DIR=$G/triton-shared
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES host $(hostname) $(date)"
while true; do
  if [ -f $G/worker_dup.py ] && [ -f $G/DUP_ENABLED ] && [ -f $G/DUPUI_LIVE ]; then $PY -u $G/worker_dup.py d$SLURM_JOB_ID >> $G/logs/worker-dup-$SLURM_JOB_ID.log 2>&1; echo "dup worker exited rc=$? $(date)" >> $G/logs/worker-dup-$SLURM_JOB_ID.log; fi
  sleep 10
done
