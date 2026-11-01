#!/bin/bash
#SBATCH -J genweb-dup
#SBATCH
#SBATCH -x slinky-9
#SBATCH --gres=gpu:2
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=110G
#SBATCH -t 24:00:00
#SBATCH -o logs/%x-%j.out
# 10-26 dupui: the playground's DUPLEX worker pair (= dup_hold.sbatch env). worker_dup.py runs in a restart loop only while DUP_ENABLED AND
# DUPUI_LIVE exist; without DUPUI_LIVE the job just holds the pair for dupui's development (srun --overlap -n 1 into this job).
: "${CANOPY_ROOT:?export CANOPY_ROOT=<cluster home that holds ppm/ venvs/ models/; see README>}"
bash $CANOPY_ROOT/ppm/aloe-posedup-live/tools/gpu_guard.sh "playground duplex unit" || { echo "GPU guard refused playground duplex unit; exiting"; exit 3; }   # 10-27 main: never slinky-4 GPUs 6-7 / SM < 1500 MHz for live / user-facing work
source $CANOPY_ROOT/ppm/aloe-posedup-live/env_live.sh
export TRITON_CACHE_DIR=/scratch/$USER/genweb/triton-cache MIMI_DIR=$CANOPY_ROOT/models/mimi FA3_DIR=$CANOPY_ROOT/models/qwen-s2/vllm-flash-attn3
G=${GENWEB:-$CANOPY_ROOT/ppm/aloe-posedup-live/genweb}
export TRITON_CACHE_AUTOTUNING=1 TRITON_CACHE_DIR=$G/triton-shared
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES host $(hostname) $(date)"
while true; do
  if [ -f $G/worker_dup.py ] && [ -f $G/DUP_ENABLED ] && [ -f $G/DUPUI_LIVE ]; then $PY -u $G/worker_dup.py d$SLURM_JOB_ID >> $G/logs/worker-dup-$SLURM_JOB_ID.log 2>&1; echo "dup worker exited rc=$? $(date)" >> $G/logs/worker-dup-$SLURM_JOB_ID.log; fi
  sleep 10
done
