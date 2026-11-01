#!/bin/bash
#SBATCH -J genweb-dup
#SBATCH -x slinky-9
#SBATCH -c 8
#SBATCH -t 24:00:00
#SBATCH -o logs/%x-%j.out
# GPU pair for the playground's DUPLEX worker (genweb/worker_dup.py). If worker_dup.py exists it runs in a restart loop; else the job holds the
# GPUs for its development (srun --overlap -n 1 into this job).
: "${CANOPY_ROOT:?export CANOPY_ROOT=<cluster home that holds ppm/ venvs/ models/; see README>}"
bash $CANOPY_ROOT/ppm/aloe-posedup-live/tools/gpu_guard.sh "playground duplex hold" || { echo "GPU guard refused playground duplex hold; exiting"; exit 3; }   # 10-27 main: never slinky-4 GPUs 6-7 / SM < 1500 MHz for live / user-facing work
source $CANOPY_ROOT/ppm/aloe-posedup-live/env_live.sh
export TRITON_CACHE_DIR=/scratch/$USER/genweb/triton-cache MIMI_DIR=$CANOPY_ROOT/models/mimi FA3_DIR=$CANOPY_ROOT/models/qwen-s2/vllm-flash-attn3
G=${GENWEB:-$CANOPY_ROOT/ppm/aloe-posedup-live/genweb}
export TRITON_CACHE_AUTOTUNING=1 TRITON_CACHE_DIR=$G/triton-shared   # 10-25: shared autotune cache = the same kernel picks in every duplex worker (reproducible links)
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES host $(hostname) $(date)"
while true; do
  if [ -f $G/worker_dup.py ] && [ -f $G/DUP_ENABLED ]; then $PY -u $G/worker_dup.py d$SLURM_JOB_ID >> $G/logs/worker-dup-$SLURM_JOB_ID.log 2>&1; echo "dup worker exited rc=$? $(date)" >> $G/logs/worker-dup-$SLURM_JOB_ID.log; fi
  sleep 10
done
