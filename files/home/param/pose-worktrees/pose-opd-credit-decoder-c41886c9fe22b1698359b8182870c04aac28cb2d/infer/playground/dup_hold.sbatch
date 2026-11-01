#!/bin/bash
#SBATCH -J genweb-dup
#SBATCH -x slinky-9
#SBATCH -c 8
#SBATCH -t 24:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-posedup-live/genweb/logs/%x-%j.out
# GPU pair for the playground's DUPLEX worker (genweb/worker_dup.py). If worker_dup.py exists it runs in a restart loop; else the job holds the
# GPUs for its development (srun --overlap -n 1 into this job).
source /data/home/amartyavarma/ppm/aloe-posedup-live/env_live.sh
export TRITON_CACHE_DIR=/scratch/amartyavarma/genweb/triton-cache MIMI_DIR=/data/home/amartyavarma/models/mimi FA3_DIR=/data/home/amartyavarma/models/qwen-s2/vllm-flash-attn3
G=/data/home/amartyavarma/ppm/aloe-posedup-live/genweb
export TRITON_CACHE_AUTOTUNING=1 TRITON_CACHE_DIR=$G/triton-shared   # 10-25: shared autotune cache = the same kernel picks in every duplex worker (reproducible links)
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES host $(hostname) $(date)"
while true; do
  if [ -f $G/worker_dup.py ] && [ -f $G/DUP_ENABLED ]; then $PY -u $G/worker_dup.py d$SLURM_JOB_ID >> $G/logs/worker-dup-$SLURM_JOB_ID.log 2>&1; echo "dup worker exited rc=$? $(date)" >> $G/logs/worker-dup-$SLURM_JOB_ID.log; fi
  sleep 10
done
