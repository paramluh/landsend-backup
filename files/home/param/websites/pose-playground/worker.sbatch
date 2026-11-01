#!/bin/bash
#SBATCH -J genweb-worker
#SBATCH -x slinky-9
#SBATCH -c 8
#SBATCH -t 24:00:00
#SBATCH -o logs/%x-%j.out
# MONOPLEX PLAYGROUND hot worker(s): sbatch -w slinky-18 --gres=gpu:2K --mem=<120K>G -c <8K> worker.sbatch  -> K workers, one per GPU pair
: "${CANOPY_ROOT:?export CANOPY_ROOT=<cluster home that holds ppm/ venvs/ models/; see README>}"
bash $CANOPY_ROOT/ppm/aloe-posedup-live/tools/gpu_guard.sh "playground worker" || { echo "GPU guard refused playground worker; exiting"; exit 3; }   # 10-27 main: never slinky-4 GPUs 6-7 / SM < 1500 MHz for live / user-facing work
source $CANOPY_ROOT/ppm/aloe-posedup-live/env_live.sh
export TRITON_CACHE_DIR=/scratch/$USER/genweb/triton-cache MIMI_DIR=$CANOPY_ROOT/models/mimi FA3_DIR=$CANOPY_ROOT/models/qwen-s2/vllm-flash-attn3
mkdir -p $TRITON_CACHE_DIR
G=${GENWEB:-$CANOPY_ROOT/ppm/aloe-posedup-live/genweb}
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES host $(hostname) $(date)"
IFS=, read -ra GP <<< "$CUDA_VISIBLE_DEVICES"
for ((i=0; i<${#GP[@]}/2; i++)); do
  ( while true; do CUDA_VISIBLE_DEVICES=${GP[$((2*i))]},${GP[$((2*i+1))]} $PY -u $G/worker.py w$SLURM_JOB_ID-$i >> $G/logs/worker-$SLURM_JOB_ID-$i.log 2>&1
      echo "worker $i exited rc=$? $(date)" >> $G/logs/worker-$SLURM_JOB_ID-$i.log; sleep 10; done ) &
done
wait
