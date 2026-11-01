#!/bin/bash
#SBATCH -J genweb-worker
#SBATCH -x slinky-9
#SBATCH -c 8
#SBATCH -t 24:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-posedup-live/genweb/logs/%x-%j.out
# MONOPLEX PLAYGROUND hot worker(s): sbatch -w slinky-18 --gres=gpu:2K --mem=<120K>G -c <8K> worker.sbatch  -> K workers, one per GPU pair
source /data/home/amartyavarma/ppm/aloe-posedup-live/env_live.sh
export TRITON_CACHE_DIR=/scratch/amartyavarma/genweb/triton-cache MIMI_DIR=/data/home/amartyavarma/models/mimi FA3_DIR=/data/home/amartyavarma/models/qwen-s2/vllm-flash-attn3
mkdir -p $TRITON_CACHE_DIR
G=/data/home/amartyavarma/ppm/aloe-posedup-live/genweb
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES host $(hostname) $(date)"
IFS=, read -ra GP <<< "$CUDA_VISIBLE_DEVICES"
for ((i=0; i<${#GP[@]}/2; i++)); do
  ( while true; do CUDA_VISIBLE_DEVICES=${GP[$((2*i))]},${GP[$((2*i+1))]} $PY -u $G/worker.py w$SLURM_JOB_ID-$i >> $G/logs/worker-$SLURM_JOB_ID-$i.log 2>&1
      echo "worker $i exited rc=$? $(date)" >> $G/logs/worker-$SLURM_JOB_ID-$i.log; sleep 10; done ) &
done
wait
