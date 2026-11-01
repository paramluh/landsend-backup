#!/bin/bash
#SBATCH -J pose6-a1-gpu -p all -x slinky-9 -N1 -n1 -c 8 --mem=128G --gres=gpu:2 -t 0-01:05:00
#SBATCH -o /data/home/amartyavarma/ppm/pose6/infer/tests/logs/gpu-%j.log
# pose6 A1 engine proofs (i)/(ii), (w), (iii), (iv): ONE process, 2 GPUs of one node (= the genweb worker layout), export-357 read-only.
# Writes only pose6/infer/tests/{out,logs}. submit (node from main): sbatch -w <node> [--gres=gpu:2] .../tests/test_engine6.sbatch
source /data/home/amartyavarma/ppm/aloe-posedup-live/env_live.sh
export TRITON_CACHE_DIR=/scratch/amartyavarma/pose6/triton-cache MIMI_DIR=/data/home/amartyavarma/models/mimi FA3_DIR=/data/home/amartyavarma/models/qwen-s2/vllm-flash-attn3
mkdir -p "${TRITON_CACHE_DIR:?}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,clocks.sm --format=csv; echo "CVD=$CUDA_VISIBLE_DEVICES host $(hostname -s) $(TZ=America/Los_Angeles date '+%F %T PT')"
# GUARD (pose6 10-27): exit 3 before loading anything if a throttled GPU is assigned (slinky-4 physical 6/7, or SM clock under load < 75 % max)
echo "guard: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
nvidia-smi --query-gpu=index,clocks.sm,clocks.max.sm,temperature.gpu --format=csv
$PY -u /data/home/amartyavarma/ppm/pose6/infer/tests/gpu_guard.py || { rc=$?; echo "GUARD FAILED rc=$rc: not running the proofs"; exit 3; }
$PY -u /data/home/amartyavarma/ppm/pose6/infer/tests/test_engine6.py ${A1_STAGES:-i,w,iii,iv}
echo "exit rc=$? $(TZ=America/Los_Angeles date '+%F %T PT')"
