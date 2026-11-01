#!/bin/bash
# pose6 A1 (iii-strong) + (loc): ONE process inside pose6's placeholder allocation (srun --overlap --jobid 31150 ...), guard first.
source /data/home/amartyavarma/ppm/aloe-posedup-live/env_live.sh
export TRITON_CACHE_DIR=/scratch/amartyavarma/pose6/triton-cache MIMI_DIR=/data/home/amartyavarma/models/mimi FA3_DIR=/data/home/amartyavarma/models/qwen-s2/vllm-flash-attn3
mkdir -p "${TRITON_CACHE_DIR:?}"
echo "host $(hostname -s) job ${SLURM_JOB_ID:-} step ${SLURM_STEP_ID:-} CVD=${CUDA_VISIBLE_DEVICES:-} start $(TZ=America/Los_Angeles date '+%F %T PT')"
nvidia-smi --query-gpu=index,clocks.sm,clocks.max.sm,temperature.gpu,memory.used --format=csv
$PY -u /data/home/amartyavarma/ppm/pose6/infer/tests/gpu_guard.py || { rc=$?; echo "GUARD FAILED rc=$rc: not running"; exit 3; }
$PY -u /data/home/amartyavarma/ppm/pose6/infer/tests/test_engine6.py loc2
echo "exit rc=$? $(TZ=America/Los_Angeles date '+%F %T PT')"
