#!/bin/bash
#SBATCH -J pose6-a3-gpu -p all -x slinky-9 -N1 -n1 -c 8 --mem=120G --gres=gpu:2 -t 0:25:00
#SBATCH -o /data/home/amartyavarma/ppm/pose6/worker/tests/logs/gpu-%j.log
# pose6 A3: GPU end-to-end test of pose6/worker/worker_s2pg0.py (the real Worker: build + check_mods + capture, 2 GPUs, one process), with the
# TEST-ONLY overlay on export-357 (read-only) until a checkpoint with both keys exists. Writes only pose6/worker/tests/{out/gpu,logs} and the
# node-local Triton cache. Submitted by main only (e.g. sbatch -w slinky-4 --dependency=afterany:31138 <this file>).
source /data/home/amartyavarma/ppm/aloe-posedup-live/env_live.sh
export TRITON_CACHE_DIR=/scratch/amartyavarma/pose6/triton-cache MIMI_DIR=/data/home/amartyavarma/models/mimi FA3_DIR=/data/home/amartyavarma/models/qwen-s2/vllm-flash-attn3
export PYTHONDONTWRITEBYTECODE=1
export GENWEB_CK=${GENWEB_CK:-/data/home/amartyavarma/ppm/aloe-pose-mono-8020/exports/export-357} POSE6_TEST_OVERLAY=${POSE6_TEST_OVERLAY:-1}
unset GENWEB_WEIGHTS PPM_S2P PPM_G0COND
mkdir -p "${TRITON_CACHE_DIR:?}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,clocks.sm --format=csv; echo "CVD=$CUDA_VISIBLE_DEVICES host $(hostname -s) ck $GENWEB_CK overlay $POSE6_TEST_OVERLAY $(TZ=America/Los_Angeles date '+%F %T PT')"
# GUARD (= A1's test_engine6.sbatch): exit 3 before loading anything if a throttled GPU is assigned (slinky-4 physical 6/7, or SM clock under load < 75 % max)
echo "guard: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
nvidia-smi --query-gpu=index,clocks.sm,clocks.max.sm,temperature.gpu --format=csv
$PY -u /data/home/amartyavarma/ppm/pose6/infer/tests/gpu_guard.py || { rc=$?; echo "GUARD FAILED rc=$rc: not running the test"; exit 3; }
$PY -B -u /data/home/amartyavarma/ppm/pose6/worker/tests/test_worker_gpu.py
rc=$?; echo "exit rc=$rc $(TZ=America/Los_Angeles date '+%F %T PT')"; exit $rc
