#!/bin/bash
# pose6 A3 GPU worker test (NOT RUN YET). Needs: 2 GPUs (one pair, H100), -c 8, --mem=120G, ~15 min, and GENWEB_CK = a checkpoint with
# tower_p.s_emb (cb0) + inp.g0 (the first 31101 export, a pose6 FT export, or A1's /scratch test copy with random non-zero new params).
#   srun --overlap --jobid <J> -w <node> -x slinky-9 -n 1 -c 8 --mem=120G --gres=gpu:2 --cpu-bind=none \
#        env GENWEB_CK=<ck> bash /data/home/amartyavarma/ppm/pose6/worker/tests/run_gpu_test.sh
# Same env as the playground worker units; own Triton cache; writes only pose6/worker/tests/out/gpu (+ node /scratch triton cache).
: "${GENWEB_CK:?set GENWEB_CK}"
source /data/home/amartyavarma/ppm/aloe-posedup-live/env_live.sh
export TRITON_CACHE_DIR=/scratch/amartyavarma/pose6/triton-cache MIMI_DIR=/data/home/amartyavarma/models/mimi FA3_DIR=/data/home/amartyavarma/models/qwen-s2/vllm-flash-attn3
export PYTHONDONTWRITEBYTECODE=1
unset GENWEB_WEIGHTS PPM_S2P PPM_G0COND
mkdir -p $TRITON_CACHE_DIR
echo "host $(hostname) CVD=$CUDA_VISIBLE_DEVICES ck $GENWEB_CK $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=index,name,clocks.sm,memory.used --format=csv,noheader
$PY -B /data/home/amartyavarma/ppm/pose6/worker/tests/test_worker_gpu.py
rc=$?; echo "rc=$rc $(date -u +%FT%TZ)"; exit $rc
