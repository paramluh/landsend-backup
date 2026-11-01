#!/bin/bash
#SBATCH -J s2p-engtest -p all -w slinky-10 -x slinky-9 -N1 -n1 -c 8 --mem=120G --gres=gpu:2 -t 0-01:30:00
#SBATCH -o /data/home/amartyavarma/ppm/pose-s2p/tests/out/engine-%j.log
# pose-s2p tests (a) / (b) / (d) on the borrowed regular-codes speech backend GPUs (job 30862's 2 GPUs on slinky-10); env = posedyn/onestep.sbatch
source /data/home/amartyavarma/ppm/aloe-posedup-live/env_live.sh
export TRITON_CACHE_DIR=/scratch/amartyavarma/genweb/triton-cache MIMI_DIR=/data/home/amartyavarma/models/mimi FA3_DIR=/data/home/amartyavarma/models/qwen-s2/vllm-flash-attn3; mkdir -p $TRITON_CACHE_DIR
nvidia-smi --query-gpu=index,name,memory.total,memory.used,clocks.sm --format=csv; echo "CVD=$CUDA_VISIBLE_DEVICES $(date)"
$PY -u /data/home/amartyavarma/ppm/pose-s2p/tests/test_engine.py ${STAGES:-a,b,d}
echo "exit rc=$? $(date)"
