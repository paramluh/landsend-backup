#!/bin/bash
# one task of a SINGLE-NODE 8,1,1 test (srun -n <G> inside an allocation): rank = SLURM_PROCID, world = SLURM_NTASKS.
# usage: srun ... bash task_1node.sh <env file> <script.py>   (env file sourced with set -a; PPM_* already in the env override it)
ENVF=$1; shift
PRE=$(env | grep '^PPM_')
set -a; source "$ENVF"; set +a
if [ -n "$PRE" ]; then while IFS= read -r l; do export "$l"; done <<< "$PRE"; fi
export RANK=$SLURM_PROCID LOCAL_RANK=$SLURM_LOCALID WORLD_SIZE=$SLURM_NTASKS LOCAL_WORLD_SIZE=$SLURM_NTASKS
export MASTER_ADDR=$(hostname -s) MASTER_PORT=${MASTER_PORT:-29871}
export HF_HOME=/data/home/amartyavarma/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export CUDA_DEVICE_MAX_CONNECTIONS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=/scratch/amartyavarma/triton-cache-ppm811
export TORCHINDUCTOR_CACHE_DIR=/scratch/amartyavarma/inductor-cache-ppm811/$(hostname -s)
source /data/home/amartyavarma/qwen-s2-imend/hera3/env.sh
[ "${GEMM_PRECISION:-bf16}" = bf16 ] || { echo "ABORT: bf16 only"; exit 1; }
exec /data/home/amartyavarma/venvs/qwen-s2/bin/python "$@"
