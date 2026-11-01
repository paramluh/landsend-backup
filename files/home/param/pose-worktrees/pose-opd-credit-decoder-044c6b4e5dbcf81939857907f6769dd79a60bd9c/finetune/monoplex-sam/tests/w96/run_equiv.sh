#!/bin/bash
# one rank of step_equiv.py (srun task). usage, on a compute node (never the login node, never MoT hosts / slinky-9 / slinky-10):
#   srun -p all -w slinky-4 -N1 -n<W> -c2 --mem=44G --gres=none -t 40 bash run_equiv.sh <tag> <gpu list> <rows_per_step> <port> [ALOE_NORM_WORLD]
# GPUs: idle demo GPUs shared with their (idle) owners; step_equiv caps itself at min(EQ_CAP_GB, free - EQ_KEEP_GB).
set -uo pipefail
T=/data/home/amartyavarma/ppm/aloe-pose-mono-8020-sam/tests/w96
set -a; source /data/home/amartyavarma/ppm/aloe-pose-mono-8020-sam/aloe_mono8020.env; set +a
export QQA_INIT=/data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-11488
export EQ_TAG=$1 CUDA_VISIBLE_DEVICES=$2 ALOE_ROWS_PER_STEP=$3 MASTER_PORT=$4 ALOE_LOSSCHECK_DUMMY=1 ALOE_NORM_WORLD=${5:-}
export EQ_OUT=${EQ_OUT:-/scratch/amartyavarma/w96test} QQA_LOG_ROOT=$T/out/runs QQA_CONTROL_DIR=$T/out/control
export RANK=$SLURM_PROCID LOCAL_RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_WORLD_SIZE=$SLURM_NTASKS MASTER_ADDR=$(hostname -s)
# = task_ppm.sh's environment
export HF_HOME=/data/home/amartyavarma/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
export CUDA_DEVICE_MAX_CONNECTIONS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=/scratch/amartyavarma/triton-cache-ppm
export TORCHINDUCTOR_CACHE_DIR=/scratch/amartyavarma/inductor-cache-ppm/$(hostname -s)
source /data/home/amartyavarma/qwen-s2-imend/hera3/env.sh
[ "${GEMM_PRECISION:-bf16}" = bf16 ] || { echo "ABORT: bf16 only"; exit 1; }
exec /data/home/amartyavarma/venvs/qwen-s2/bin/python $T/step_equiv.py
