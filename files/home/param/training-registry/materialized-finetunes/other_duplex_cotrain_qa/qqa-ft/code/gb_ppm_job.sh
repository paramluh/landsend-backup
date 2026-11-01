#!/bin/bash
# sbatch body for gb_ppm.sh: the infer_ppm.sh environment, then gb_ppm.py with the given arguments (bash, not sh: env.sh needs `source`).
source /data/home/amartyavarma/qwen-s2-imend/hera3/env.sh
export HF_HOME=/data/home/amartyavarma/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 \
  PPM_CD=2 GEMM_PRECISION=bf16 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=/scratch/amartyavarma/triton-cache-ppm-infer
exec /data/home/amartyavarma/venvs/qwen-s2/bin/python -u /data/home/amartyavarma/ppm/code/gb_ppm.py "$@"
