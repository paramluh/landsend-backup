#!/bin/bash
# single-node PPM runner (inside an allocation / srun on ONE node): torchrun x 8 GPUs of SCRIPT with ARGS. Env knobs PPM_* pass through.
# usage: bash run_local.sh <script.py> [args...]      (e.g. train_ppm.py, tests/step0_full.py <hf_ref>)
set -uo pipefail
C=/data/home/amartyavarma/ppm/code
PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
source /data/home/amartyavarma/qwen-s2-imend/hera3/env.sh
export HF_HOME=/data/home/amartyavarma/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export CUDA_DEVICE_MAX_CONNECTIONS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/scratch/amartyavarma/triton-cache-ppm}
[ "${GEMM_PRECISION:-bf16}" = bf16 ] || { echo "ABORT: bf16 only"; exit 1; }
for f in $C/model_ppm.py $C/data_ppm.py $C/train_ppm.py; do $PY -m py_compile $f || { echo "COMPILE FAIL $f"; exit 1; }; done
NP=${NPROC:-8}
exec $PY -m torch.distributed.run --standalone --nproc_per_node $NP "$C/$1" "${@:2}"
