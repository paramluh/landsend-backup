# sourced by every fast pose engine launcher (= ppm/aloe-m1vox-live/env_live.sh + node-scratch triton cache)
source /data/home/amartyavarma/qwen-s2-imend/hera3/env.sh
export HF_HOME=/data/home/amartyavarma/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export PPM_CD=2 GEMM_PRECISION=bf16 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1
export VOXTRAL_DIR=/data/home/amartyavarma/models/voxtral-realtime
export TRITON_CACHE_DIR=/scratch/amartyavarma/triton-cache-ppm-fast
export PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python FI=/data/home/amartyavarma/ppm/fastinfer
mkdir -p $TRITON_CACHE_DIR
