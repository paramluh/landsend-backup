# sourced by the aloe-posedup-live launchers (= aloe-m1vox-live/env_live.sh + this demo's paths)
source /data/home/amartyavarma/qwen-s2-imend/hera3/env.sh
export HF_HOME=/data/home/amartyavarma/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export PPM_CD=2 GEMM_PRECISION=bf16 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VOXTRAL_DIR=/data/home/amartyavarma/models/voxtral-realtime
export TRITON_CACHE_DIR=/scratch/amartyavarma/triton-cache-ppm-live
export PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python LV=/data/home/amartyavarma/ppm/aloe-posedup-live
export PPMLIVE_DEFAULTS=nopersona
mkdir -p $TRITON_CACHE_DIR
