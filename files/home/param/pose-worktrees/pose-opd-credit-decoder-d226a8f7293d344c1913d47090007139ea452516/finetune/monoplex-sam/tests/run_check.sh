#!/bin/bash
export HF_HOME=/data/home/amartyavarma/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
source /data/home/amartyavarma/qwen-s2-imend/hera3/env.sh
exec nice /data/home/amartyavarma/venvs/qwen-s2/bin/python /data/home/amartyavarma/ppm/aloe-pose-mono-8020/tests/plan_mono_check.py "$@"
