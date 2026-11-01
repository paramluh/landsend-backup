#!/bin/bash
# xabi-view runner: CPU only, offline HF cache, trainer env (aloe_v2pure.env), low threads
set -a; source /data/home/amartyavarma/ppm/aloe-v2pure-ft/aloe_v2pure.env; set +a
export PPM_CD=2 CUDA_VISIBLE_DEVICES= HF_HOME=/data/home/amartyavarma/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
source /data/home/amartyavarma/qwen-s2-imend/hera3/env.sh
exec nice -n 10 /data/home/amartyavarma/venvs/qwen-s2/bin/python -u "$@"
