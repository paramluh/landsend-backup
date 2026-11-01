#!/bin/bash
# env = run_poseduplex.env (job 30074's data config) ; the inference prompt rule (voice clip always, no persona) is applied inside prep_convs.py
unset $(compgen -e | grep -E '^(PPM_|QQA_|ALOE_|VOX_|POSEDUP_|S3_|COTRAIN_)')
export POSEDUP_HOSTS=none
set -a; source /data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/run_poseduplex.env; set +a
export ALOE_FLAC_ROOT=$ALOE_SHARED_FLAC          # the shared /data copy of the A FLACs (node-local copies live on the training hosts only)
source /data/home/amartyavarma/qwen-s2-imend/hera3/env.sh
export HF_HOME=/data/home/amartyavarma/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=
exec /data/home/amartyavarma/venvs/qwen-s2/bin/python -u /data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/samples/code/prep_convs.py "$@"
