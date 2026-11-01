#!/bin/bash
P=/data/home/amartyavarma/ppm/aloe-ft-32-samsep/persona; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=
cd /tmp
set -a; source $P/run_persona.env; set +a
export QQA_RUN=$QQA_RUN-jDRY
for f in $P/code/train_samsep_persona.py $P/code/data_aloe_persona.py; do $PY -m py_compile $f && echo "compile ok $f"; done
source /data/home/amartyavarma/qwen-s2-imend/hera3/env.sh
$PY -I -B $P/tests/cfg_dry.py $P/code $P/tests/dry
