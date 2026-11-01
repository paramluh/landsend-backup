#!/bin/bash
C=/data/home/amartyavarma/ppm/code-811; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
set -a; source $C/prod_811_fresh.env; set +a
export RANK=0 WORLD_SIZE=128 LOCAL_RANK=0 LOCAL_WORLD_SIZE=8 PPM_RUN=ppm-v1-jFRESH PPM_RUN_BASE=ppm-v1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
for f in $C/model_ppm.py $C/data_ppm.py $C/train_ppm.py $C/s811.py $C/safety_ppm.py $C/chat_ppm.py $C/fresh/fresh_pose.py; do $PY -m py_compile $f || echo "COMPILE FAIL $f"; done
$PY $C/test811/cfg_diff.py /data/home/amartyavarma/ppm/logs/ppm-26391.log $C/fresh/cfgdiff/CFG_DIFF_vs_26391.txt
