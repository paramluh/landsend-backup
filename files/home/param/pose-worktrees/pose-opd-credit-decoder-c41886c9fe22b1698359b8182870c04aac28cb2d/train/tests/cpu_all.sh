#!/bin/bash
# all CPU tests on one node with the stores
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/data/home/amartyavarma/.cache/huggingface
source /data/home/amartyavarma/qwen-s2-imend/hera3/env.sh
PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python; C=/data/home/amartyavarma/ppm/code
for t in test_units_cpu.py "test_data_cpu.py 40" "test_cd_align_cpu.py 40" test_chat_cpu.py; do echo "=== $t"; $PY $C/tests/$t 2>&1 | tail -12; done
