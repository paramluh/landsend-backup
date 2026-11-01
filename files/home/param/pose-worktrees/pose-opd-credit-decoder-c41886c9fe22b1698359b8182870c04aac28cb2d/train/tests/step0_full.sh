#!/bin/bash
# step-0 checks 1 + 2 on the FULL-SIZE model, one node: (a) stock HF Qwen3-32B reference logits -> /scratch; (b) full PPM FSDP x 8 compare
# + init losses on this node's real micros. Results: /data/home/amartyavarma/ppm/runs/step0/step0_full.json (small json).
set -uo pipefail
C=/data/home/amartyavarma/ppm/code; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
source /data/home/amartyavarma/qwen-s2-imend/hera3/env.sh
export HF_HOME=/data/home/amartyavarma/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
REF=/scratch/amartyavarma/ppm/step0/hf_ref.pt
[ -f $REF ] || $PY $C/tests/hf_ref.py $REF 1024 bf16 || { echo "HF REF FAIL"; exit 1; }
[ -f ${REF%.pt}_fp32.pt ] || $PY $C/tests/hf_ref.py ${REF%.pt}_fp32.pt 1024 fp32 || { echo "HF REF fp32 FAIL"; exit 1; }
PPM_RUN=ppm-step0-full bash $C/run_local.sh tests/step0_full.py $REF
