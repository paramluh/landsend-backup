#!/bin/bash
# resolved-cfg diff of <env> vs a logged run (default: job 27114's cfg line): test811/cfg_diff.py under the env, rank 0, CPU only.
# usage (on a compute node): cfgdiff.sh <env> <out.txt> [ppm log] [KEY=VAL overrides...]
C=/data/home/amartyavarma/ppm/code-811; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
ENVF=$1; OUT=$2; LOG=${3:-/data/home/amartyavarma/ppm/logs/ppm-27114.log}; shift 3 2>/dev/null
for v in $(env | grep -oE '^PPM_[A-Za-z0-9_]*'); do unset $v; done
set -a; source "$ENVF"; for kv in "$@"; do export "$kv"; done; set +a
export RANK=0 WORLD_SIZE=128 LOCAL_RANK=0 LOCAL_WORLD_SIZE=8 PPM_RUN=ppm-v1-j27114 PPM_RUN_BASE=ppm-v1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1
exec $PY $C/test811/cfg_diff.py "$LOG" "$OUT"
