#!/bin/bash
# pose5 C: RESOLVED-cfg diff of the pose5 env (+ the launch overrides) vs job 27653's logged cfg (its rank-0 "cfg {...}" line), with the trainer
# of CODE (default the s2p copy). CPU only, run on a spare via srun (never the login node). = code-811/fresh/append/tools/cfgdiff.sh, re-pointed.
# usage: cfgdiff_pose5.sh <out.txt> [CODE dir] [KEY=VAL overrides...]
set -u
OUT=$1; CODE=${2:-/data/home/amartyavarma/ppm/pose5/code-811-s2p}; shift 2 2>/dev/null
ENVF=/data/home/amartyavarma/ppm/pose5/prod_811_fresh_append5_s2p.env; LOG=/data/home/amartyavarma/ppm/logs/ppm-27653.log
PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
for v in $(env | grep -oE '^PPM_[A-Za-z0-9_]*'); do unset $v; done
set -a; source "$ENVF"; for kv in "$@"; do export "$kv"; done; set +a
export RANK=0 WORLD_SIZE=128 LOCAL_RANK=0 LOCAL_WORLD_SIZE=8 PPM_RUN=ppm-v1-jPOSE5 PPM_RUN_BASE=ppm-v1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1
exec $PY $CODE/test811/cfg_diff.py "$LOG" "$OUT"
