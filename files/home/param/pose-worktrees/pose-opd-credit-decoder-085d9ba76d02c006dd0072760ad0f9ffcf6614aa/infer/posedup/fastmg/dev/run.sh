#!/bin/bash
# run a python script inside a dev allocation with the live env: run.sh <gpus> <script> [args...]
#   <gpus> = "job" (the allocation's own CUDA_VISIBLE_DEVICES), "job:i,j" (positions within it), or physical ids (slinky-4 hold job only)
source /data/home/amartyavarma/ppm/aloe-posedup-live/env_live.sh
G=$1; shift
if [ "$G" = job ]; then :
elif [ "${G#job:}" != "$G" ]; then IFS=, read -ra A <<< "$CUDA_VISIBLE_DEVICES"; IFS=, read -ra I <<< "${G#job:}"; S=""; for i in "${I[@]}"; do S="$S${S:+,}${A[$i]}"; done; export CUDA_VISIBLE_DEVICES=$S
else export CUDA_VISIBLE_DEVICES=$G; fi
case "$(hostname)" in slinky-18) for g in ${CUDA_VISIBLE_DEVICES//,/ }; do [ $g -ge 4 ] || { echo "REFUSE: GPU $g on slinky-18 belongs to the backends agent"; exit 3; }; done;; slinky-4) for g in ${CUDA_VISIBLE_DEVICES//,/ }; do [ $g -le 5 ] || { echo "REFUSE: slinky-4 GPU $g is throttled"; exit 3; }; done;; esac
export TRITON_CACHE_DIR=/scratch/amartyavarma/triton-cache-fastmg
mkdir -p $TRITON_CACHE_DIR
cd /data/home/amartyavarma/ppm/aloe-posedup-live/fastmg/dev
echo "[run] $(hostname) CVD=$CUDA_VISIBLE_DEVICES $*" >&2
exec $PY -u "$@"
