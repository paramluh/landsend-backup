#!/bin/bash
#SBATCH -J posefix-sim3
#SBATCH -x slinky-9
#SBATCH --gres=gpu:6
#SBATCH -c 18
#SBATCH --mem=420G
#SBATCH -t 6:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-posedup-live/posefix/sim/logs/%x-%j.out
# usage: sbatch -w slinky-4 sim3.sbatch <ck> <out dir> <runs0.json> <runs1.json> <runs2.json>: 3 simulator instances x 2 GPUs.
# slinky-4: physical GPUs 6-7 are throttled -> abort if the allocation includes them.
source /data/home/amartyavarma/ppm/aloe-posedup-live/env_live.sh
export TRITON_CACHE_DIR=/scratch/amartyavarma/posefix/triton-cache; mkdir -p $TRITON_CACHE_DIR
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"; nvidia-smi --query-gpu=index,pci.bus_id,clocks.max.sm --format=csv,noheader
if [ "$(hostname)" = slinky-4 ]; then case ",$CUDA_VISIBLE_DEVICES," in *,6,*|*,7,*) echo "ABORT: got throttled GPU 6/7"; exit 3;; esac; fi
CK=$1; OUT=$2; shift 2
IFS=, read -ra G <<< "$CUDA_VISIBLE_DEVICES"
i=0; for R in "$@"; do
  CUDA_VISIBLE_DEVICES=${G[$((2*i))]},${G[$((2*i+1))]} $PY -u /data/home/amartyavarma/ppm/aloe-posedup-live/posefix/sim/replay.py $CK $R $OUT > $OUT/inst$i.log 2>&1 &
  i=$((i+1)); done
wait
