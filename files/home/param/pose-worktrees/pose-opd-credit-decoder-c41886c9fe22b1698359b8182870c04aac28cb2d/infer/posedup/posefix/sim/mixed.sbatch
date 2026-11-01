#!/bin/bash
#SBATCH -J posefix-mix
#SBATCH -x slinky-9
#SBATCH -c 16
#SBATCH -t 6:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-posedup-live/posefix/sim/logs/%x-%j.out
# usage: sbatch -w <node> --gres=gpu:2K --mem=.. mixed.sbatch <spec> [<spec> ...]; spec = kind:ck:runs.json:outdir (kind = dup | mono), one
# instance per spec on GPU pairs in allocation order. slinky-4: aborts if the allocation includes the throttled GPUs 6 / 7.
source /data/home/amartyavarma/ppm/aloe-posedup-live/env_live.sh
export TRITON_CACHE_DIR=/scratch/amartyavarma/posefix/triton-cache; mkdir -p $TRITON_CACHE_DIR
export MIMI_DIR=/data/home/amartyavarma/models/mimi FA3_DIR=/data/home/amartyavarma/models/qwen-s2/vllm-flash-attn3
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES host $(hostname)"
if [ "$(hostname)" = slinky-4 ] && [ -z "$ALLOW_THROTTLED" ]; then case ",$CUDA_VISIBLE_DEVICES," in *,6,*|*,7,*) echo "ABORT: got throttled GPU 6/7"; exit 3;; esac; fi
IFS=, read -ra G <<< "$CUDA_VISIBLE_DEVICES"
i=0; for spec in "$@"; do
  if [ "$spec" = render ]; then   # videort's render_multi.sh: one render_service instance per GPU of this pair (shared dirs, atomic claims)
    bash /data/home/amartyavarma/video/dmd-rt/rt/render_multi.sh ${G[$((2*i))]},${G[$((2*i+1))]} posefix-$SLURM_JOB_ID-$i &
    i=$((i+1)); continue; fi
  IFS=: read -r kind ck runs out <<< "$spec"; mkdir -p $out
  scr=/data/home/amartyavarma/ppm/aloe-posedup-live/posefix/sim/replay.py; [ "$kind" = mono ] && scr=/data/home/amartyavarma/ppm/aloe-posedup-live/posefix/sim/mono.py
  CUDA_VISIBLE_DEVICES=${G[$((2*i))]},${G[$((2*i+1))]} $PY -u $scr $ck $runs $out > $out/inst-$(basename $runs .json)-$SLURM_JOB_ID.log 2>&1 &
  i=$((i+1)); done
wait
