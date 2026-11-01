#!/bin/bash
#SBATCH -J genweb-s4
#SBATCH -x slinky-9
#SBATCH -c 32
#SBATCH -t 24:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-posedup-live/genweb/logs/%x-%j.out
# slinky-4, all 8 GPUs held: 3 playground workers on GPUs 0-5 (pairs) + 2 render_service instances on the throttled 6 and 7 (extra throughput only)
source /data/home/amartyavarma/ppm/aloe-posedup-live/env_live.sh
export TRITON_CACHE_DIR=/scratch/amartyavarma/genweb/triton-cache MIMI_DIR=/data/home/amartyavarma/models/mimi FA3_DIR=/data/home/amartyavarma/models/qwen-s2/vllm-flash-attn3
mkdir -p $TRITON_CACHE_DIR
G=/data/home/amartyavarma/ppm/aloe-posedup-live/genweb
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES host $(hostname) $(date)"
IFS=, read -ra GP <<< "$CUDA_VISIBLE_DEVICES"
[ ${#GP[@]} = 8 ] || { echo "ABORT: need all 8 GPUs (got ${#GP[@]})"; exit 1; }
for i in 0 1 2; do
  ( while true; do CUDA_VISIBLE_DEVICES=${GP[$((2*i))]},${GP[$((2*i+1))]} $PY -u $G/worker.py s4w$SLURM_JOB_ID-$i >> $G/logs/worker-s4-$SLURM_JOB_ID-$i.log 2>&1
      echo "worker $i exited rc=$? $(date)" >> $G/logs/worker-s4-$SLURM_JOB_ID-$i.log; sleep 10; done ) &
done
export VRT_VAE=$(grep -oP "^VRT_VAE=\K\S+" /data/home/amartyavarma/video/dmd-rt/demo/backend.env 2>/dev/null || echo fp32)   # 10-26 main: the avatar VAE in videort's contract precision (backend.env), never render_service.py's bf16 default
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=/scratch/amartyavarma/videort/inductor RTRITON=/scratch/amartyavarma/videort/triton
export PATH=/data/home/amartyavarma/tools/ffmpeg-static/ffmpeg-7.0.2-amd64-static:$PATH
export VRT_PEOPLE=$(ls /data/home/amartyavarma/video/demo-data/prompts | paste -sd,) VRT_CACHE_INPUTS=0
for k in 6 7; do
  ( while true; do CUDA_VISIBLE_DEVICES=${GP[$k]} TRITON_CACHE_DIR=$RTRITON RS_REVERSE=$((k % 2)) /data/home/amartyavarma/video/venv-distilled/bin/python -W ignore \
      /data/home/amartyavarma/video/dmd-rt/rt/render_service.py 01_ab63d745 $G/render 9999 pg:$G/render/jobs:$G/render/out >> $G/logs/render-s4-$SLURM_JOB_ID-g$k.log 2>&1
      echo "render exited rc=$? $(date)" >> $G/logs/render-s4-$SLURM_JOB_ID-g$k.log; sleep 10; done ) &
done
wait
