#!/bin/bash
#SBATCH -J genweb-render
#SBATCH -x slinky-9
#SBATCH -c 8
#SBATCH -t 24:00:00
#SBATCH -o logs/%x-%j.out
# videort's render service for the playground: one instance per GPU, playground jobs genweb/render/jobs -> genweb/render/out (atomic claims)
: "${CANOPY_ROOT:?export CANOPY_ROOT=<cluster home that holds ppm/ venvs/ models/; see README>}"
bash $CANOPY_ROOT/ppm/aloe-posedup-live/tools/gpu_guard.sh "playground renderer" || { echo "GPU guard refused playground renderer; exiting"; exit 3; }   # 10-27 main: never slinky-4 GPUs 6-7 / SM < 1500 MHz for live / user-facing work
export VRT_VAE=$(grep -oP "^VRT_VAE=\K\S+" $CANOPY_ROOT/video/dmd-rt/demo/backend.env 2>/dev/null || echo fp32)   # 10-26 main: the avatar VAE in videort's contract precision (backend.env), never render_service.py's bf16 default
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=/scratch/$USER/videort/inductor TRITON_CACHE_DIR=/scratch/$USER/videort/triton
export PATH=$CANOPY_ROOT/tools/ffmpeg-static/ffmpeg-7.0.2-amd64-static:$PATH
export VRT_PEOPLE=$(ls $CANOPY_ROOT/video/demo-data/prompts | paste -sd,) VRT_CACHE_INPUTS=0
G=${GENWEB:-$CANOPY_ROOT/ppm/aloe-posedup-live/genweb}
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES host $(hostname) $(date)"
IFS=, read -ra GP <<< "$CUDA_VISIBLE_DEVICES"; k=0
for g in "${GP[@]}"; do
  ( while true; do CUDA_VISIBLE_DEVICES=$g RS_REVERSE=$((k % 2)) $CANOPY_ROOT/video/venv-distilled/bin/python -W ignore \
      $CANOPY_ROOT/video/dmd-rt/rt/render_service.py 01_ab63d745 $G/render 9999 pg:$G/render/jobs:$G/render/out >> $G/logs/render-$SLURM_JOB_ID-g$g.log 2>&1
      echo "render exited rc=$? $(date)" >> $G/logs/render-$SLURM_JOB_ID-g$g.log; sleep 10; done ) &
  k=$((k + 1))
done
wait
