#!/bin/bash
# run prep_edits.py on one slinky-18 GPU (shared with an idle hot worker; needs a few GB)
V=/data/home/amartyavarma/video
export CUDA_VISIBLE_DEVICES=${GPU:-0} PATH=$V/venv-distilled/bin:/data/home/amartyavarma/tools/ffmpeg-static/ffmpeg-7.0.2-amd64-static:/usr/local/cuda/bin:$PATH \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
exec nice -n 10 $V/venv-distilled/bin/python -W ignore -u /data/home/amartyavarma/ppm/gaze-drift/render/code/prep_edits.py "$@"
