#!/bin/bash
# 10-26 (main): ONE render_service in the foreground (killable), in one role, on one GPU (or one GPU pair). The env is copied EXACTLY from
# videort's rt/render_multi.sh (Param SeqEngine) / rt/render_vansh.sh (Vansh SeqEngine); only the spec list differs (keep in sync):
#   param-pg  playground Param jobs only: pg:render/jobs -> render/out; RS_SPECS names no file, so no extra (posefix) specs
#   posefix   posefix's low-priority re-renders only (SPECS_seq.txt), never playground jobs: for the throttled slinky-4 GPUs 6 / 7
#   vansh20   (user 10-26: 20 steps, seq/vansh_seq20.yaml) Vansh weights: pg:render/jobs_vansh -> render/out
#   vansh50   (his aloe.yaml, 50 steps; 'vansh' = alias) Vansh weights: pg:render/jobs_vansh50 -> render/out
#   render_flex.sh <gpu | gpu,gpu> <role> <tag>      (a GPU pair = one instance: sampler | resolver + pose + decode, VRT_SEQ_FAST res)
: "${CANOPY_ROOT:?export CANOPY_ROOT=<cluster home that holds ppm/ venvs/ models/; see README>}"
g=$1; role=$2; tag=$3
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=/scratch/$USER/videort/seq-inductor TRITON_CACHE_DIR=/scratch/$USER/videort/seq-triton
export PATH=$CANOPY_ROOT/tools/ffmpeg-static/ffmpeg-7.0.2-amd64-static:$PATH
export VRT_SEQ_FAST=commit_after,audio_graph,audio_once,pose_proc   # 10-26 videort: exact speed options (bit-identical, seq/PROOF.md)
case $g in *,*) export VRT_SEQ_FAST=res,$VRT_SEQ_FAST;; esac          # a pair: + the resolver on its own GPU (videort render_vansh.sh PER=2)
P=$CANOPY_ROOT/ppm/aloe-posedup-live/posefix
G=${GENWEB:-$CANOPY_ROOT/ppm/aloe-posedup-live/genweb}/render
L=$CANOPY_ROOT/video/dmd-rt/logs
[ "$role" = vansh ] && role=vansh50
case $role in
  param-pg) export RS_ENGINE=seq RS_SPECS=NO_EXTRA_SPECS_aloedemo.txt VRT_SEQ_WEIGHTS=${VRT_SEQ_WEIGHTS:-ema}; specs="pg:$G/jobs:$G/out";;
  posefix)  export RS_ENGINE=seq RS_SPECS=SPECS_seq.txt VRT_SEQ_WEIGHTS=${VRT_SEQ_WEIGHTS:-ema}; specs="";;
  param-test) export RS_ENGINE=seq RS_SPECS=NO_EXTRA_SPECS_aloedemo.txt VRT_SEQ_WEIGHTS=${VRT_SEQ_WEIGHTS:-ema}; mkdir -p $G/jobs_test $G/out_test; specs="pg:$G/jobs_test:$G/out_test";;   # 10-26 co-location memory test only
  vansh20)  export RS_ENGINE=seq VRT_SEQ_MODEL=vansh20 RS_SPECS=SPECS_vansh.txt; mkdir -p $G/jobs_vansh; specs="pg:$G/jobs_vansh:$G/out";;
  vansh50)  export RS_ENGINE=seq VRT_SEQ_MODEL=vansh50 RS_SPECS=SPECS_vansh.txt; mkdir -p $G/jobs_vansh50; specs="pg:$G/jobs_vansh50:$G/out";;
  *) echo "role must be param-pg | posefix | vansh20 | vansh50"; exit 2;;
esac
gl=${g//,/_}
echo "$(date -u +%T) render_flex g$gl role $role" >> $L/render-flex-$tag-g$gl.log
CUDA_VISIBLE_DEVICES=$g exec $CANOPY_ROOT/video/venv-distilled/bin/python -W ignore \
  $CANOPY_ROOT/video/dmd-rt/rt/render_service.py 01_ab63d745 $P/clips 9999 $specs >> $L/render-flex-$tag-g$gl.log 2>&1
