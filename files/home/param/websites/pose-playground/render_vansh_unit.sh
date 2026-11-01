#!/bin/bash
# 10-26 (user: Vansh compare toggle; "20 steps not 50 and many many gpus"; main: the Vansh pair serves Param playground jobs when the Vansh
# queues are empty): a playground renderer pair of FLEX lanes (flex_lib.sh render_pair): vansh20 > vansh50 > param-pg, longest wait first,
# switching only between jobs. Logs: video/dmd-rt/logs/render-flex-pg<job>-g<gpu>.log; switches in genweb/logs/genweb-u-vansh-<job>.out.
: "${CANOPY_ROOT:?export CANOPY_ROOT=<cluster home that holds ppm/ venvs/ models/; see README>}"
bash $CANOPY_ROOT/ppm/aloe-posedup-live/tools/gpu_guard.sh "playground Vansh renderer" || { echo "GPU guard refused playground Vansh renderer; exiting"; exit 3; }   # 10-27 main: never slinky-4 GPUs 6-7 / SM < 1500 MHz for live / user-facing work
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES host $(hostname) role flex renderer pair (VANSH_PER=${VANSH_PER:-1}) $(date)"
source ${GENWEB:-$CANOPY_ROOT/ppm/aloe-posedup-live/genweb}/flex_lib.sh
render_pair
