#!/bin/bash
# 10-28 main (user: "set up backend server for both when they are ready"): ONE hot monoplex worker for an extra weights id (genweb/weights/<W>.json)
# on 2 GPUs: worker.py with GENWEB_WEIGHTS=$MONO_W claims only runs whose request carries mono_weights $MONO_W. Same env as hot_unit.sh's worker.
#   sbatch -J genweb-$MONO_W -w <node> -x slinky-9 --gres=gpu:2 -c 8 --mem=120G -t 3-0 --export=ALL,MONO_W=<id> mono_unit.sh
: "${CANOPY_ROOT:?export CANOPY_ROOT=<cluster home that holds ppm/ venvs/ models/; see README>}"
bash $CANOPY_ROOT/ppm/aloe-posedup-live/tools/gpu_guard.sh "playground monoplex $MONO_W" || { echo "GPU guard refused; exiting"; exit 3; }
G=${GENWEB:-$CANOPY_ROOT/ppm/aloe-posedup-live/genweb}
# 10-28: the unit may start before its export is verified: it holds its GPUs and waits for genweb/weights/$MONO_W.json (installed on verify)
[ "$MONO_W" = base ] || { until [ -f $G/weights/$MONO_W.json ]; do sleep 5; done; echo "weights config present $(date)"; }   # base = MONO_CK.json (the 32B)
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES host $(hostname) role monoplex worker $MONO_W $(date)"
source $CANOPY_ROOT/ppm/aloe-posedup-live/env_live.sh
export TRITON_CACHE_DIR=/scratch/$USER/genweb/triton-cache MIMI_DIR=$CANOPY_ROOT/models/mimi FA3_DIR=$CANOPY_ROOT/models/qwen-s2/vllm-flash-attn3
export GENWEB_WEIGHTS=$MONO_W
mkdir -p $TRITON_CACHE_DIR
while true; do $PY -u $G/worker.py $MONO_W$SLURM_JOB_ID >> $G/logs/worker-$MONO_W-$SLURM_JOB_ID.log 2>&1; echo "exited rc=$? $(date)" >> $G/logs/worker-$MONO_W-$SLURM_JOB_ID.log; sleep 10; done
