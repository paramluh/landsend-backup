#!/bin/bash
#SBATCH -x slinky-9
#SBATCH -c 8
#SBATCH -t 24:00:00
#SBATCH -o logs/%x-%j.out
# ONE GPU PAIR of the playground, role = $1: dup (duplex worker) | mono (monoplex worker) | render (2 render_service instances) | vansh (2 Vansh-weights renderers).
# A worker role landing on slinky-4's throttled GPUs 6 / 7 (345 MHz) becomes a renderer instead (no slow user runs).
: "${CANOPY_ROOT:?export CANOPY_ROOT=<cluster home that holds ppm/ venvs/ models/; see README>}"
bash $CANOPY_ROOT/ppm/aloe-posedup-live/tools/gpu_guard.sh "playground unit" || { echo "GPU guard refused playground unit; exiting"; exit 3; }   # 10-27 main: never slinky-4 GPUs 6-7 / SM < 1500 MHz for live / user-facing work
ROLE=${1:-mono}; G=${GENWEB:-$CANOPY_ROOT/ppm/aloe-posedup-live/genweb}
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES host $(hostname) role $ROLE $(date)"
# 10-26 user (fixed roles): vanshonly = Vansh overflow lanes only (the throttled slinky-4 6-7 included: never Param or generation there);
# paramonly = a dedicated Param render pair (fallback layout)
if [ "$ROLE" = vanshonly ]; then source $G/flex_lib.sh; vansh_pair; exit; fi
if [ "$ROLE" = paramonly ]; then source $G/flex_lib.sh; param_pair; exit; fi
if [ "$(hostname)" = slinky-4 ]; then case ",$CUDA_VISIBLE_DEVICES," in *,6,*|*,7,*) [ "$ROLE" != render ] && { echo "throttled GPUs -> render"; ROLE=render; };; esac; fi
# 10-26 dupui: the duplex worker loads the NODE-LOCAL export-498 DCP (no /data copy; on slinky-18 + slinky-10 only): elsewhere -> mono (no crash loop)
if [ "$ROLE" = dup ] && [ ! -d "${GENWEB_DUP_CK:-/scratch/$USER/aloe-posedup-live/export-498-dcp}" ]; then echo "no export-498 DCP on $(hostname) -> mono"; ROLE=mono; fi
if [ "$ROLE" = render ]; then exec bash $G/render2.sbatch; fi
if [ "$ROLE" = vansh ]; then exec bash $G/render_vansh_unit.sh; fi      # 10-26: Vansh-weights renderers (videort render_vansh.sh)
source $CANOPY_ROOT/ppm/aloe-posedup-live/env_live.sh
export TRITON_CACHE_DIR=/scratch/$USER/genweb/triton-cache MIMI_DIR=$CANOPY_ROOT/models/mimi FA3_DIR=$CANOPY_ROOT/models/qwen-s2/vllm-flash-attn3
mkdir -p $TRITON_CACHE_DIR
[ "$ROLE" = dup ] && export TRITON_CACHE_AUTOTUNING=1 TRITON_CACHE_DIR=${GENWEB:-$CANOPY_ROOT/ppm/aloe-posedup-live/genweb}/triton-shared   # 10-25: shared autotune cache = the same kernel picks in every duplex worker (reproducible links)
W=$G/worker.py; [ "$ROLE" = dup ] && W=$G/worker_dup.py
if [ "${GENWEB_FLEX:-1}" = 0 ]; then
while true; do $PY -u $W ${ROLE}$SLURM_JOB_ID >> $G/logs/worker-$ROLE-$SLURM_JOB_ID.log 2>&1; echo "exited rc=$? $(date)" >> $G/logs/worker-$ROLE-$SLURM_JOB_ID.log; sleep 10; done
fi
# 10-26 user ("many many gpus" for Vansh): FLEX worker pair (flex_lib.sh worker_pair). The worker runs exactly as above (same env, same
# command); while it is idle, nothing of its model is queued and a Vansh render has waited >= 10 s, it is stopped and the pair renders Vansh
# (one render_service per GPU, env isolated); a queued run of its model, or 60 s without Vansh work, brings the worker back after the
# current renders finish. GENWEB_FLEX=0 = the plain worker loop above.
WID=${ROLE}$SLURM_JOB_ID
source $G/flex_lib.sh
worker_pair $ROLE bash -c "exec $PY -u $W $WID >> $G/logs/worker-$ROLE-$SLURM_JOB_ID.log 2>&1"
