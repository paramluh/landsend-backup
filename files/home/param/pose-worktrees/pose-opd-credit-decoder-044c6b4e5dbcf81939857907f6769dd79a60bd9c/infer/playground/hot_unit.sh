#!/bin/bash
# 10-26 (user: "N samples START together"; main): a playground GPU pair = ONE HOT monoplex worker (both GPUs, exactly unit.sbatch's mono worker:
# same env, worker.py, one run per process) + ONE Param render lane on the pair's second GPU (where the worker holds ~41 GB; the first holds
# ~52 GB). LANE_ROLE (default param-pg; none = no lane: the fallback layout) picks the lane's role; the lane is one render_flex.sh in its own process group, env isolated.
ROLE=mono; G=/data/home/amartyavarma/ppm/aloe-posedup-live/genweb
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES host $(hostname) role hot mono worker + 1 render lane (${LANE_ROLE:-param-pg}) $(date)"
IFS=, read -ra GP <<< "$CUDA_VISIBLE_DEVICES"; LG=${GP[1]}
[ "${LANE_ROLE:-param-pg}" != none ] && setsid env -i HOME=$HOME USER=$USER LOGNAME=$USER PATH=/usr/local/bin:/usr/bin:/bin LANG=${LANG:-C.UTF-8} SLURM_JOB_ID=$SLURM_JOB_ID \
  bash -c "while true; do bash $G/render_flex.sh $LG ${LANE_ROLE:-param-pg} pg$SLURM_JOB_ID; echo \"lane g$LG exited \$(date -u +%T)\"; sleep 10; done" &
source /data/home/amartyavarma/ppm/aloe-posedup-live/env_live.sh
export TRITON_CACHE_DIR=/scratch/amartyavarma/genweb/triton-cache MIMI_DIR=/data/home/amartyavarma/models/mimi FA3_DIR=/data/home/amartyavarma/models/qwen-s2/vllm-flash-attn3
mkdir -p $TRITON_CACHE_DIR
while true; do $PY -u $G/worker.py ${ROLE}$SLURM_JOB_ID >> $G/logs/worker-$ROLE-$SLURM_JOB_ID.log 2>&1; echo "exited rc=$? $(date)" >> $G/logs/worker-$ROLE-$SLURM_JOB_ID.log; sleep 10; done
