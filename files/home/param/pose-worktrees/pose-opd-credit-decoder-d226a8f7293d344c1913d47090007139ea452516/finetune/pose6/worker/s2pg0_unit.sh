#!/bin/bash
# pose6 A3 (NOT LAUNCHED; run only on main's word, see README "How to hot-swap later"): ONE hot monoplex worker serving an s2p cb0 + G0COND
# checkpoint on 2 GPUs = genweb/rosie_unit.sh line for line, except the worker file (pose6/worker/worker_s2pg0.py), GENWEB_CK = $S2PG0_CK
# (required) and GENWEB_WEIGHTS=s2pg0: it claims ONLY runs whose request carries mono_weights "s2pg0" (base / Rosie runs never), never
# rewrites SCHEMA / OPTIONS. Worker id = s2pgw<job id>: no digit right before the job id (the server reaper takes the FIRST 5-digit run of the id as the job; "s2pg0<id>" would read 0+4 digits).
#   sbatch -J genweb-s2pg0 -w <node> -x slinky-9 --gres=gpu:2 -c 8 --mem=120G -t 7-0 \
#          -o /data/home/amartyavarma/ppm/aloe-posedup-live/genweb/logs/%x-%j.out \
#          --export=ALL,S2PG0_CK=<export dir> /data/home/amartyavarma/ppm/pose6/worker/s2pg0_unit.sh
#SBATCH -J genweb-s2pg0
#SBATCH -x slinky-9
G=/data/home/amartyavarma/ppm/aloe-posedup-live/genweb
W=/data/home/amartyavarma/ppm/pose6/worker/worker_s2pg0.py
: "${S2PG0_CK:?set S2PG0_CK to the s2p cb0 + G0COND checkpoint}"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES host $(hostname) role S2PG0 monoplex worker ck $S2PG0_CK $(date)"
source /data/home/amartyavarma/ppm/aloe-posedup-live/env_live.sh
export TRITON_CACHE_DIR=/scratch/amartyavarma/genweb/triton-cache MIMI_DIR=/data/home/amartyavarma/models/mimi FA3_DIR=/data/home/amartyavarma/models/qwen-s2/vllm-flash-attn3
export GENWEB_CK=$S2PG0_CK GENWEB_WEIGHTS=s2pg0
unset PPM_S2P PPM_G0COND                                                 # the engine follows the checkpoint's keys
mkdir -p $TRITON_CACHE_DIR
while true; do $PY -u $W s2pgw$SLURM_JOB_ID >> $G/logs/worker-s2pg0-$SLURM_JOB_ID.log 2>&1; echo "exited rc=$? $(date)" >> $G/logs/worker-s2pg0-$SLURM_JOB_ID.log; sleep 10; done
