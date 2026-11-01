#!/bin/bash
# 10-26 (main: >= 2 hot monoplex workers): a HOT mono worker on GPUs lent by videort INSIDE its job 29805 (slinky-10 GPUs 3,6). Slurm can't
# shrink a running job, so this runs as an overlap step:
#   srun --overlap --jobid 29805 -w slinky-10 -x slinky-9 -n 1 -c 8 --mem=120G --gres=gpu:6 --cpu-bind=none \
#        env CUDA_VISIBLE_DEVICES=3,6 bash genweb/mono_guest.sh
# Same env + command as unit.sbatch's plain mono worker (GENWEB_FLEX=0 path): worker.py, worker id mono29805g36 (the reaper maps 29805 to the
# job). Restart loop. Never flexes. Stop = scancel the step (scontrol listpids), never the job.
G=/data/home/amartyavarma/ppm/aloe-posedup-live/genweb
source /data/home/amartyavarma/ppm/aloe-posedup-live/env_live.sh
export TRITON_CACHE_DIR=/scratch/amartyavarma/genweb/triton-cache MIMI_DIR=/data/home/amartyavarma/models/mimi FA3_DIR=/data/home/amartyavarma/models/qwen-s2/vllm-flash-attn3
mkdir -p $TRITON_CACHE_DIR
echo "mono guest on $(hostname) CVD=$CUDA_VISIBLE_DEVICES $(date)"
while true; do $PY -u $G/worker.py mono29805g36 >> $G/logs/worker-mono-29805g36.log 2>&1; echo "exited rc=$? $(date)" >> $G/logs/worker-mono-29805g36.log; sleep 10; done
