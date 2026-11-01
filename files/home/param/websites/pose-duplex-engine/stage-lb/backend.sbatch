#!/bin/bash
#SBATCH -N1 -n1 -c 8
#SBATCH --gres=gpu:2
#SBATCH --mem=100G
#SBATCH -t 3-00:00:00
# EXTRA BACKEND for the aloe-posedup-live load balancer (10-25): the LIVE server.py / live_ppm_pose.py, same env as demo.sbatch's server line
# (env_live.sh: PPMLIVE_DEFAULTS=nopersona; token file; CK = node-local export-498 DCP), its own port + state dir, bound 0.0.0.0 so the
# slinky-10 front proxy can route to it (token-checked). NO proxy, NO tunnel. Waits for a VERIFIED model copy (DL_VERIFY.json no bad / SWAP_OK).
#   sbatch -J posedup-lb-<k> -w <host> -x slinky-9 -o .../stage-lb/logs/backend-<k>-%j.out --export=ALL,BACK=<port>,INST=<k> backend.sbatch
# Restart only this backend: kill its server.py python PID (loop restarts it in 10 s).
bash /data/home/amartyavarma/ppm/aloe-posedup-live/tools/gpu_guard.sh "pose live instance" || { echo "GPU guard refused pose live instance; exiting"; exit 3; }   # 10-27 main: never slinky-4 GPUs 6-7 / SM < 1500 MHz for live / user-facing work
set -u
H=/data/home/amartyavarma/ppm/aloe-posedup-live
CK=${CK:-/scratch/amartyavarma/aloe-posedup-live/export-498-dcp}
export PPMLIVE_SWAP_TO=${PPMLIVE_SWAP_TO:-$CK}
export PPMLIVE_STATE_DIR=$H/stage-lb/inst$INST PPMLIVE_BIND=0.0.0.0
TOK=$HOME/.s4live_token
source $H/env_live.sh
hostname; scontrol show job -d $SLURM_JOB_ID | grep -o 'GRES=[^ ]*'
if [ "$(hostname)" = slinky-4 ] && scontrol show job -d $SLURM_JOB_ID | grep -o 'IDX:[0-9,-]*' | grep -qE '[67]'; then echo "ABORT: slinky-4 GPU 6/7 (throttled) allocated"; exit 1; fi
for i in $(seq 720); do
  if [ -f $CK/COMPLETE ] && { python3 -c "import json,sys; r=json.load(open('$CK/DL_VERIFY.json')); sys.exit(0 if r['n']>0 and not r['bad'] else 1)" 2>/dev/null || [ -f $CK/SWAP_OK ]; }; then break; fi
  [ $((i % 12)) = 1 ] && echo "waiting for a verified $CK $(date)"; sleep 5
done
[ -f $CK/COMPLETE ] || { echo "ABORT: no verified $CK after 60 min"; exit 1; }
[ -s "$TOK" ] && [ "$(stat -c %a "$TOK")" = 600 ] || { echo "ABORT: token file"; exit 1; }
mkdir -p $PPMLIVE_STATE_DIR/sessions
nvidia-smi --query-gpu=index,name,memory.used,clocks.sm,temperature.gpu --format=csv,noheader
echo "CVD=${CUDA_VISIBLE_DEVICES:-} ck $CK port $BACK state $PPMLIVE_STATE_DIR defaults $PPMLIVE_DEFAULTS bind $PPMLIVE_BIND $(date)"
while true; do
  $PY -u $H/server.py --ck "$CK" --port $BACK --token-file $TOK
  echo "server exited rc=$? $(date)"; sleep 10
done
