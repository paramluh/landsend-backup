#!/bin/bash
#SBATCH -p all -w slinky-15 -N1 -n1 -c 12 --mem=160G --gres=gpu:4 --gres-flags=disable-binding -t 3-00:00:00 -J ppm-pose-server
#SBATCH -o /data/home/amartyavarma/ppm/ppm-pose-server/logs/server-%j.out
# PPM 32B pose-FT server (persistent, model kept hot, resumable). Restart = sbatch this file again: workers re-scan requests/ and finish
# anything not DONE (claims of dead workers go stale after 45 s); identical requests return cached results.
#   1. HTTP front :8840 + Cloudflare quick tunnel (URL -> ROOT/URL)   2. one PPM worker on the first 2 GPUs (fast_ppm_srv, loaded once)
#   3. one video worker per remaining GPU (distilled pose-v3 pair, loaded + compiled once). Workers auto-restart; a PPM worker that exits 3
#      (hot-swap) restarts on control/next-<tag>.txt.
R=/data/home/amartyavarma/ppm/ppm-pose-server; SCR=/scratch/amartyavarma/ppm-pose-server; H=/data/home/amartyavarma/video/amu-handover
V=/data/home/amartyavarma/video; PORT=${PORT:-8840}; FF=/data/home/amartyavarma/tools/ffmpeg-static/ffmpeg-7.0.2-amd64-static
IFS=, read -ra ALL <<< "$CUDA_VISIBLE_DEVICES"
NODE=$(hostname)
echo "$(TZ=America/Los_Angeles date '+%F %T PT') $NODE job $SLURM_JOB_ID, slurm GPUs $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
for g in "${ALL[@]}"; do
  if grep -qE "^$NODE $g " /data/home/amartyavarma/ppm/RESERVED_GPUS.txt && ! grep -qE "^$NODE $g ppm-pose-server" /data/home/amartyavarma/ppm/RESERVED_GPUS.txt; then
    echo "WARNING: GPU $g is listed as reserved by someone else"; fi
  grep -qE "^$NODE $g ppm-pose-server" /data/home/amartyavarma/ppm/RESERVED_GPUS.txt || \
    echo "$NODE $g ppm-pose-server   # $(TZ=America/Los_Angeles date '+%m-%d %H:%M') PT PPM 32B pose-FT server job $SLURM_JOB_ID (ppm/ppm-pose-server/code/)" >> /data/home/amartyavarma/ppm/RESERVED_GPUS.txt
done
echo "$SLURM_JOB_ID $NODE ${ALL[*]}" > $R/JOB
mkdir -p $SCR/cache $R/logs $R/requests $R/workers $R/control
( while true; do python3 -u $R/code/server.py --port $PORT >> $R/logs/front.log 2>&1; sleep 2; done ) &
( sleep 3; while true; do /home/amartyavarma/bin/cloudflared tunnel --no-autoupdate --url http://127.0.0.1:$PORT 2>&1 | tee -a $R/logs/tunnel.log | \
    while read -r l; do u=$(echo "$l" | grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com'); [ -n "$u" ] && echo "$u" > $R/URL; done; sleep 5; done ) &
echo "http://$NODE:$PORT" > $R/URL_INTERNAL

ppm_loop() {   # $1 = "g,g", $2 = tag
  while true; do
    CK=$(cat $R/control/next-$2.txt 2>/dev/null || python3 -c "import sys; sys.path.insert(0, '$R/code'); import srv_common as C; print(C.default_ckpt())")
    rm -f $R/control/next-$2.txt
    ( source /data/home/amartyavarma/ppm/fastinfer/env_fast.sh; export CUDA_VISIBLE_DEVICES=$1 PPM_CK=$CK
      exec /data/home/amartyavarma/venvs/qwen-s2/bin/python -u $R/code/ppm_worker.py --tag $2 ) >> $R/logs/$2.log 2>&1
    echo "$(date -Is) ppm worker $2 exited ($?), restarting" >> $R/logs/$2.log; sleep 3
  done
}
video_loop() {  # $1 = GPU
  local C=$SCR/cache/video-$NODE-g$1
  if [ ! -d $C ]; then mkdir -p $C; cp -r $H/cache/triton $C/triton; cp -r $H/cache/inductor $C/inductor; fi
  while true; do
    CUDA_VISIBLE_DEVICES=$1 PATH=$V/venv-distilled/bin:$FF:/usr/local/cuda/bin:$PATH TRITON_CACHE_DIR=$C/triton TORCHINDUCTOR_CACHE_DIR=$C/inductor \
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 VIDEO_AUTOTUNE=0 \
      $V/venv-distilled/bin/python -W ignore -u $R/code/video_worker.py --tag video-$NODE-g$1 >> $R/logs/video-$NODE-g$1.log 2>&1
    echo "$(date -Is) video worker g$1 exited, restarting" >> $R/logs/video-$NODE-g$1.log; sleep 5
  done
}
ppm_loop "${ALL[0]},${ALL[1]}" ppm-$NODE-g${ALL[0]}${ALL[1]} &
for g in "${ALL[@]:2}"; do video_loop $g & done
wait
