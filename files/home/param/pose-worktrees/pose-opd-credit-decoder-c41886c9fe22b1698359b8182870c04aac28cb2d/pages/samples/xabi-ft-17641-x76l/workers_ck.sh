#!/bin/bash
# (10-22 23:10 PT) workers.sh with the PPM export as a parameter: PIN_CK (default the 17,641 export-200). Was: X76L (10-23) hot 17,641 pose server on the spares. Usage: workers.sh "<ppm GPU pairs, e.g. 0,1 2,3>" "<video GPUs, e.g. 4 5>"
# PPM workers are PINNED to FT-(b) aloe-v2pure-17641 export-200 (node-local copy, same path on every node; no hot-swap away from it).
# Video workers = same env as s15_workers.sh / s20 xvideo_persist.sbatch (9 sinks + 9 reference images). Stop: touch $B/ctl/STOP-<node>.
R=/data/home/amartyavarma/ppm/ppm-pose-server; H=/data/home/amartyavarma/video/amu-handover; V=/data/home/amartyavarma/video
FF=/data/home/amartyavarma/tools/ffmpeg-static/ffmpeg-7.0.2-amd64-static; SCR=/scratch/amartyavarma/ppm-pose-server; NODE=$(hostname -s)
B=/data/home/amartyavarma/ppm/samples/xabi-ft-17641-x76l; CK=${PIN_CK:-/scratch/amartyavarma/xabi-samples/w17641/export-200}
stop() { [ -e $B/ctl/STOP-$NODE ]; }
ppm() {  # $1 GPUs $2 tag
  while ! stop; do
    while [ ! -e $CK/COMPLETE ] || { [ "$CK" = /scratch/amartyavarma/xabi-samples/w17641/export-200 ] && [ ! -e $B/ctl/WEIGHTS_OK-$NODE ]; }; do stop && return; sleep 2; done
    ( source /data/home/amartyavarma/ppm/fastinfer/env_fast.sh; export CUDA_VISIBLE_DEVICES=$1 PPM_CK=$CK
      exec nice -n 5 /data/home/amartyavarma/venvs/qwen-s2/bin/python -u $R/code/ppm_worker.py --tag $2 ) >> $R/logs/$2.log 2>&1 &
    local p=$!
    while kill -0 $p 2>/dev/null; do stop && { kill $p; wait $p; break 2; }; sleep 2; done
    wait $p; echo "$(date -Is) ppm worker $2 exited ($?), restarting on the SAME export" >> $R/logs/$2.log; rm -f $R/control/next-$2.txt; sleep 3
  done
  rm -f $R/workers/$2.json
}
video() {  # $1 GPU
  local g=$1 T=video-$NODE-g$1-x76l C=$SCR/cache/video-$NODE-g$1
  if [ ! -d $C/triton ]; then mkdir -p $C; cp -r $H/cache/triton $C/triton; cp -r $H/cache/inductor $C/inductor; fi
  while ! stop; do
    CUDA_VISIBLE_DEVICES=$g PATH=$V/venv-distilled/bin:$FF:/usr/local/cuda/bin:$PATH TRITON_CACHE_DIR=$C/triton TORCHINDUCTOR_CACHE_DIR=$C/inductor \
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 VIDEO_AUTOTUNE=0 \
      $V/venv-distilled/bin/python -W ignore -u $R/code/video_worker.py --tag $T >> $R/logs/$T.log 2>&1 &
    local p=$!
    while kill -0 $p 2>/dev/null; do stop && { kill $p; wait $p; break 2; }; sleep 2; done
    wait $p; echo "$(date -Is) video worker $T exited, restarting" >> $R/logs/$T.log; sleep 10
  done
  rm -f $R/workers/$T.json
}
echo "$(TZ=America/Los_Angeles date "+%m-%d %T PT") $NODE start; ppm pairs [$1] video GPUs [$2] ckpt $CK"
nvidia-smi --query-gpu=index,memory.used,clocks.sm --format=csv,noheader
for pr in $1; do ppm $pr ppm-x76l-$NODE-g${pr/,/} & done
for g in $2; do video $g & done
wait
echo "$(TZ=America/Los_Angeles date '+%m-%d %T PT') $NODE stopped"
