#!/bin/bash
# xabi-ft-17352-17641 samples (10-22): on slinky-15 (holder job), 3 PPM pose-server workers, each PINNED to one export (no hot-swap):
#   GPUs 0,1 -> FT-(a) 17352 (node-local copy), 2,3 -> FT-(b) 17641 (node-local copy), 4,5 -> 16610 (/data export-200)
# + one video worker per GPU 6, 7 (same env as s20/code/xvideo_persist.sbatch). Stop: touch $B/ctl/STOP-s15.
R=/data/home/amartyavarma/ppm/ppm-pose-server; H=/data/home/amartyavarma/video/amu-handover; V=/data/home/amartyavarma/video
FF=/data/home/amartyavarma/tools/ffmpeg-static/ffmpeg-7.0.2-amd64-static; SCR=/scratch/amartyavarma/ppm-pose-server; NODE=$(hostname)
B=/data/home/amartyavarma/ppm/samples/xabi-ft-17352-17641
stop() { [ -e $B/ctl/STOP-s15 ]; }
ppm() {  # $1 GPUs $2 ckpt $3 tag
  while ! stop; do
    ( source /data/home/amartyavarma/ppm/fastinfer/env_fast.sh; export CUDA_VISIBLE_DEVICES=$1 PPM_CK=$2
      exec nice -n 5 /data/home/amartyavarma/venvs/qwen-s2/bin/python -u $R/code/ppm_worker.py --tag $3 ) >> $R/logs/$3.log 2>&1 &
    local p=$!
    while kill -0 $p 2>/dev/null; do stop && { kill $p; wait $p; break 2; }; sleep 5; done
    wait $p; echo "$(date -Is) ppm worker $3 exited ($?), restarting on the SAME export" >> $R/logs/$3.log; rm -f $R/control/next-$3.txt; sleep 3
  done
  rm -f $R/workers/$3.json
}
video() {  # $1 GPU
  local g=$1 T=video-$NODE-g$1-xs C=$SCR/cache/video-$NODE-g$1
  if [ ! -d $C ]; then mkdir -p $C; cp -r $H/cache/triton $C/triton; cp -r $H/cache/inductor $C/inductor; fi
  while ! stop; do
    CUDA_VISIBLE_DEVICES=$g PATH=$V/venv-distilled/bin:$FF:/usr/local/cuda/bin:$PATH TRITON_CACHE_DIR=$C/triton TORCHINDUCTOR_CACHE_DIR=$C/inductor \
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 VIDEO_AUTOTUNE=0 \
      $V/venv-distilled/bin/python -W ignore -u $R/code/video_worker.py --tag $T >> $R/logs/$T.log 2>&1 &
    local p=$!
    while kill -0 $p 2>/dev/null; do stop && { kill $p; wait $p; break 2; }; sleep 5; done
    wait $p; echo "$(date -Is) video worker $T exited, restarting" >> $R/logs/$T.log; sleep 10
  done
  rm -f $R/workers/$T.json
}
echo "$(TZ=America/Los_Angeles date '+%m-%d %T PT') $NODE start; GPUs $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,memory.used,clocks.sm --format=csv,noheader
ppm 0,1 /scratch/amartyavarma/xabi-samples/w17352/export-200 ppm-xs-17352 &
ppm 2,3 /scratch/amartyavarma/xabi-samples/w17641/export-200 ppm-xs-17641 &
ppm 4,5 /data/home/amartyavarma/ppm/aloe-v2pure-16610/export-200 ppm-xs-16610 &
video 6 &
video 7 &
wait
echo "$(TZ=America/Los_Angeles date '+%m-%d %T PT') stopped"
