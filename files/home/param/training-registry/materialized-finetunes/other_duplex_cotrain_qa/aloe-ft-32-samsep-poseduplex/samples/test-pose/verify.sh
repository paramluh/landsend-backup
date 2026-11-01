#!/bin/bash
# verify skeleton mp4s of a run dir: container duration, decoded frame count, codec; stills at 5 s and 25 s -> <run>/still-*.png
FF=/data/home/amartyavarma/venvs/qwen-s2/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2
R=$1
for f in $R/pose_gen_s1.mp4 $R/pose_gen_s2.mp4 $R/pose_gt_s1.mp4 $R/pose_gt_s2.mp4; do
  echo "== $(basename $f) $(stat -c %s $f) bytes"
  $FF -hide_banner -i $f 2>&1 | grep -E "Duration|Stream" | sed 's/^ *//'
  $FF -hide_banner -i $f -map 0:v -f null - 2>&1 | grep -oE "frame= *[0-9]+" | tail -1
  $FF -hide_banner -loglevel error -y -ss 5 -i $f -frames:v 1 $R/still-$(basename $f .mp4)-5s.png
  $FF -hide_banner -loglevel error -y -ss 25 -i $f -frames:v 1 $R/still-$(basename $f .mp4)-25s.png
done
head -c 4096 $R/pose_gen_s1.mp4 | grep -c moov | sed 's/^/moov-in-first-4KB (faststart): /'
