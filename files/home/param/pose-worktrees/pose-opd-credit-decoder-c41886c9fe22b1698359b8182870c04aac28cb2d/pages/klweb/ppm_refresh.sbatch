#!/bin/bash
#SBATCH -J klweb-ppm-refresh -p all -w slinky-15 -N1 -n1 -c2 --mem=4G -t 7-00:00:00 -o /data/home/amartyavarma/setup-logs/klweb/ppmrefresh.log
# 60 s rebuild of stage3_ppm.html / s3live_ppm.json / ver_s3_ppm.txt (S3_PAGE=ppm; s3branches_ppm.json). Served by the PPM tunnel job (ppm_serve.sbatch). CPU only.
W=/data/home/amartyavarma/setup-logs/klweb
while true; do
  S3_PAGE=ppm timeout 300 python3 $W/build_stage3.py $W 2>&1 | sed "s/^/$(date +%a\ %H:%M) /" | tail -3
  sleep 60
done
