#!/bin/bash
#SBATCH -J klweb-ppm811-refresh -p all -w slinky-15 -N1 -n1 -c1 --mem=4G -t 7-00:00:00 -o /data/home/amartyavarma/setup-logs/klweb/ppm811refresh.log
# 10-21: 60 s rebuild of the 8,1,1 page stage3_ppm811.html / s3live_ppm811.json / ver_s3_ppm811.txt (S3_PAGE=ppm811; s3branches_ppm811.json; ppm_doc.py in 8,1,1 mode).
# Modelled on ppm_refresh.sbatch; served by the PPM tunnel job (ppm_serve.sbatch, every file in klweb). CPU only.
W=/data/home/amartyavarma/setup-logs/klweb
while true; do
  S3_PAGE=ppm811 timeout 300 python3 $W/build_stage3.py $W 2>&1 | sed "s/^/$(date +%a\ %H:%M) /" | tail -3
  sleep 60
done
