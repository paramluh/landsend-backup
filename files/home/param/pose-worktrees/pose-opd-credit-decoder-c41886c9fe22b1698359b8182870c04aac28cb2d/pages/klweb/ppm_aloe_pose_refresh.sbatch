#!/bin/bash
#SBATCH -J klweb-ppm-aloe-pose-refresh -p all -w slinky-15 -N1 -n1 -c1 --mem=2G -t 3-00:00:00 -o /data/home/amartyavarma/setup-logs/klweb/ppmaloeposerefresh.log
# 60 s rebuild of ppm_aloe_pose.html (PPM Aloe POSE fine-tune live curves + row-build progress before launch; served by the existing klweb tunnel). CPU only.
W=/data/home/amartyavarma/setup-logs/klweb
while true; do
  timeout 240 python3 $W/build_ppm_aloe_pose.py 2>&1 | sed "s/^/$(TZ=America/Los_Angeles date +%a\ %H:%M\ PT) /" | tail -3
  sleep 60
done
