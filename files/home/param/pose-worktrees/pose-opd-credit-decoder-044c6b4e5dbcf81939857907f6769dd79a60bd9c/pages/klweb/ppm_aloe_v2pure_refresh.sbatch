#!/bin/bash
#SBATCH -J klweb-ppm-aloe-v2pure-refresh -p all -w slinky-4 -N1 -n1 -c1 --mem=2G -t 3-00:00:00 -o /data/home/amartyavarma/setup-logs/klweb/ppmaloev2purerefresh.log
# 60 s rebuild of ppm_aloe_v2pure.html (PPM Aloe v2pure fine-tune live curves = the uncond page format + the 8B SAM pose FT reference series;
# served by the existing klweb tunnel, klweb-ppm / ppm_serve.sbatch). CPU only, no GPUs (slinky-4: slinky-18 CPUs full on 10-17).
W=/data/home/amartyavarma/setup-logs/klweb
while true; do
  timeout 240 python3 $W/build_ppm_aloe_v2pure.py 2>&1 | sed "s/^/$(TZ=America/Los_Angeles date +%a\ %H:%M\ PT) /" | tail -3
  sleep 60
done
