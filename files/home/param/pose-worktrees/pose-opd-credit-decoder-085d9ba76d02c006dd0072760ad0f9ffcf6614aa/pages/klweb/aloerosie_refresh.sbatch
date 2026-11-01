#!/bin/bash
#SBATCH -J klweb-aloerosie -p all -x slinky-9 -N1 -n1 -c2 --mem=4G -t 0-03:00:00
#SBATCH -o /data/home/amartyavarma/setup-logs/klweb/aloerosierefresh.log
# 10-27 (user: "give me live curve"): 60 s rebuild of stage3_aloerosie.html (S3_PAGE=aloerosie; s3branches_aloerosie.json) for the Rosie pose FT
W=/data/home/amartyavarma/setup-logs/klweb
while true; do
  S3_PAGE=aloerosie timeout 300 python3 $W/build_stage3.py $W 2>&1 | sed "s/^/$(date +%a\ %H:%M) /" | tail -2
  sleep 60
done
