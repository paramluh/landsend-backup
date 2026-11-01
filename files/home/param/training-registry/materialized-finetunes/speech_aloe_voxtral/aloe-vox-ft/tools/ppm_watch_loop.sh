#!/bin/bash
# every 60 s: PPM s/step over its last 5 steps + our side-work steps in job 22224 (log only; throttling decided by the operator)
while true; do
  echo "$(TZ=America/Los_Angeles date +%T) $(python3 /data/home/amartyavarma/ppm/aloe-vox-ft/tools/ppm_steps.py 22224 5) | side: $(squeue -h -s -j 22224 -o %.40j 2>/dev/null | grep aloevox | sort | tr '\n' ' ')"
  sleep 60
done
