#!/bin/bash
# remove the KILLED FT-(a) job 27211's node-local leftovers on the spares (cancelled at step ~41 by the coordinator 10-22 12:19 PT on the user's
# 'kill it'; the FT restarts from step 0 on PPM hosts): its intermediate export dirs (step 33) and any save dir. Nothing of value: a killed run's
# partial outputs. scratch_guard guard_rm; one line each -> setup-logs/hfb/DELETED-1021-hfb-hold.log
source /data/home/amartyavarma/ppm/scratch_guard.sh
H=$(hostname -s); LOG=/data/home/amartyavarma/setup-logs/hfb/DELETED-1021-hfb-hold.log; S=/scratch/amartyavarma/aloe-v2pure-17352
for d in $S/exports/export-* $S/saves/step-*; do [ -d "$d" ] || continue
  sz=$(du -s -BG "$d" | cut -f1); f0=$(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9)
  if guard_rm "$d"; then echo "$(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S') PT DEL $H $d size $sz free ${f0}G -> $(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9)G :: node-local output of the KILLED FT-(a) job 27211 (aloe-v2pure-17352, cancelled at step ~41, 10-22 12:19 PT, user 'kill it'; the FT restarts from step 0 on PPM hosts)" | tee -a $LOG
  else echo "$H guard refused $d"; fi
done
echo "$H done; /scratch free $(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9)G"
