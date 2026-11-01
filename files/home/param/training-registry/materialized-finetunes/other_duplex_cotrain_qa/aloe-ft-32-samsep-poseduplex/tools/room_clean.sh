#!/bin/bash
# pose-FT room on THIS host (main amartyavarma-5f 10-25 ~03:10 PT): free /scratch until >= NEED GB, in this order, stopping as soon as it fits:
#   1. D1: /scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-17850 (user's explicit OK; plain rm, logged) via d1_rm_dcp17850.sh
#   2. guard_rm (user said clear): ppm14b, conv-pool-v2, demeter-x, then old FT dirs aloe-uncond-ft, aloe-nopose-ft, aloe-pose-ft, aloe-zoom-cotrain,
#      aloe-v2vox-asr-w15973*, aloe-v2pure-clip80 (oldest first). NEVER anything else under ppm/ (ppm-next, ppm-mot, ppmvox, s3-rows*, pose stores,
#      new-pose-llm-data, npl-append, the v5 pause save...). Logged to DELETED-1021-hfb-hold.log.
NEED=${NEED:-149}; H=$(hostname -s); S=/scratch/amartyavarma; LOG=/data/home/amartyavarma/setup-logs/hfb/DELETED-1021-hfb-hold.log
source /data/home/amartyavarma/ppm/scratch_guard.sh
fr() { df -BG --output=avail /scratch | tail -1 | tr -dc 0-9; }
[ $(fr) -ge $NEED ] && { echo "$H fits: $(fr) GB >= $NEED"; exit 0; }
bash /data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/tools/d1_rm_dcp17850.sh
for d in $S/ppm14b $S/conv-pool-v2 $S/demeter-x $(ls -dtr $S/aloe-uncond-ft $S/aloe-nopose-ft $S/aloe-pose-ft $S/aloe-zoom-cotrain $S/aloe-v2vox-asr-w15973* $S/aloe-v2pure-clip80 2>/dev/null); do
  [ $(fr) -ge $NEED ] && break
  [ -d "$d" ] || continue
  sz=$(du -sh "$d" | cut -f1)
  if guard_rm "$d"; then echo "[$(TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT')] ALOEFT32 pose-FT room (user said clear these; main amartyavarma-5f 10-25): deleted $H:$d ($sz)" | tee -a $LOG
  else echo "$H: guard_rm refused $d"; fi
done
echo "$H final free $(fr) GB (need $NEED): $([ $(fr) -ge $NEED ] && echo OK || echo STILL-SHORT)"; [ $(fr) -ge $NEED ]
