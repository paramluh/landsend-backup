#!/bin/bash
# PPM launch prep (10-11, main-thread order): free /scratch to >= TARGET GB by deleting REGENERABLE caches of old runs only, in this order:
# s4val, qwen-s2-tcache, qwen-s3-tcache-utkl, qwen-s3-tcache, inductor-cache. Never a dcp-*/checkpoint dir, s3-rows, pose-s3-samdata, ppm stores.
TARGET=${1:-250}; LOG=/data/home/amartyavarma/setup-logs/cleanup_deletions.log; H=$(hostname -s)
free() { df -BG --output=avail /scratch | tail -1 | tr -dc 0-9; }
for d in s4val qwen-s2-tcache qwen-s3-tcache-utkl qwen-s3-tcache inductor-cache; do
  f=$(free); [ "$f" -ge "$TARGET" ] && break
  p=/scratch/amartyavarma/$d; [ -d "$p" ] || continue
  s=$(du -sBG $p | cut -f1); /data/home/amartyavarma/ppm/guard_rm "$p" || continue
  echo "$(TZ=America/Los_Angeles date '+%F %T PT') $H deleted $p ($s; regenerable cache; PPM launch needs >= ${TARGET}G free) free $f -> $(free)G" >> $LOG
done
echo "$H free=$(free)G"
