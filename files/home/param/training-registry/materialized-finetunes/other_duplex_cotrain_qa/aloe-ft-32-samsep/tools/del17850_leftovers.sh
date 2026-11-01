#!/bin/bash
# delete our own partial 17850 staging leftovers on this host (main GO 10-24); logs to DELETED-1021-hfb-hold.log
h=$(hostname -s); LOGF=/data/home/amartyavarma/setup-logs/hfb/DELETED-1021-hfb-hold.log
for p in /scratch/amartyavarma/aloe-samsep-17850-init /scratch/amartyavarma/aloe-samsep-17850-dl; do
  [ -e "$p" ] || { echo "$h $p absent"; continue; }
  [ -L "$p" ] && { echo "$h $p is a SYMLINK: refuse"; continue; }
  r=$(readlink -f "$p"); case "$r" in /scratch/amartyavarma/ppm|/scratch/amartyavarma/ppm/*) echo "$h $p resolves into /scratch/amartyavarma/ppm: refuse"; continue;; esac
  [ -n "$(find "$p" -type l -print -quit)" ] && { echo "$h $p contains symlinks: refuse"; continue; }
  [ -n "$(find "$p" -maxdepth 3 -name 'COMPLETE*' -print -quit)" ] && { echo "$h $p has COMPLETE: refuse"; continue; }
  sz=$(du -sh "$p" | cut -f1)
  if bash /data/home/amartyavarma/ppm/guard_rm "$p"; then
    [ -e "$p" ] && { echo "$h $p still exists after guard_rm"; continue; }
    msg="[$(TZ=America/Los_Angeles date '+%F %T PT')] aloeft32 (main GO 10-24): deleted $h:$p ($sz; partial dcp-17850 staging copy, never COMPLETE; source verified on HF InternalCan/ppm-v1-c/dcp-17850)"
    echo "$msg" >> $LOGF; echo "$msg"
  else echo "$h $p guard_rm REFUSED"; fi
done
df -h /scratch | tail -1 | sed "s/^/$h /"
