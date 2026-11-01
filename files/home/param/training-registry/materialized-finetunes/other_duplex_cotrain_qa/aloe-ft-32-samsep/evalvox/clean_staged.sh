#!/bin/bash
# evalvox: delete ONE staged PPMVOX export copy on this host after its Aloe eval (main's go: DELETE_STAGED_OK). usage: bash clean_staged.sh <dir> <source note>
source /data/home/amartyavarma/ppm/scratch_guard.sh
d=$1; note=$2; H=$(hostname -s); LOG=/data/home/amartyavarma/setup-logs/hfb/DELETED-1021-hfb-hold.log
case "$d" in /scratch/amartyavarma/evalvox/*) ;; *) echo "$H: $d not under /scratch/amartyavarma/evalvox: refusing"; exit 1;; esac
[ -d "$d" ] || { echo "$H: $d absent"; exit 0; }
[ -f "$d/EVALVOX_STAGE.json" ] || { echo "$H: $d has no EVALVOX_STAGE.json (not an evalvox staged copy): refusing"; exit 1; }
sz=$(du -sh "$d" | cut -f1)
guard_rm "$d" || { echo "$H: guard_rm refused $d"; exit 1; }
echo "[$(TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT')] evalvox (ALOEFT32, main go): deleted $H:$d ($sz, staged copy of $note, after its Aloe eval)" | tee -a $LOG
