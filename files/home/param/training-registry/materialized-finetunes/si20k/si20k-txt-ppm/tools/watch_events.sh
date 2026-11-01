#!/bin/bash
# si20k-txt silent monitor (login node, read-only checks): exits with a one-line reason on CRASH (FT job gone without the final export),
# STALL (no new step / eval / export log line for 12 min while the job runs), HF FAIL, or DONE (both finals exported + HF-verified).
P=/data/home/amartyavarma/ppm/si20k-txt-ppm; I=/data/home/amartyavarma/ppm/si20k-txt-intl
while :; do
  done_n=0
  for R in $P $I; do
    J=$(cat $R/LAUNCHED_JOB); L=$(ls $R/logs/si20ktxt-*-$J.log 2>/dev/null | head -1); n=$(basename $R)
    st=$(squeue -h -j $J -o %T 2>/dev/null)
    if [ -f $R/export-198/HF_BACKUP.json ] && [ -f $R/FINAL_EXPORT_READY ]; then done_n=$((done_n + 1)); continue; fi
    if [ -z "$st" ] && [ ! -f $R/FINAL_EXPORT_READY ]; then echo "CRASH $n job $J: $(grep -v Warning $L | grep -m1 -A3 Traceback | tr '\n' ' ' | cut -c1-300) last: $(grep -v Warning $L | tail -1 | cut -c1-200)"; exit 1; fi
    if [ "$st" = RUNNING ] && [ -n "$L" ]; then age=$(( $(date +%s) - $(stat -c %Y $L) )); [ $age -gt 720 ] && { echo "STALL $n job $J: log silent ${age}s; last: $(tail -1 $L | cut -c1-200)"; exit 1; }; fi
    grep -q "HF upload rc [1-9]" $R/exports/BACKUP_LOG.txt 2>/dev/null && { echo "HF FAIL $n: $(grep 'HF upload rc [1-9]' $R/exports/BACKUP_LOG.txt | tail -1)"; exit 1; }
  done
  [ $done_n -eq 2 ] && { echo "DONE both finals exported + HF-verified"; exit 0; }
  sleep 30
done
