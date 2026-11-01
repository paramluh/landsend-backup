#!/bin/bash
# silent monitor for si20k-tin-ppm, si20k-tin-intl, si20k-tdx-intl (login node, read-only): exits with ONE line on the first event:
# STEP1 <run> (first time a run logs step 1; reported runs listed in $SEEN), CRASH, STALL (running log silent 12 min), HF FAIL,
# CHAIN (the tdx job still not running 10 min after the text-in INTL job ended), DONE (all three finals exported + HF-verified).
SEEN=${1:-/data/home/amartyavarma/ppm/si20k-tin-ppm/logs/events_seen.txt}; touch $SEEN
RUNS="/data/home/amartyavarma/ppm/si20k-tin-ppm /data/home/amartyavarma/ppm/si20k-tin-intl /data/home/amartyavarma/ppm/si20k-tdx-intl /data/home/amartyavarma/ppm/si20k-vox-ppm-nocorrupt"
tin_end=""
while :; do
  done_n=0
  for R in $RUNS; do
    [ -f $R/LAUNCHED_JOB ] || continue
    n=$(basename $R); J=$(cat $R/LAUNCHED_JOB); L=$(ls $R/logs/si20k*-$J.log 2>/dev/null | head -1); st=$(squeue -h -j $J -o %T 2>/dev/null)
    if [ -f $R/export-198/HF_BACKUP.json ] && [ -f $R/FINAL_EXPORT_READY ]; then done_n=$((done_n + 1)); continue; fi
    if [ -n "$L" ] && grep -q "\] step 1/198 " $L && ! grep -qx "STEP1 $n $J" $SEEN; then echo "STEP1 $n $J" >> $SEEN; echo "STEP1 $n $J: $(grep -m1 'EVAL step 0:' $L | grep -o 'eval_speech/word_ce_norm [0-9.]*') $(grep -m1 'WITHOUT the user input' $L | grep -o 'word_ce_norm [0-9.]*') $(grep '\] step [0-9]*/198' $L | tail -1 | grep -o 'step [0-9]*/198 [0-9.]*s')"; exit 0; fi
    if [ -z "$st" ] && [ ! -f $R/FINAL_EXPORT_READY ] && ! grep -q "done 198 steps" $L 2>/dev/null; then echo "CRASH $n job $J: $(grep -v Warning $L 2>/dev/null | grep -m1 -A6 Traceback | tail -2 | tr '\n' ' ' | cut -c1-300) last: $(grep -v Warning $L 2>/dev/null | tail -1 | cut -c1-200)"; exit 1; fi
    if [ "$st" = RUNNING ] && [ -n "$L" ]; then age=$(( $(date +%s) - $(stat -c %Y $L) )); [ $age -gt 720 ] && { echo "STALL $n job $J: log silent ${age}s; last: $(tail -1 $L | cut -c1-200)"; exit 1; }; fi
    grep -q "HF upload rc [1-9]" $R/exports/BACKUP_LOG.txt 2>/dev/null && { echo "HF FAIL $n: $(grep 'HF upload rc [1-9]' $R/exports/BACKUP_LOG.txt | tail -1)"; exit 1; }
  done
  JI=$(cat /data/home/amartyavarma/ppm/si20k-tin-intl/LAUNCHED_JOB); JX=$(cat /data/home/amartyavarma/ppm/si20k-tdx-intl/LAUNCHED_JOB)
  if [ -z "$(squeue -h -j $JI 2>/dev/null)" ]; then
    [ -z "$tin_end" ] && tin_end=$(date +%s)
    sx=$(squeue -h -j $JX -o "%T %R" 2>/dev/null)
    if echo "$sx" | grep -q PENDING && [ $(( $(date +%s) - tin_end )) -gt 600 ]; then echo "CHAIN tdx job $JX still $sx 10 min after text-in INTL $JI ended"; exit 1; fi
  fi
  [ $done_n -eq 4 ] && { echo "DONE all four finals exported + HF-verified"; exit 0; }
  sleep 30
done
