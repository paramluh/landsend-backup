#!/bin/bash
# exit with one line when an si20k run crashes / NaNs / stalls (> 12 min without a new step or eval line), or a final export is HF-verified
P=/data/home/amartyavarma/ppm; declare -A JOB=([ppm]=24920 [intl]=24921); declare -A DONE
while true; do
  for r in ppm intl; do
    [ -n "${DONE[$r]:-}" ] && continue
    L=$P/si20k-vox-$r/logs/si20kvox-$r-${JOB[$r]}.log
    if grep -qE "Traceback|non-finite|^ABORT|ABORT [a-z0-9-]*:" $L 2>/dev/null; then echo "$r CRASH/NaN: $(grep -E "Traceback|non-finite|ABORT|Error" $L | tail -2 | tr '\n' ' ' | cut -c1-300)"; exit 1; fi
    st=$(squeue -h -j ${JOB[$r]} -o %T 2>/dev/null)
    if [ -f $P/si20k-vox-$r/export-198/HF_BACKUP.json ]; then echo "$r FINAL export-198 HF-verified: $(cat $P/si20k-vox-$r/export-198/HF_BACKUP.json | cut -c1-200)"; DONE[$r]=1; exit 0; fi
    if [ -z "$st" ] && ! grep -q "\[export\] $P/si20k-vox-$r/export-198:" $L 2>/dev/null; then echo "$r job ended WITHOUT the final export: $(tail -2 $L | tr '\n' ' ' | cut -c1-300)"; exit 1; fi
    age=$(( $(date +%s) - $(stat -c %Y $L) ))
    [ -n "$st" ] && [ $age -gt 720 ] && { echo "$r STALL: log silent ${age}s, last: $(grep -E '\] step [0-9]+/198' $L | tail -1 | cut -c1-120)"; exit 1; }
  done
  sleep 60
done
