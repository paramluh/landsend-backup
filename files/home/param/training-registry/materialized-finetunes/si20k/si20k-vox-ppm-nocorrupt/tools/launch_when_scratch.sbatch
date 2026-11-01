#!/bin/bash
#SBATCH -p all -N1 -n1 -c 1 --mem=1G --gres=none -t 6:00:00 -J si20knoc-launch --exclude=slinky-4,slinky-18
#SBATCH -o /data/home/amartyavarma/ppm/si20k-vox-ppm-nocorrupt/logs/launch-%j.out
# launch the no-corruption run (+ watcher) once EVERY PPM host has >= 30 GB free /scratch (the 11488 rescue frees its node-local exports one at
# a time); polls every 60 s (one line per host, min taken); launch once (LAUNCH_DONE guard). Fallback: after 45 min, launch anyway if the
# minimum is >= 22 GB (two node-local exports' room; the trainer skips a node-local copy that would not fit, the watcher moves them promptly).
Q=/data/home/amartyavarma/ppm/si20k-vox-ppm-nocorrupt; HP=slinky-0,slinky-1,slinky-3,slinky-5,slinky-8,slinky-9,slinky-16,slinky-17
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
t0=$(date +%s)
while :; do
  [ -f $Q/LAUNCH_DONE ] && { echo "[$(pt)] already launched: $(cat $Q/LAUNCH_DONE)"; exit 0; }
  mn=$(env -i PATH=/usr/bin:/bin HOME=$HOME USER=$USER srun -p all -w $HP -N8 -n8 --ntasks-per-node=1 -c1 --mem=100M --gres=none -t 2 -J si20knoc-df bash -c 'echo "DF $(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9)"' 2>/dev/null | awk '$1=="DF"{print $2}' | sort -n)
  n=$(echo "$mn" | grep -c .); lo=$(echo "$mn" | head -1)
  el=$(( $(date +%s) - t0 ))
  echo "[$(pt)] hosts reporting $n, min /scratch free ${lo:-?} GB"
  if [ "$n" -eq 8 ] && { [ "$lo" -ge 30 ] || { [ $el -gt 2700 ] && [ "$lo" -ge 22 ]; }; }; then
    j=$(sbatch --parsable -w $HP $Q/si20k_vox_ppm_nocorrupt.sbatch) || { sleep 60; continue; }
    w=$(sbatch --parsable --exclude=slinky-4,slinky-18 -o $Q/logs/watch-%j.out $Q/tools/watch_noc.sbatch $Q $j si20kvoxnocorrupt 80 600 -)
    echo $j > $Q/LAUNCHED_JOB; echo "$(pt) nocorrupt $j (watch $w): min /scratch free $lo GB" | tee $Q/LAUNCH_DONE
    echo "- $(pt): launched job $j (min /scratch free $lo GB after the 11488 rescue; earlier attempts 25283 / 25293 aborted at preflight: /scratch full)." >> $Q/README.md
    exit 0
  fi
  sleep 60
done
