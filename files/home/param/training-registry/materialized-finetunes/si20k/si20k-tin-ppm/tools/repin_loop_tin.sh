#!/bin/bash
# si20k-tin (copy of si20k-txt-ppm/tools/repin_loop.sh): per-node CPU re-pin loop (one task per node, srun --overlap into the FT job, started by the FT sbatch right after the training
# srun): waits for the 8 rank mains, then applies repin_smt_txt.py every 30 s for 25 min (catches the DataLoader workers / corruption servers
# as they spawn) and every 120 s afterwards. Operational only (affinity), no maths change. Log: <run>/logs/repin-<job>-<host>.log
LOG=$1/logs/repin-$SLURM_JOB_ID-$(hostname -s).log; PY=/usr/bin/python3; [ -x $PY ] || PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
t0=$(date +%s)
while :; do
  n=$(pgrep -u $USER -f "code/train_si20k_tin.py" | wc -l)
  [ "$n" -ge 8 ] && break
  [ $(( $(date +%s) - t0 )) -gt 1800 ] && { echo "$(date +%T) no training processes after 30 min: exit" >> $LOG; exit 0; }
  sleep 10
done
sleep 20
while pgrep -u $USER -f "code/train_si20k_tin.py" > /dev/null; do
  echo "== $(TZ=America/Los_Angeles date '+%T PT')" >> $LOG
  $PY -I /data/home/amartyavarma/ppm/si20k-tin-ppm/tools/repin_smt_tin.py >> $LOG 2>&1
  el=$(( $(date +%s) - t0 )); [ $el -lt 1500 ] && sleep 30 || sleep 120
done
