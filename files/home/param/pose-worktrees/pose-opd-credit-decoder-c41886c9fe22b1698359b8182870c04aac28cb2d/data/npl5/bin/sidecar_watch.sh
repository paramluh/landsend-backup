#!/bin/bash
# npl5 orig_h/orig_w sidecars for slots handled by a v1 lane (stage_lane.sh, already running when change #2 arrived): runs as an
# `srun --overlap` step inside that lane's own job; every slot this host claimed and fetched+verified gets bin/sidecar_origwh.py on its raw
# share (minutes) long before the lane deletes the raw after its ~25 min scan. -> state/origwh-<slot>.ok | .missed
W=/data/home/amartyavarma/ppm/pose5/npl5; RAW=${1:?raw root}; ME=$(hostname -s)
SLOTS="slinky-0 slinky-1 slinky-2 slinky-3 slinky-5 slinky-6 slinky-7 slinky-8 slinky-9 slinky-11 slinky-12 slinky-13 slinky-14 slinky-16 slinky-17 slinky-19"
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
while :; do
  for s in $SLOTS; do
    grep -qs "@$ME" $W/state/claim-$s/by || continue
    [ -f $W/state/fetched-$s.ok ] && [ ! -f $W/state/origwh-$s.ok ] && [ ! -f $W/state/origwh-$s.missed ] || continue
    t0=$(date +%s)
    if [ -d $RAW/$s/jobs ] && nice -n19 /data/home/amartyavarma/venvs/qwen-s2/bin/python -I $W/bin/sidecar_origwh.py $RAW/$s $s --workers 2 > $W/logs/origwh-$s.log 2>&1; then
      echo "$(pt) $ME $(( $(date +%s) - t0 ))s" > $W/state/origwh-$s.ok; printf "origwh\t%s\t%ss\twatch@%s\t%s\n" $s $(( $(date +%s) - t0 )) $ME "$(pt)" >> $W/TIMES.tsv
      echo "== $(pt) [origwh@$ME] $s sidecar done ($(( $(date +%s) - t0 )) s)" >> $W/logs/lanes.log
    else
      echo "$(pt) $ME raw gone or failed" > $W/state/origwh-$s.missed; echo "== $(pt) [origwh@$ME] $s sidecar MISSED ($W/logs/origwh-$s.log)" >> $W/logs/lanes.log
    fi
  done
  sleep 20
done
