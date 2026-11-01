#!/bin/bash
# main (amartyavarma-ef) go 10-24 ~03:05 PT: delete the staged 16861 init SUBSET on slinky-11..19 (full verified copies stay on slinky-18 / slinky-4
# /scratch/amartyavarma/aloe-samsep-16861-init + HF InternalCan/ppm-v1-b/dcp-16861). guard_rm + DELETED-1021-hfb-hold.log.
source /data/home/amartyavarma/ppm/scratch_guard.sh
H=$(hostname -s); LOG=/data/home/amartyavarma/setup-logs/hfb/DELETED-1021-hfb-hold.log; d=/scratch/amartyavarma/aloe-samsep-16861-init
case $H in slinky-11|slinky-12|slinky-13|slinky-14|slinky-15|slinky-16|slinky-17|slinky-19) ;; *) echo "$H: not a w64 host, refusing"; exit 1;; esac
[ -f $d/SUBSET_W64.json ] || { echo "$H: $d has no SUBSET_W64.json (not my staged subset): refusing"; exit 1; }
b=$(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9); sz=$(du -sh $d | cut -f1)
if guard_rm $d; then echo "[$(TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT')] aloe-ft-32-samsep w64 (ALOEFT32, main amartyavarma-ef go): deleted $H:$d ($sz, staged world-64 SUBSET of the 16861 init; full copies on slinky-18/4 + HF InternalCan/ppm-v1-b/dcp-16861)" >> $LOG; else echo "$H guard_rm REFUSED"; exit 1; fi
a=$(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9); echo "$H /scratch ${b}G -> ${a}G (+$((a-b))G)"
