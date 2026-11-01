#!/bin/bash
# main (amartyavarma-8d) go 10-24 ~05:00 PT: delete the node-local FINAL export-249 of the w64 Aloe FT (job 28511, ppm-aloe-samsep-a16861w480) on
# slinky-11..19: HF VERIFY PASS on all 8 hosts (private InternalCan/ppm-aloe-samsep-a16861w480-final/export-249/); the live demo has its own copy on slinky-10.
source /data/home/amartyavarma/ppm/scratch_guard.sh
H=$(hostname -s); LOG=/data/home/amartyavarma/setup-logs/hfb/DELETED-1021-hfb-hold.log; d=/scratch/amartyavarma/aloe-ft-32-samsep/exports-a16861w480/export-249
case $H in slinky-11|slinky-12|slinky-13|slinky-14|slinky-15|slinky-16|slinky-17|slinky-19) ;; *) echo "$H: not a w64 host, refusing"; exit 1;; esac
[ -d $d ] || { echo "$H: $d absent"; exit 0; }
b=$(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9); sz=$(du -sh $d | cut -f1)
if guard_rm $d; then echo "[$(TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT')] aloe-ft-32-samsep w64 (ALOEFT32, main amartyavarma-8d go): deleted $H:$d ($sz, node-local FINAL export of job 28511) after HF VERIFY PASS -> private InternalCan/ppm-aloe-samsep-a16861w480-final/export-249/ (all 8 hosts); demo copy on slinky-10" >> $LOG; else echo "$H guard_rm REFUSED"; exit 1; fi
a=$(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9); echo "$H /scratch ${b}G -> ${a}G (+$((a-b))G)"
