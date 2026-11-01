#!/bin/bash
# LOGIN NODE: launch-only driver. Per slot, one at a time: stage source -> Weka xfer (srun on source), fan out to all destinations at once
# (one srun), delete the staged copy once every destination verified (srun on a PPM host).
set -uo pipefail
source /data/home/amartyavarma/ppm/ttconvo-ft/tools/log.sh
B=/data/home/amartyavarma/ppm/ttconvo-ft; T=$B/tools
OV="--overlap -n1 -c1 --mem=0 --cpu-bind=none"; NI="nice -n 19 ionice -c3"
NJ="-p all --ntasks-per-node=1 -c 16 --mem=32G --gres=none -t 30 -J ttft-stage"
declare -A SRCJOB=([slinky-4]=24188 [slinky-7]=26450 [slinky-18]=26231)
declare -A DEST=([slinky-4]="slinky-0,slinky-1,slinky-2,slinky-3" [slinky-7]="slinky-5,slinky-6,slinky-8,slinky-9" \
  [slinky-11]="slinky-10,slinky-12,slinky-14" [slinky-13]="slinky-16,slinky-17,slinky-19" \
  [slinky-18]="slinky-0,slinky-1,slinky-2,slinky-3,slinky-5,slinky-6,slinky-8,slinky-9,slinky-10,slinky-11,slinky-12,slinky-13,slinky-14,slinky-16,slinky-17,slinky-19")
for SLOT in ${@:-slinky-4 slinky-7 slinky-11 slinky-13 slinky-18}; do
  log "orchestrate $SLOT: /data free $(df -B1G --output=avail /data | tail -1 | tr -dc 0-9)G before staging"
  if [ -n "${SRCJOB[$SLOT]:-}" ]; then
    timeout 1800 srun --jobid=${SRCJOB[$SLOT]} $OV $NI $T/stage_src.sh $SLOT; rc=$?
  else
    timeout 1800 srun $NJ -w $SLOT -N1 $T/stage_src.sh $SLOT; rc=$?
  fi
  [ $rc -eq 0 ] || { log "orchestrate ABORT: stage $SLOT rc=$rc (staged copy left at $B/xfer/$SLOT)"; exit 1; }
  n=$(echo ${DEST[$SLOT]} | tr ',' '\n' | wc -l)
  timeout 1800 srun $NJ -w ${DEST[$SLOT]} -N$n $T/fanout.sh $SLOT; rc=$?
  ok=$(grep -c "fanout $SLOT OK" $B/logs/staging.log)
  if [ $rc -ne 0 ] || [ $ok -lt $n ]; then log "orchestrate ABORT: fanout $SLOT rc=$rc ok_lines=$ok/$n (staged copy kept)"; exit 1; fi
  srun $NJ -w slinky-12 -N1 rm -rf $B/xfer/$SLOT; rc=$?
  [ $rc -eq 0 ] && [ ! -e $B/xfer/$SLOT ] || { log "orchestrate ABORT: rm xfer/$SLOT rc=$rc"; exit 1; }
  log "orchestrate $SLOT done: $n/$n destinations verified, staged copy deleted; /data free $(df -B1G --output=avail /data | tail -1 | tr -dc 0-9)G"
done
log "orchestrate finished: ${@:-all slots}"
