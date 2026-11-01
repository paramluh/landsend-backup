#!/bin/bash
# keep this run's HF uploader processes at the lowest CPU + IO priority on this host while train job $1 runs (diag 10-24: the nice-10
# uploader shared rank 0 / rank 8 CPUs and slowed steps 42-46 by ~30 s). Touches only amartyavarma's hfb_up/hf upload processes.
J=$1
while squeue -h -j $J 2>/dev/null | grep -q .; do
  for p in $(pgrep -u amartyavarma -f "hfb_up|huggingface|hf_upload" 2>/dev/null); do
    [ "$(ps -o ni= -p $p 2>/dev/null | tr -d ' ')" = 19 ] || { renice -n 19 -p $p >/dev/null 2>&1; ionice -c3 -p $p 2>/dev/null; echo "[$(TZ=America/Los_Angeles date '+%T PT')] $(hostname -s) reniced $p"; }
  done
  sleep 10
done
