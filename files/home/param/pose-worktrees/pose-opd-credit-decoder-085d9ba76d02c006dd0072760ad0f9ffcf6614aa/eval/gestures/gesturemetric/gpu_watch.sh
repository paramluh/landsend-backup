#!/bin/bash
# prints the first non-slinky-9 node with >= 2 unallocated GPUs, then exits; polls every 60 s
while true; do
  for n in $(sinfo -N -h -o "%N" | sort -u | grep -v '^slinky-9$'); do
    a=$(scontrol show node "$n" | grep -oE "AllocTRES=\S*" | grep -oE "gres/gpu=[0-9]+" | cut -d= -f2)
    a=${a:-0}
    if [ $((8 - a)) -ge 2 ]; then echo "$(date -u +%H:%M) $n free=$((8 - a))"; exit 0; fi
  done
  sleep 60
done
