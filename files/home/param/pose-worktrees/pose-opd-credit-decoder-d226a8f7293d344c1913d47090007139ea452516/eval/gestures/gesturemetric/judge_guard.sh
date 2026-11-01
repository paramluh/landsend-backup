#!/bin/bash
# motasr guard for gesturemetric side work on slinky-18 (main 10-26): baseline = mean of the last 20 motasr step times before start;
# touches judge/STOP_S18 if the rolling mean of the last 8 steps exceeds baseline x 1.02 or any step shows data wait > 0.0.
LOG=/data/home/amartyavarma/ppm/mot-asr/logs/motasr-30741.log
STOP=/data/home/amartyavarma/ppm/agents1026/gesturemetric/judge/STOP_S18
base=$(grep -oE "\| [0-9.]+s \(data wait" "$LOG" | tail -n 20 | grep -oE "[0-9.]+" | awk '{s+=$1} END {printf "%.3f", s/NR}')
echo "$(date -u +%H:%M:%S) guard start baseline ${base}s (limit x1.02)"
tail -n 0 -F "$LOG" | while read -r line; do
  t=$(echo "$line" | grep -oE "\| [0-9.]+s \(data wait [0-9.]+\)") || continue
  st=$(echo "$t" | grep -oE "[0-9.]+s" | head -1 | tr -d s); dw=$(echo "$t" | grep -oE "wait [0-9.]+" | grep -oE "[0-9.]+")
  win="$win $st"; win=$(echo $win | awk '{n=NF; s=""; for(i=(n>8?n-7:1);i<=n;i++) s=s" "$i; print s}')
  m=$(echo $win | awk '{s=0; for(i=1;i<=NF;i++) s+=$i; printf "%.3f", s/NF}'); n=$(echo $win | wc -w)
  bad=$(awk -v m="$m" -v b="$base" -v d="$dw" -v n="$n" 'BEGIN {print ((n>=8 && m > b*1.02) || d > 0.0) ? 1 : 0}')
  echo "$(date -u +%H:%M:%S) step ${st}s wait ${dw} roll8 ${m}s base ${base}s"
  if [ "$bad" = "1" ]; then touch "$STOP"; echo "$(date -u +%H:%M:%S) STOP: roll8 ${m}s vs base ${base}s, data wait ${dw}"; fi
done
