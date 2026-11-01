#!/bin/bash
# Adds npl5 scan lanes (stage_lane3.sh) on slinky-18 / slinky-10 as soon as A/D free CPUs there (coordinator 10-27 07:15 PT), and releases the
# claims of lanes that died. Orchestration only (sinfo / squeue / sbatch). Every added lane: raw on /data (DATA_MIN_GB 530: two /data lanes
# fetching at once keep /data >= ~376 GB), 2 scan workers per CPU (Weka-bound), --deadline 10:54 PT and LANE_DEADLINE_EPOCH so it never
# claims a slot it cannot finish (SLOT_SECS). A lane still PENDING after 3 min is cancelled (never block A/C/D).
W=/data/home/amartyavarma/ppm/pose5/npl5; B=$W/bin
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
say() { echo "== $(pt) [adder] $*" | tee -a $W/logs/lanes.log; }
DL_UTC=${DL_UTC:-2026-10-27T17:54:00}; DL_EPOCH=$(date -u -d "$DL_UTC" +%s); STOP_AT=$(date -u -d "${STOP_UTC:-2026-10-27T16:50:00}" +%s)
declare -A JN=([A]=30889 [B]=30922); n=0
alloc_mem() { scontrol show node $1 | awk '{for(i=1;i<=NF;i++){split($i,a,"="); if(a[1]=="RealMemory")r=a[2]; if(a[1]=="AllocMem")m=a[2]; if(a[1]=="MemSpecLimit")s=a[2]}} END{print int((r-s-m)/1024)}'; }
while [ $(date +%s) -lt $STOP_AT ]; do
  # release claims of dead lanes (no scan-ok, owning lane job gone)
  for c in $W/state/claim-*; do
    s=${c##*/claim-}; [ -f $W/state/scan-$s.ok ] && continue
    ln=$(cut -d@ -f1 $c/by 2>/dev/null); j=${JN[$ln]:-}
    [ -n "$j" ] && ! squeue -h -j $j 2>/dev/null | grep -q . && { rm -rf "${c:?}" "${W:?}/state/adopted-${s:?}"; say "released stale claim $s (lane $ln job $j gone)"; }
  done
  left=$(( 16 - $(ls -d $W/state/claim-* 2>/dev/null | wc -l) ))
  [ $left -le 0 ] && { say "all 16 slots claimed: adder exits"; exit 0; }
  for node in slinky-18 slinky-10; do
    [ $n -ge ${MAX_EXTRA:-3} ] && break
    free=$(sinfo -h -n $node -o %C | awk -F/ '{print $2}'); mem=$(alloc_mem $node)
    if [ "${free:-0}" -ge 4 ] && [ "${mem:-0}" -ge 22 ] && [ $(( $(date +%s) + ${SLOT_SECS:-2700} )) -lt $DL_EPOCH ]; then
      c=$(( free > 8 ? 8 : free )); n=$((n + 1)); L=$(printf "\\x$(printf %x $((67 + n)))")   # D, E, F
      J=$(sbatch --parsable -w $node -c $c --mem=20G -t 3:00:00 --time-min=0:50:00 --deadline=$DL_UTC \
        --export=ALL,LANE_SCRIPT=stage_lane3.sh,WORKERS=$((2 * c)),DATA_MIN_GB=530,LANE_DEADLINE_EPOCH=$DL_EPOCH,SLOT_SECS=${SLOT_SECS:-2700} \
        $B/stage_lane.sbatch $L $W/raw)
      JN[$L]=$J; echo "$(cat $W/state/LANE_JOBS) $J" > $W/state/LANE_JOBS; say "lane $L job $J on $node (-c $c, $free free CPUs, ${mem}G mem)"
      for i in 1 2 3 4 5 6; do sleep 30; [ "$(squeue -h -j $J -o %T)" = RUNNING ] && break; done
      [ "$(squeue -h -j $J -o %T)" = PENDING ] && { scancel $J; say "lane $L job $J still PENDING after 3 min: cancelled (not blocking others)"; n=$((n - 1)); sleep 300; }
    fi
  done
  sleep 60
done
say "adder stop time reached"
