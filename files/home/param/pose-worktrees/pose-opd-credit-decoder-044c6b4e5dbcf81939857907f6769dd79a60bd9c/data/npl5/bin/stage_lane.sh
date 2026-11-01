#!/bin/bash
# npl5 (delivery 5) OFF-HOST staging lane (pose5 workstream B, 10-27). Runs inside its own sbatch allocation on a SPARE (stage_lane.sbatch),
# CPU / network / scratch only, nice 19 + ionice idle. Loop: claim the next unclaimed slot (plan key = FRESH_HOST, 16 = the 811 plan keys)
# -> fetch its placement share from the HF bucket (code-811/fresh/append/src_fetch.py, unchanged) into RAW/<slot> -> xet verify
# (src_verify.py, unchanged) -> fresh_pose.py scan (code-811, unchanged; FRESH_HOST=<slot>, FRESH_SRC=RAW/<slot>/jobs, FRESH_SCAN=W/scan:
# the scan holds no paths, so it equals the on-host scan) -> delete the raw share. The next slot is fetched while this one scans.
# usage: stage_lane.sh <lane name> <RAW root>   (RAW on /data -> pauses while /data free < DATA_MIN_GB, default 450)
# Stop gracefully: touch W/state/STOP_LANES (exits at the next slot boundary). Claims: W/state/claim-<slot> (mkdir). Done: W/state/scan-<slot>.ok
set -uo pipefail
LANE=${1:?lane}; RAW=${2:?raw root}
W=/data/home/amartyavarma/ppm/pose5/npl5; A=/data/home/amartyavarma/ppm/code-811/fresh/append; B=$W/bin
PYV=/data/home/amartyavarma/venvs/voicebank/bin/python; PYQ=/data/home/amartyavarma/venvs/qwen-s2/bin/python
SLOTS="slinky-0 slinky-1 slinky-2 slinky-3 slinky-5 slinky-6 slinky-7 slinky-8 slinky-9 slinky-11 slinky-12 slinky-13 slinky-14 slinky-16 slinky-17 slinky-19"
CPT=${SLURM_CPUS_PER_TASK:-$(nproc)}; WORKERS=${WORKERS:-$CPT}
N="nice -n19 ionice -c3"; ME=$(hostname -s)
DATA_MIN_GB=${DATA_MIN_GB:-450}
if [ "$RAW" = auto ]; then   # this spare's /scratch if it keeps >= 200 GB free (2 slot shares in flight + a 45 GB floor), else rolling on /data
  F=$(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9)
  RAW=$([ "$F" -ge 200 ] && echo /scratch/amartyavarma/npl5-stage || echo $W/raw)
  echo "== $(TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT') [$LANE@$ME] RAW auto: /scratch free ${F}G -> $RAW" | tee -a $W/logs/lanes.log
fi
mkdir -p $W/state $W/logs $W/scan $RAW
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
say() { echo "== $(pt) [$LANE@$ME] $*" | tee -a $W/logs/lanes.log; }
rec() { printf "%s\t%s\t%s\t%s\t%s\n" "$1" "$2" "$3" "$LANE@$ME" "$(pt)" >> $W/TIMES.tsv; }
claim() { for s in $SLOTS; do [ -f $W/state/scan-$s.ok ] && continue; mkdir $W/state/claim-$s 2>/dev/null && { echo $LANE@$ME > $W/state/claim-$s/by; echo $s; return; }; done; }
unclaim() { rm -rf "${W:?}/state/claim-${1:?}"; }
data_ok() { [ "$(df -BG --output=avail /data | tail -1 | tr -dc 0-9)" -ge $DATA_MIN_GB ]; }

fetch_verify() {   # <slot>: per-slot work dir (src_fetch / src_verify pick placement/<this host>.tsv)
  local s=$1 t0 S=$W/stage/$1
  mkdir -p $S/placement; cp $W/LISTING.json $S/; cp $W/placement/$s.tsv $S/placement/$ME.tsv
  if [[ $RAW == /data/* ]]; then
    while ! data_ok; do say "$s: /data free $(df -BG --output=avail /data | tail -1 | tr -d ' ') < ${DATA_MIN_GB}G: fetch paused"; sleep 120; done
  fi
  for try in 1 2 3; do
    t0=$(date +%s)
    HF_RATE=${HF_RATE:-150} THREADS_HF=${THREADS_HF:-64} $N $PYV -I $A/src_fetch.py $S $RAW/$s > $W/logs/fetch-$s.log 2>&1 && { rec fetch $s "$(( $(date +%s) - t0 ))s"; break; }
    say "$s: fetch try $try failed (rc $?), retrying"; [ $try = 3 ] && return 1
  done
  t0=$(date +%s)
  VERIFY_THREADS=$CPT $N $PYV -I $A/src_verify.py $S $RAW/$s > $W/logs/verify-$s.log 2>&1 || { say "$s: VERIFY FAILED ($W/logs/verify-$s.log)"; return 1; }
  rec verify $s "$(( $(date +%s) - t0 ))s"; touch $W/state/fetched-$s.ok
}

scan() {
  local s=$1 t0=$(date +%s)
  env FRESH_SRC=$RAW/$s/jobs FRESH_SCAN=$W/scan FRESH_HOST=$s OMP_NUM_THREADS=1 $N $PYQ -I /data/home/amartyavarma/ppm/code-811/fresh/fresh_pose.py scan --workers $WORKERS \
    > $W/logs/scan-$s.log 2>&1 && [ -f $W/scan/$s.json ] || { say "$s: SCAN FAILED ($W/logs/scan-$s.log)"; return 1; }
  rec scan $s "$(( $(date +%s) - t0 ))s"
  echo "$(pt) $LANE@$ME $(python3 -c "import json;d=json.load(open('$W/scan/$s.json'));print(d['segments'],'segments',round(d['frames']/12.5/3600),'h',d['releases'],'releases',d['seconds'],'s')")" > $W/state/scan-$s.ok
}

say "lane start: raw $RAW, $CPT CPUs, $WORKERS scan workers, HF_RATE ${HF_RATE:-150}"
s=$(claim); [ -n "$s" ] || { say "nothing to claim"; exit 0; }
say "$s: claimed; fetching"
fetch_verify $s || { unclaim $s; exit 1; }
while :; do
  nxt=""; FP=""
  if [ ! -f $W/state/STOP_LANES ]; then
    nxt=$(claim)
    if [ -n "$nxt" ]; then say "$nxt: claimed; prefetching while $s scans"; fetch_verify $nxt & FP=$!; fi
  fi
  say "$s: scanning"
  if ! scan $s; then [ -n "$FP" ] && wait $FP; unclaim $s; [ -n "$nxt" ] && unclaim $nxt; exit 1; fi
  rm -rf "${RAW:?}/${s:?}"; say "$s: scanned ($(cut -d' ' -f5- $W/state/scan-$s.ok)); raw deleted; /data free $(df -BG --output=avail /data | tail -1 | tr -d ' ')"
  [ -n "$nxt" ] || break
  wait $FP || { say "$nxt: fetch/verify failed"; unclaim $nxt; exit 1; }
  s=$nxt
done
n=$(ls $W/state/scan-*.ok 2>/dev/null | wc -l); say "lane done; scans complete $n/16"
if [ "$n" = 16 ] && mkdir $W/state/plan.submitted 2>/dev/null; then
  J=$(sbatch --parsable $B/plan_offhost.sbatch); echo $J > $W/state/plan.submitted/job; say "all 16 scans done -> plan job $J"
fi
