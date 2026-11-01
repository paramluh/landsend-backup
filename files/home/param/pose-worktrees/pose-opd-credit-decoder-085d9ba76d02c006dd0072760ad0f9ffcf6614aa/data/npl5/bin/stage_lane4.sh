#!/bin/bash
# npl5 OFF-HOST staging lane v4 (10-27 07:50 PT, main's /data RAW CAP enforced in code): raw under /data -> before a slot's first file the lane
# reserves it under flock (W/state/rawcap.lock) only if committed raw (full share of every slot dir in W/raw + reservations) + this slot <= RAWCAP_GB
# (155) and /data free >= DATA_MIN_GB (600); bin/src_fetch_capped.py re-checks both before EVERY file. Raw deleted right after the scan products
# are written; reservation released then. PREFER="slot .." claims those first (e.g. a cancelled lane's half-fetched raw).
# v3 (= v2 + LANE_DEADLINE_EPOCH: never claims a slot it cannot finish (SLOT_SECS) before the
# deadline; extra lanes added by bin/lane_adder.sh when slinky-10/18 CPUs free up). v2 (10-27 06:05 PT: + the orig_h/orig_w SIDECAR of user change #2, written right after verify, before
# the scan and long before the raw is deleted; + ADOPT="slot .." to take over a stopped lane's claims; + per-lane stop file STOP_LANE_<name>).
# v1 (stage_lane.sh) stays untouched while lane A runs it (its sidecars come from bin/sidecar_watch.sh). Runs inside its own sbatch allocation on a SPARE (stage_lane.sbatch),
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
DATA_MIN_GB=${DATA_MIN_GB:-600}
if [ "$RAW" = auto ]; then   # this spare's /scratch if it keeps >= 200 GB free (2 slot shares in flight + a 45 GB floor), else rolling on /data
  F=$(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9)
  RAW=$([ "$F" -ge 200 ] && echo /scratch/amartyavarma/npl5-stage || echo $W/raw)
  echo "== $(TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT') [$LANE@$ME] RAW auto: /scratch free ${F}G -> $RAW" | tee -a $W/logs/lanes.log
fi
mkdir -p $W/state $W/logs $W/scan $RAW
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
say() { echo "== $(pt) [$LANE@$ME] $*" | tee -a $W/logs/lanes.log; }
rec() { printf "%s\t%s\t%s\t%s\t%s\n" "$1" "$2" "$3" "$LANE@$ME" "$(pt)" >> $W/TIMES.tsv; }
claim() { for s in ${PREFER:-}; do [ -f $W/state/scan-$s.ok ] && continue; mkdir $W/state/claim-$s 2>/dev/null && { echo $LANE@$ME > $W/state/claim-$s/by; echo $s; return; }; done
  if [ -n "${LANE_DEADLINE_EPOCH:-}" ] && [ $(( $(date +%s) + ${SLOT_SECS:-2700} )) -gt $LANE_DEADLINE_EPOCH ]; then return; fi
  for s in ${ADOPT:-}; do [ -f $W/state/scan-$s.ok ] && continue; grep -qs "^$LANE@" $W/state/claim-$s/by || continue; [ -f $W/state/adopted-$s ] && continue
    echo $LANE@$ME > $W/state/claim-$s/by; touch $W/state/adopted-$s; echo $s; return; done
  for s in $SLOTS; do [ -f $W/state/scan-$s.ok ] && continue; mkdir $W/state/claim-$s 2>/dev/null && { echo $LANE@$ME > $W/state/claim-$s/by; echo $s; return; }; done; }
unclaim() { rm -rf "${W:?}/state/claim-${1:?}"; }
data_ok() { [ "$(df -BG --output=avail /data | tail -1 | tr -dc 0-9)" -ge $DATA_MIN_GB ]; }

fetch_verify() {   # <slot>: per-slot work dir (src_fetch / src_verify pick placement/<this host>.tsv)
  local s=$1 t0 S=$W/stage/$1
  mkdir -p $S/placement; cp $W/LISTING.json $S/; cp $W/placement/$s.tsv $S/placement/$ME.tsv
  if [[ $RAW == /data/* ]]; then   # RAW CAP gate (main 10-27): reserve this slot only if it fits under the cap and /data free >= DATA_MIN_GB
    local k=0
    until ( flock -x 9; python3 - "$W" "$s" "${RAWCAP_GB:-155}" "${DATA_MIN_GB:-600}" <<'PYEOF'
import json, os, sys
W, s, cap, dmin = sys.argv[1], sys.argv[2], float(sys.argv[3]) * 1e9, float(sys.argv[4]) * 1e9
sh = {h: float(v["GB"]) * 1e9 for h, v in json.load(open(f"{W}/PLACEMENT.json"))["hosts"].items()}
sl = set(os.listdir(f"{W}/raw")) if os.path.isdir(f"{W}/raw") else set()
sl |= {f[len("rawres-"):] for f in os.listdir(f"{W}/state") if f.startswith("rawres-")}
st = os.statvfs("/data"); free = st.f_bavail * st.f_frsize
com = sum(sh.get(x, 77e9) for x in sl | {s})
if com <= cap + 1e9 and free >= dmin:
    open(f"{W}/state/rawres-{s}", "w").write(f"{sh.get(s, 77e9):.0f}\n"); sys.exit(0)
print(f"committed raw with {s} {com / 1e9:.0f} GB (cap {cap / 1e9:.0f}), /data free {free / 1e9:.0f} GB (need {dmin / 1e9:.0f})", flush=True); sys.exit(1)
PYEOF
    ) 9>$W/state/rawcap.lock > $W/logs/rawcap-$s.last 2>&1; do
      [ $((k % 5)) = 0 ] && say "$s: RAW CAP wait: $(cat $W/logs/rawcap-$s.last | tail -n 1)"; k=$((k + 1)); sleep 60
    done
    say "$s: reserved under the raw cap ($(cat $W/state/rawres-$s | awk '{printf "%.1f GB", $1/1e9}'))"
  fi
  for try in 1 2 3; do
    t0=$(date +%s)
    RAW_ON_DATA=$([[ $RAW == /data/* ]] && echo 1 || echo 0) RAWCAP_GB=${RAWCAP_GB:-155} DATA_MIN_GB=${DATA_MIN_GB:-600} NPL5_W=$W \
      HF_RATE=${HF_RATE:-150} THREADS_HF=${THREADS_HF:-64} $N $PYV -I $B/src_fetch_capped.py $S $RAW/$s > $W/logs/fetch-$s.log 2>&1 && { rec fetch $s "$(( $(date +%s) - t0 ))s"; break; }
    say "$s: fetch try $try failed (rc $?), retrying"; [ $try = 3 ] && return 1
  done
  t0=$(date +%s)
  VERIFY_THREADS=$CPT $N $PYV -I $A/src_verify.py $S $RAW/$s > $W/logs/verify-$s.log 2>&1 || { say "$s: VERIFY FAILED ($W/logs/verify-$s.log)"; return 1; }
  rec verify $s "$(( $(date +%s) - t0 ))s"; touch $W/state/fetched-$s.ok
  if [ ! -f $W/state/origwh-$s.ok ]; then
    t0=$(date +%s)
    $N $PYQ -I $B/sidecar_origwh.py $RAW/$s $s --workers ${SIDECAR_WORKERS:-2} > $W/logs/origwh-$s.log 2>&1 || { say "$s: SIDECAR FAILED ($W/logs/origwh-$s.log)"; return 1; }
    rec origwh $s "$(( $(date +%s) - t0 ))s"; echo "$(pt) $LANE@$ME $(( $(date +%s) - t0 ))s" > $W/state/origwh-$s.ok
  fi
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
  if [ ! -f $W/state/STOP_LANES ] && [ ! -f $W/state/STOP_LANE_$LANE ]; then
    nxt=$(claim)
    if [ -n "$nxt" ]; then say "$nxt: claimed; prefetching while $s scans"; fetch_verify $nxt & FP=$!; fi
  fi
  say "$s: scanning"
  if ! scan $s; then [ -n "$FP" ] && wait $FP; unclaim $s; [ -n "$nxt" ] && unclaim $nxt; exit 1; fi
  rm -rf "${RAW:?}/${s:?}"; rm -f "${W:?}/state/rawres-${s:?}"; say "$s: scanned ($(cut -d' ' -f5- $W/state/scan-$s.ok)); raw deleted; /data free $(df -BG --output=avail /data | tail -1 | tr -d ' ')"
  [ -n "$nxt" ] || break
  wait $FP || { say "$nxt: fetch/verify failed"; unclaim $nxt; exit 1; }
  s=$nxt
done
n=$(ls $W/state/scan-*.ok 2>/dev/null | wc -l); say "lane done; scans complete $n/16"
if [ "$n" = 16 ] && mkdir $W/state/plan.submitted 2>/dev/null; then
  J=$(sbatch --parsable $B/plan_offhost.sbatch); echo $J > $W/state/plan.submitted/job; say "all 16 scans done -> plan job $J"
fi
