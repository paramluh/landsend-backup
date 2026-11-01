#!/bin/bash
# 10-25 main (max-parallel): STANDING GPU BACKFILL POOL for the spare nodes slinky-18 / slinky-4. Every 30 s: on each node, if >= 2 GPUs are
# free, start a backfill job there: a playground worker pair (worker.sbatch) when >= 2 playground runs are waiting, else a renderer pair
# (render2.sbatch: playground jobs first, then posefix's batches). Backfill jobs YIELD: if any other job is pending for that node, the newest
# backfill job there is cancelled so it can start. Never slinky-9; explicit --mem. Log: logs/pool.log
: "${CANOPY_ROOT:?export CANOPY_ROOT=<cluster home that holds ppm/ venvs/ models/; see README>}"
G=${GENWEB:-$CANOPY_ROOT/ppm/aloe-posedup-live/genweb}
log(){ echo "[pool $(date -u +%T)] $*" >> $G/logs/pool.log; }
mkdir -p $G/flex/drain
bf_busy(){ local J=$1 c st N GL u; for c in $G/runs/*/CLAIMED.*$J; do [ -e "$c" ] || continue
    st=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('state'))" "$(dirname "$c")/status.json" 2>/dev/null)
    [ "$st" = running ] || [ "$st" = queued ] && return 0; done
  N=$(squeue -h -j $J -o %N); GL=$(scontrol show job -d $J | grep -oP 'IDX:\K[0-9,-]+' | head -1 | sed 's/-/,/')
  u=$(srun --jobid $J --overlap -w $N -x slinky-9 -n 1 -c 1 --mem=1G nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $GL 2>/dev/null | sort -n | tail -1 | tr -d ' ')
  [ -n "$u" ] && [ "$u" -ge 10 ] && return 0; return 1; }
while true; do
  ready=$(ls $G/runs/*/READY 2>/dev/null | wc -l)
  for node in slinky-18 slinky-4; do
    used=$(squeue -h -w $node -t R,CG -o "%b" | grep -o 'gpu:[0-9]*' | cut -d: -f2 | awk '{s+=$1} END {print s+0}')
    free=$((8 - used))
    # 10-27 main: slinky-4 GPUs 6-7 (thermally throttled) never serve user-facing work -> there, count only the free healthy GPUs 0-5
    [ "$node" = slinky-4 ] && free=$(for jj in $(squeue -h -w slinky-4 -t R,CG -o %i); do scontrol show job -d $jj | grep -oP 'IDX:\K[0-9,-]+' | head -1; done | tr '\n' ',' | python3 -c "
import sys
u = set()
for p in sys.stdin.read().strip().split(','):
    if p:
        a, _, b = p.partition('-'); u.update(range(int(a), int(b or a) + 1))
print(len(set(range(6)) - u))")
    pend=$(squeue -h -t PD -o "%j|%n|%b" | grep "|$node|" | grep gpu | grep -v -c "^genweb-pool")
    if [ "$pend" -gt 0 ]; then
      j=$(squeue -h -u $USER -w $node -n genweb-pool-r,genweb-pool-w -o "%i %S" | sort -k2 | tail -1 | cut -d' ' -f1)
      # 10-26: never cancel a backfill with work in flight: drain its workers (claim nothing new), wait for its runs, and for its GPUs to idle
      if [ -n "$j" ]; then
        for w in mono$j dup$j flexmono$j; do touch $G/flex/drain/$w; done
        if bf_busy $j; then log "$node: $pend job(s) pending -> backfill $j draining (work in flight)"
        else scancel $j; for w in mono$j dup$j flexmono$j; do rm -f $G/flex/drain/$w; done; log "$node: $pend job(s) pending -> cancelled idle backfill $j"; fi
      fi
      continue
    fi
    if [ $free -ge 2 ]; then                                   # 10-25 main: idle renders -> hot workers; render only when render jobs wait
      rb=$(for f in $G/render/jobs/*.feed.jsonl; do [ -e "$f" ] || continue; n=$(basename $f .feed.jsonl); [ -f $G/render/out/$n.mp4 ] || [ -f $G/render/out/$n.claim ] || echo x; done | wc -l)
      db=$(grep -l '"model": "duplex"' $(ls $G/runs/*/READY 2>/dev/null | sed 's#/READY$#/request.json#') /dev/null 2>/dev/null | wc -l)   # 10-26: queued duplex runs
      role=mono; [ $rb -ge 3 ] && role=render; [ $db -ge 1 ] && [ $node = slinky-18 ] && ! squeue -h -u $USER -n genweb-dup -t R,PD -o %i | grep -q . && role=dup   # 10-26 dupui: a 2nd duplex worker while >= 2 duplex runs wait (export-498 DCP is on slinky-18 /scratch only)
      j=$(sbatch --parsable -w $node -x slinky-9 --gres=gpu:2 -c 6 --mem=110G -J genweb-pool-w $G/unit.sbatch $role) && log "$node: $free free, render backlog $rb, duplex queue $db -> $role unit $j (throttled GPUs become render)"
    fi
  done
  sleep 30
done
