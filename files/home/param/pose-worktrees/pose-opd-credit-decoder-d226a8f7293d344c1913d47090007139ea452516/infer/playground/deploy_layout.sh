#!/bin/bash
# 10-26 (main/user): FIXED-MINIMUM layout, FALLBACK version (the co-location test failed: a Param lane on the hot worker's GPU slowed its
# generation 58.7 -> 95.6 ms/frame). Empty-queue windows only, make-before-break, each step drain-safe (roll_flexpair.sh):
#   1. slinky-18 4-5: 30666 -> hot mono worker A, no lane            (mono30674 serves meanwhile)
#   2. slinky-18 0-1: mono flex 30674 -> duplex unit, FLEX_NO_MONO=1  (only once A is loaded)
#   3. slinky-4 6-7:  pool render 30667 -> Vansh-only pair            (never Param or generation there)
#   4. slinky-4 0-1:  render pair 30668 -> dedicated Param pair       (only with the hybrid ready + idle as the other Param renderer)
G=/data/home/amartyavarma/ppm/aloe-posedup-live/genweb; GR=$G/render
log(){ echo "$(date -u +%T) $*"; }
queue_empty(){ local f b r; ls $G/runs/*/READY >/dev/null 2>&1 && return 1
  for r in $G/runs/*/status.json; do grep -q '"state": "running"' "$r" 2>/dev/null && return 1; done
  for f in $GR/jobs/*.feed.jsonl $GR/jobs_vansh/*.feed.jsonl $GR/jobs_vansh50/*.feed.jsonl; do [ -e "$f" ] || continue; b=$(basename "$f" .feed.jsonl); [ -f $GR/out/$b.mp4 ] || return 1; done; return 0; }
wait_empty(){ local n=0; until queue_empty; do [ $((n % 60)) = 0 ] && log "waiting for an empty queue ($1)"; n=$((n + 1)); sleep 5; done; }
jid(){ squeue -h -u $USER -n $1 -o %i | head -1; }
loaded(){ grep -q "loaded + captured" $G/logs/worker-mono-$1.log 2>/dev/null; }
[ -n "$(jid genweb-u-hot)" ] || { wait_empty "step 1"; log "1: 30666 -> hot A (no lane)"; LANE_ROLE=none bash $G/roll_flexpair.sh 30666 hot; }   # step 1 done 13:57 (30675)
A=$(jid genweb-u-hot); for i in $(seq 120); do loaded $A && break; sleep 5; done; loaded $A && log "hot A $A loaded" || { log "hot A not loaded; stop"; exit 1; }
wait_empty "step 2"; M=$(jid genweb-u-mono); log "2: $M -> duplex unit"; DUP_PREFER_MONO=1 bash $G/roll_flexpair.sh $M dup   # 10-26 main: the duplex pair is a hot mono worker while no duplex run waits
wait_empty "step 3"; P=$(squeue -h -u $USER -n genweb-pool-w -w slinky-4 -o %i | head -1); [ -n "$P" ] && { log "3: pool $P -> Vansh-only pair"; bash $G/roll_flexpair.sh $P vanshonly; }
for i in $(seq 720); do
  s=$(srun --jobid 30664 --overlap -w slinky-4 -x slinky-9 -n 1 -c 1 --mem=1G curl -s -m 5 http://127.0.0.1:8963/api/inputs | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('state'), d.get('busy'))" 2>/dev/null)
  [ "$s" = "ready False" ] && queue_empty && break; sleep 5; done
[ "$s" = "ready False" ] || { log "hybrid never ready + idle; step 4 not done"; exit 1; }
R=$(jid genweb-u-vansh); log "4: render pair $R -> dedicated Param pair (hybrid ready as the other Param renderer)"; bash $G/roll_flexpair.sh $R paramonly
log "done"; squeue -h -u $USER -o "%i %j %N %T" | grep -E "genweb-(u-|dup|pool)"
