#!/bin/bash
# 10-26 (main): deploy in an EMPTY-QUEUE window only, make-before-break, one piece at a time:
#   render_service lanes reload (videort's encode-while-rendering) one lane at a time, the staged flex_lib.sh.next (Param-before-Vansh worker flex)
#   and the mono flex unit's controller, then the playground hybrid (no call open, nothing rendering on it). Every step re-checks the queue.
G=/data/home/amartyavarma/ppm/aloe-posedup-live/genweb; GR=$G/render; L=/data/home/amartyavarma/video/dmd-rt/logs
log(){ echo "$(date -u +%T) $*"; }
queue_empty(){ local f b r
  ls $G/runs/*/READY >/dev/null 2>&1 && return 1
  for r in $G/runs/*/status.json; do grep -q '"state": "running"' "$r" 2>/dev/null && return 1; done
  for f in $GR/jobs/*.feed.jsonl $GR/jobs_vansh/*.feed.jsonl $GR/jobs_vansh50/*.feed.jsonl; do [ -e "$f" ] || continue; b=$(basename "$f" .feed.jsonl); [ -f $GR/out/$b.mp4 ] || return 1; done
  return 0; }
wait_empty(){ local n=0; until queue_empty; do [ $((n % 60)) = 0 ] && log "waiting for an empty queue ($1)"; n=$((n + 1)); sleep 5; done; }
gpu_idle(){ local J=$1 g=$2 N u ok=0; N=$(squeue -h -j $J -o %N); for i in $(seq 40); do
  u=$(srun --jobid $J --overlap -w $N -x slinky-9 -n 1 -c 1 --mem=1G nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $g 2>/dev/null | tr -d ' ')
  [ -n "$u" ] && [ "$u" -lt 10 ] && ok=$((ok + 1)) || ok=0; [ $ok -ge 3 ] && return 0; sleep 3; done; return 1; }
# restart ONE render_service lane (job, gpu, log file): kill its process (between jobs), its unit's loop restarts it; wait for engine ready
lane_reload(){ local J=$1 g=$2 lf=$3 N n0; N=$(squeue -h -j $J -o %N)
  wait_empty "lane $J g$g"; gpu_idle $J $g || { log "lane $J g$g not idle; skipped"; return 1; }
  n0=$(grep -c "engine ready" $lf 2>/dev/null)
  srun --jobid $J --overlap -w $N -x slinky-9 -n 1 -c 1 --mem=1G bash -c "for p in \$(scontrol listpids $J | awk 'NR>1{print \$1}'); do
      c=\$(tr '\0' ' ' < /proc/\$p/cmdline 2>/dev/null); e=\$(tr '\0' '\n' < /proc/\$p/environ 2>/dev/null | grep '^CUDA_VISIBLE_DEVICES=' | cut -d= -f2)
      case \"\$c\" in *python*rt/render_service.py*) [ \"\$e\" = $g ] && { pg=\$(ps -o pgid= -p \$p | tr -d ' '); if [ \"\$pg\" = \"\$p\" ]; then kill -- -\$pg; else pkill -P \$p; kill \$p; fi; echo killed \$p; };; esac; done"
  for i in $(seq 60); do [ "$(grep -c 'engine ready' $lf 2>/dev/null)" -gt "$n0" ] && { log "lane $J g$g reloaded"; return 0; }; sleep 5; done
  log "lane $J g$g: no engine ready after 5 min"; return 1; }
wait_empty start
# 1. renderer pair 30668 (slinky-4 0-1): g1 first (non-warm lane), then the warm g0 once g1 is back up (make-before-break)
if lane_reload 30668 1 $L/render-flex-pg30668-g1.log; then                 # make-before-break: g1 must be a warm Param lane again before g0 goes
  for i in $(seq 120); do [ -n "$(find $G/flex/warm -name '30668-g1' -newermt '-20 seconds' 2>/dev/null)" ] && break; sleep 5; done
  [ -n "$(find $G/flex/warm -name '30668-g1' -newermt '-20 seconds' 2>/dev/null)" ] && lane_reload 30668 0 $L/render-flex-pg30668-g0.log || log "g1 not warm on Param; g0 left as is"
fi
# 2. posefix-only lanes of pool unit 30667 (slinky-4 6-7) and the hot unit's test lane (30666 g5)
for g in 6 7; do f=$(ls -t $L/render-flex-pg30667-g$g.log 2>/dev/null | head -1); [ -n "$f" ] && lane_reload 30667 $g $f; done
lane_reload 30666 5 $L/render-flex-pg30666-g5.log
# 3. the staged flex logic, then the mono flex unit's controller (another mono worker, mono30666, is up: make-before-break)
wait_empty "flex_lib"; cp -p $G/flex_lib.sh $G/flex_lib.sh.bak-pre-next-1026; cp -p $G/flex_lib.sh.next $G/flex_lib.sh; log "flex_lib.sh.next deployed"
J=$(squeue -h -u $USER -n genweb-u-mono -o %i | head -1); grep -q "loaded + captured" $G/logs/worker-mono-30666.log && [ -n "$J" ] && { wait_empty "mono unit $J"; bash $G/roll_flexpair.sh $J mono; }
# 4. the playground hybrid (30664): only with no call open on P and nothing rendering on it
for i in $(seq 720); do
  s=$(srun --jobid 30664 --overlap -w slinky-4 -x slinky-9 -n 1 -c 1 --mem=1G curl -s -m 5 http://127.0.0.1:8963/api/inputs | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('state'), d.get('busy'), d.get('kind'))" 2>/dev/null)
  [ "$s" = "ready False None" ] && queue_empty && break; sleep 5; done
if [ "$s" = "ready False None" ]; then
  srun --jobid 30664 --overlap -w slinky-4 -x slinky-9 -n 1 -c 1 --mem=1G bash -c "for p in \$(scontrol listpids 30664 | awk 'NR>1{print \$1}'); do c=\$(tr '\0' ' ' < /proc/\$p/cmdline 2>/dev/null); case \"\$c\" in *python*demo/server.py*) kill \$p; echo killed \$p;; esac; done"
  log "hybrid restarting"; for i in $(seq 120); do s=$(srun --jobid 30664 --overlap -w slinky-4 -x slinky-9 -n 1 -c 1 --mem=1G curl -s -m 5 http://127.0.0.1:8963/api/inputs | python3 -c "import json,sys; print(json.load(sys.stdin).get('state'))" 2>/dev/null); [ "$s" = ready ] && break; sleep 10; done; log "hybrid $s"
else log "hybrid never idle (last: $s); not restarted"; fi
log "done"
