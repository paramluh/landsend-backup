#!/bin/bash
# 10-26 (main): CO-LOCATION TEST on the hot unit 30666 (slinky-18 4-5): its hot mono worker (both GPUs) + its render lane on GPU 5 (the
# worker's ~41 GB GPU; lane role param-test = its own queue render/jobs_test -> render/out_test, never user jobs). Only while the user's
# queues are empty; aborts + cleans up within 5 s when a user run arrives. Test runs are labelled "co-location test (aloedemo)"; their
# Param renders are parked (job file moved to render/jobs_parked_test, run state "parked") so they never take a user render lane.
# Phases: A solo generation (2 test runs, lane idle) | B generation + a 40 s render on the lane at the same time. nvidia-smi every 0.5 s.
G=/data/home/amartyavarma/ppm/aloe-posedup-live/genweb; GR=$G/render; J=30666; N=slinky-18; S=${1:-/tmp/coloc}; mkdir -p $S $GR/jobs_parked_test
U=$(head -1 $G/URL.txt); U=${U%/}; FEED=$G/runs/20261026-124442-69b7/feed.jsonl; LBL="co-location test (aloedemo)"; TESTS=""
log(){ echo "$(date -u +%T) $*"; }
is_test(){ grep -q "$LBL" $G/runs/$1/request.json 2>/dev/null; }
user_busy(){ local r d; for r in $G/runs/*/READY; do [ -e "$r" ] || continue; d=$(basename $(dirname $r)); is_test $d || return 0; done
  for r in $G/runs/*/status.json; do grep -q '"state": "running"' "$r" 2>/dev/null || continue; d=$(basename $(dirname $r)); is_test $d || return 0; done
  local f b; for f in $GR/jobs/*.feed.jsonl $GR/jobs_vansh/*.feed.jsonl $GR/jobs_vansh50/*.feed.jsonl; do [ -e "$f" ] || continue; b=$(basename "$f" .feed.jsonl); [ -f $GR/out/$b.mp4 ] && continue; is_test ${b%%.*} || return 0; done
  return 1; }
park(){ local r; for r in $TESTS; do [ -f $GR/jobs/$r.feed.jsonl ] && [ ! -f $GR/out/$r.claim ] && mv $GR/jobs/$r.feed.jsonl $GR/jobs_parked_test/
  python3 - "$G/runs/$r/status.json" <<'PY' 2>/dev/null
import json,sys,os
p=sys.argv[1]; d=json.load(open(p))
if d.get("state")=="done": d["state"]="parked"; d["msg"]="co-location test run (render parked)"; json.dump(d,open(p+".tmp","w")); os.replace(p+".tmp",p)
PY
  done; }
abort(){ log "ABORT: a user run arrived"; local r; for r in $TESTS; do curl -s -m 5 -X POST "$U/api/cancel/$r" >/dev/null; done; rm -f $GR/jobs_test/coloc-*.feed.jsonl; park; kill $SP 2>/dev/null; exit 3; }
gen(){ curl -s -m 15 -X POST -H 'content-type: application/json' "$U/api/gen" -d "{\"model\":\"monoplex\",\"seed\":$1,\"clip_s\":40,\"stream\":false,\"prefill\":{\"person\":\"01_ab63d745\",\"s\":1.92,\"src\":\"start\",\"pose\":true,\"audio\":true,\"words\":true},\"prompt\":{\"voice\":\"01_ab63d745\",\"persona\":\"\"},\"params\":{},\"label\":\"$LBL\"}" | python3 -c "import json,sys;print(json.load(sys.stdin)['run_id'])"; }
waitruns(){ local r ok; while true; do user_busy && abort; ok=1; for r in "$@"; do grep -qE '"state": "(done|error|parked)"' $G/runs/$r/status.json 2>/dev/null || ok=0; done; park; [ $ok = 1 ] && return; sleep 2; done; }
msf(){ python3 -c "import json;d=json.load(open('$G/runs/$1/summary.json'));print(d.get('ms_per_frame'))" 2>/dev/null; }
until ! user_busy; do sleep 5; done; log "queues empty: start"
srun --jobid $J --overlap -w $N -x slinky-9 -n 1 -c 1 --mem=1G timeout 1200 nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits -i 4,5 -lms 500 > $S/mem.csv 2>&1 & SP=$!
echo "A $(date +%s)" > $S/phase
a1=$(gen 9101); a2=$(gen 9102); TESTS="$a1 $a2"; waitruns $a1 $a2
log "A solo gen: $a1 $(ls $G/runs/$a1 | grep CLAIMED) $(msf $a1) ms/frame | $a2 $(ls $G/runs/$a2 | grep CLAIMED) $(msf $a2) ms/frame"
user_busy && abort; echo "B $(date +%s)" > $S/phase
t=$(date +%s); b1=$(gen 9103); b2=$(gen 9104); TESTS="$TESTS $b1 $b2"; cp $FEED $GR/jobs_test/coloc-$t.feed.jsonl; waitruns $b1 $b2
log "B gen with render: $b1 $(ls $G/runs/$b1 | grep CLAIMED) $(msf $b1) ms/frame | $b2 $(ls $G/runs/$b2 | grep CLAIMED) $(msf $b2) ms/frame"
until [ -f $GR/out_test/coloc-$t.mp4 ]; do user_busy && abort; sleep 2; done
log "B render: $(python3 -c "import json;print(json.load(open('$GR/out_test/coloc-$t.json')).get('render_s'))") s (solo on this lane earlier: 73.9 s)"
sleep 3; kill $SP 2>/dev/null; wait $SP 2>/dev/null; echo "end $(date +%s)" >> $S/phase; park
python3 - "$S/mem.csv" <<'PY'
import sys
mx={}
for ln in open(sys.argv[1]):
    p=[x.strip() for x in ln.split(',')]
    if len(p)!=3 or not p[0].isdigit(): continue
    g,m=int(p[0]),int(p[1]); mx[g]=max(mx.get(g,0),m)
print("peak MiB per GPU (4 = worker ~52 GB, 5 = worker ~41 GB + lane):", mx)
PY
grep -iE "out of memory|CUDA error" $G/logs/worker-mono-$J.log /data/home/amartyavarma/video/dmd-rt/logs/render-flex-pg$J-g5.log | tail -3
log "done"
