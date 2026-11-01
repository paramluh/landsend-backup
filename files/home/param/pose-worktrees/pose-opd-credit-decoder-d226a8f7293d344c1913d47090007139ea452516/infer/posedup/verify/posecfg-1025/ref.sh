#!/bin/bash
# pose-CFG work (10-25): reference sessions on the CURRENT (pre-posecfg) backend: fixed seed + fixed mic wav, defaults, NON-PREEMPTING test client.
# usage (inside the demo job): bash ref.sh <tag> [secs] [params json] [live_set json]
H=/data/home/amartyavarma/ppm/aloe-posedup-live; O=$H/verify/posecfg-1025; TAG=$1; SECS=${2:-40}; PRM=${3:-'{"seed": 4242}'}; LS=${4:-}
source $H/env_live.sh
U=$(cat $H/URL.txt); W=/data/home/amartyavarma/ppm/aloe-m1vox-live/a16861/verify-2min/a_125s.wav
for i in $(seq 60); do b=$(curl -s --max-time 5 http://127.0.0.1:9105/proxyhealth | python3 -c "import json,sys; print(json.load(sys.stdin)['backend'].get('busy'))" 2>/dev/null); [ "$b" = "False" ] && break; sleep 5; done
echo "$TAG start $(date) busy=$b"
if [ -n "$LS" ]; then nice -n 5 $PY -I $H/test_client.py $U $W $O/$TAG $SECS - "$PRM" "$LS" > $O/$TAG.log 2>&1
else nice -n 5 $PY -I $H/test_client.py $U $W $O/$TAG $SECS - "$PRM" > $O/$TAG.log 2>&1; fi
echo "$TAG rc=$? $(date)"
SID=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['session'])" $O/$TAG/result.json 2>/dev/null)
[ -n "$SID" ] && cp $H/sessions/$SID/session.npz $H/sessions/$SID/session.json $O/$TAG/ && echo "copied session $SID"
