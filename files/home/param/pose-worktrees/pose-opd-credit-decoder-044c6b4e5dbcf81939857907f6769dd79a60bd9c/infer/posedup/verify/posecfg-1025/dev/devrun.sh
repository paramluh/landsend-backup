#!/bin/bash
# dev end-to-end: staged server (random weights) + the demo's proxy on dev ports 9305/9306 (slinky-18, dev job), 2 test sessions, then stop both.
DEMO=/data/home/amartyavarma/ppm/aloe-posedup-live; D=$DEMO/verify/posecfg-1025/dev
source $DEMO/env_live.sh
export PPMLIVE_STATE_DIR=$D/state TRITON_CACHE_DIR=/scratch/amartyavarma/triton-cache-posecfg-dev PPMLIVE_SWAP_TO=
W=/data/home/amartyavarma/ppm/aloe-m1vox-live/a16861/verify-2min/a_125s.wav
$PY -u $DEMO/proxy.py --port 9305 --backend 9306 --token-file $D/devtoken --sessions $D/sessions > $D/devproxy.log 2>&1 & PP=$!
$PY -u $D/devserver.py $D/ci498.pkl 9306 $D/devtoken $D/sessions > $D/devserver.log 2>&1 & SP=$!
T0=$(date +%s)
until curl -s --max-time 3 http://127.0.0.1:9305/proxyhealth | grep -q '"state": "ready"'; do sleep 3; kill -0 $SP 2>/dev/null || { echo "server died"; break; }; done
echo "ready after $(( $(date +%s) - T0 )) s"
curl -s --max-time 10 http://127.0.0.1:9305/api/presets > $D/dev_presets.json
python3 -c "import json; d=json.load(open('$D/dev_presets.json')); print([ (r['name'], r.get('type'), r.get('default'), r.get('group')) for r in d['schema'] if r['name'].startswith('pose_cfg')]); print([r.get('latency_note') for r in d['schema'] if r['name']=='pose_cfg_on'])"
$PY -I $DEMO/test_client.py http://127.0.0.1:9305 $W $D/t-switch 30 - '{"seed": 4242}' '{"pose_cfg_on": 1}' > $D/t-switch.log 2>&1; echo "switch rc=$?"
$PY -I $DEMO/test_client.py http://127.0.0.1:9305 $W $D/t-on 20 - '{"seed": 4242, "pose_cfg_on": 1, "cfg_on": 1, "cfg_cb0": 1}' > $D/t-on.log 2>&1; echo "on+audio cfg rc=$?"
curl -s --max-time 10 http://127.0.0.1:9305/api/presets | python3 -c "import json,sys; d=json.load(sys.stdin); print([r.get('latency_note') for r in d['schema'] if r['name']=='pose_cfg_on'])"
kill $SP $PP; wait $SP $PP 2>/dev/null
echo done
