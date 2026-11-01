#!/bin/bash
O=/data/home/amartyavarma/ppm/aloe-posedup-live/verify/posecfg-1025
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
bash $O/ref.sh ref-new-off 40; sleep 3
bash $O/ref.sh pcfg-on 40 '{"seed": 4242, "pose_cfg_on": 1}'; sleep 3
bash $O/ref.sh pcfg-switch 40 '{"seed": 4242}' '{"pose_cfg_on": 1}'; sleep 3
source /data/home/amartyavarma/ppm/aloe-posedup-live/env_live.sh
$PY $O/post_cmp.py ref-new-off pcfg-on pcfg-switch > $O/post_cmp.out 2>&1; echo "cmp rc=$?"
for i in $(seq 60); do b=$(curl -s --max-time 5 http://127.0.0.1:9105/proxyhealth | python3 -c "import json,sys; print(json.load(sys.stdin)['backend'].get('busy'))" 2>/dev/null); [ "$b" = "False" ] && break; sleep 5; done
export LD_LIBRARY_PATH=/data/home/amartyavarma/chromelibs/root/usr/lib/x86_64-linux-gnu
mkdir -p $O/page
nice -n 5 /data/home/amartyavarma/venvs/pagecheck/bin/python -I $O/pcfg_pagecheck.py $(cat /data/home/amartyavarma/ppm/aloe-posedup-live/URL.txt) /data/home/amartyavarma/ppm/aloe-m1vox-live/a16861/verify-2min/a_125s.wav $O/page 30 8 > $O/page/out.txt 2> $O/page/err.txt; echo "page rc=$?"
echo POSTDONE
