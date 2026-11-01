#!/bin/bash
# 10-25 second quick tunnel to the same front (:9105) over TCP (http2), for the page's auto-reconnect: URL -> ALT_URLS.txt (read by server.py
# at each session start, sent to the page as alt_ws). Restart loop; self-exits after $1 seconds (default 24 h). Usage: tunnel2.sh [secs] [front]
H=/data/home/amartyavarma/ppm/aloe-posedup-live; CF=/home/amartyavarma/bin/cloudflared; END=$(( $(date +%s) + ${1:-86400} )); FRONT=${2:-9105}
while [ $(date +%s) -lt $END ]; do
  L=$H/logs/cloudflared2-${SLURM_JOB_ID:-x}-$(date +%s).log
  timeout $(( END - $(date +%s) )) $CF tunnel --no-autoupdate --protocol http2 --url http://127.0.0.1:$FRONT > $L 2>&1 &
  CP=$!
  for i in $(seq 90); do u=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' $L 2>/dev/null | head -1); [ -n "$u" ] && { echo "$u" > $H/ALT_URLS.txt; echo "tunnel2 URL $u $(date)"; break; }; sleep 2; done
  wait $CP; sleep 5
done
