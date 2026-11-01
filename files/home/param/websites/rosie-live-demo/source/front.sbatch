#!/bin/bash
#SBATCH -N1 -n1 -c 2
#SBATCH --mem=4G
#SBATCH -t 7-00:00:00
# 10-29 livecall (OPD A/B): always-up front = the 10-24 clone's proxy.py (no-auth, token injected upstream, LB by model from lb.json) on :8917
# + a cloudflared quick tunnel -> URL.txt. CPU only, so the URL survives backend restarts.  sbatch -w slinky-10 -J lc-front front.sbatch
S=/data/home/amartyavarma/ppm/opd/livecall/srv; L=/data/home/amartyavarma/ppm/opd/livecall; CF=/home/amartyavarma/bin/cloudflared; TOK=$HOME/.s4live_token; ID=${SLURM_JOB_ID:-x}
source $S/env_live.sh
[ -s "$TOK" ] && [ "$(stat -c %a "$TOK")" = 600 ] || { echo "ABORT: token file"; exit 1; }
( while true; do $PY -u $S/proxy.py --port 8917 --backend 8918 --token-file $TOK --sessions $L/state/base1/sessions --lb $L/lb.json >> $L/logs/proxy-$ID.log 2>&1; sleep 0.5; done ) &
( while true; do
    LG=$L/logs/cloudflared-$ID-$(date +%s).log
    $CF tunnel --no-autoupdate --url http://127.0.0.1:8917 > $LG 2>&1 &
    CP=$!
    for i in $(seq 90); do u=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' $LG 2>/dev/null | head -1); [ -n "$u" ] && { echo "$u" > $L/URL.txt; echo "tunnel URL $u $(date)"; break; }; sleep 2; done
    wait $CP; sleep 5
  done ) &
wait
