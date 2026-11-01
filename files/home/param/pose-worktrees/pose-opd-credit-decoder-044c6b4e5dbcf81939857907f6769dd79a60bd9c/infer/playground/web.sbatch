#!/bin/bash
#SBATCH -J genweb-web
#SBATCH -x slinky-9
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -t 72:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-posedup-live/genweb/logs/%x-%j.out
# MONOPLEX PLAYGROUND web server (:9205, no GPU) + a cloudflared quick tunnel -> genweb/URL.txt (plain URL, no auth). Restart loops.
G=/data/home/amartyavarma/ppm/aloe-posedup-live/genweb; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python; CF=/home/amartyavarma/bin/cloudflared
( while true; do $PY -u $G/server.py --port 9205 >> $G/logs/web-$SLURM_JOB_ID.log 2>&1; sleep 2; done ) &
( while true; do L=$G/logs/cloudflared-$SLURM_JOB_ID-$(date +%s).log
    $CF tunnel --no-autoupdate --protocol http2 --url http://127.0.0.1:9205 > $L 2>&1 & CP=$!
    for i in $(seq 90); do u=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' $L 2>/dev/null | head -1); [ -n "$u" ] && { echo "$u" > $G/URL.txt; echo "URL $u $(date)"; break; }; sleep 2; done
    wait $CP; sleep 5; done ) &
wait
