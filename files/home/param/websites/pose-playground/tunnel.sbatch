#!/bin/bash
#SBATCH -J genweb-tunnel
#SBATCH -x slinky-9
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -t 7-00:00:00
#SBATCH -o logs/%x-%j.out
# 10-28 (ppmguard deploy agent, user order "host xabi's monoplex + raw monoplex"): = web.sbatch's cloudflared loop ONLY (since 10-26 the
# web job ran only cloudflared: front.py on :9205 belongs to genweb-app / app_sup.sh, so web.sbatch's own server.py on :9205 must NOT run).
# Must run on the SAME node as genweb-app (the tunnel points at 127.0.0.1:9205).   sbatch -w <app node> tunnel.sbatch
: "${CANOPY_ROOT:?export CANOPY_ROOT=<cluster home that holds ppm/ venvs/ models/; see README>}"
G=${GENWEB:-$CANOPY_ROOT/ppm/aloe-posedup-live/genweb}; CF=$HOME/bin/cloudflared
( while true; do L=$G/logs/cloudflared-$SLURM_JOB_ID-$(date +%s).log
    $CF tunnel --no-autoupdate --protocol http2 --url http://127.0.0.1:9205 > $L 2>&1 & CP=$!
    for i in $(seq 90); do u=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' $L 2>/dev/null | head -1); [ -n "$u" ] && { echo "$u" > $G/URL.txt; echo "URL $u $(date)"; break; }; sleep 2; done
    wait $CP; sleep 5; done ) &
wait
