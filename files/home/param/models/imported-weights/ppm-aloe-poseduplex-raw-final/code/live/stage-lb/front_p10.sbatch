#!/bin/bash
#SBATCH -N1 -n1 -c 2
#SBATCH --mem=4G
#SBATCH -t 30-00:00:00
# 10-28 duplexhost (user "host the duplex demo"): copy of front_s4.sbatch for ONE pose-duplex backend = instance P on slinky-10:9126 (job posedup-lb-P).
# Writes URL.txt / ALT_URLS.txt directly (as demo.sbatch), so the playground Live mic (livedup_api reads URL.txt) follows tunnel restarts.
# (front 1 = /avatar/ at 127.0.0.1:8960 is local-only there): proxy.py --port 9145 with lb_p10.json (the slinky-4 instance + the playground P
# entry, whose fallback is that instance) + a cloudflared quick tunnel -> URL.txt + the http2 alt tunnel (tools/tunnel2.sh) -> ALT_URLS.txt.
#   sbatch -w slinky-10 -x slinky-9 -c 2 --mem=4G -t 7-00:00:00 -J posedup-front-p10 -o .../logs/front-p10-%j.out stage-lb/front_p10.sbatch
H=/data/home/amartyavarma/ppm/aloe-posedup-live; CF=/home/amartyavarma/bin/cloudflared; TOK=$HOME/.s4live_token; FRONT=9145; BACK=9136
source $H/env_live.sh
# every avatar mount -> front 1 (127.0.0.1:8960 on slinky-10): fronts B / P (8940 / 8945) die with videort job 29805 at go; the playground page
# preloads /avatar3/, so that name stays and points at front 1 too
export PPMLIVE_AVATAR_UPSTREAM=http://127.0.0.1:8960 PPMLIVE_AVATAR2_UPSTREAM=http://127.0.0.1:8960 PPMLIVE_AVATAR3_UPSTREAM=http://127.0.0.1:8960
URLF=$H/URL.txt; ID=p10-${SLURM_JOB_ID:-x}
[ -s "$TOK" ] && [ "$(stat -c %a "$TOK")" = 600 ] || { echo "ABORT: token file"; exit 1; }
echo "front $FRONT lb $H/stage-lb/lb_p10.json host $(hostname) $(date)"
( while true; do $PY -u $H/proxy.py --port $FRONT --backend $BACK --token-file $TOK --sessions $H/stage-lb/instP/sessions --lb $H/stage-lb/lb_p10.json >> $H/logs/proxy-$ID.log 2>&1; sleep 0.5; done ) &
( while true; do
    L=$H/logs/cloudflared-$ID-$(date +%s).log
    $CF tunnel --no-autoupdate --url http://127.0.0.1:$FRONT > $L 2>&1 &
    CP=$!
    for i in $(seq 90); do u=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' $L 2>/dev/null | head -1); [ -n "$u" ] && { echo "$u" > $URLF; echo "tunnel URL $u $(date)"; break; }; sleep 2; done
    wait $CP; sleep 5
  done ) &
bash $H/tools/tunnel2.sh 2592000 $FRONT >> $H/logs/tunnel2-$ID.out 2>&1 &
# 10-30 dupfreeze: pause the playground render lane(s) on the avatar GPUs (RS_PAUSE_FILE=render/PAUSE_LIVE_S4) during Live mic calls; without it the previous call's replay render shares slinky-4 GPU 2 with the live avatar -> video freezes
bash $H/genweb/stage-v2pure/live_pause_watch.sh >> $H/genweb/logs/live-pause-watch.out 2>&1 &
wait
