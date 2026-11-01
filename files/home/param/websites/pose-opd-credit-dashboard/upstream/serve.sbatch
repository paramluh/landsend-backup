#!/bin/bash
#SBATCH -J curves-serve -N1 -n1 -c2 --mem=2G -t 7-00:00:00 -o curves-serve-%j.log
# Curves page server (CPU only): Range-capable static server on 127.0.0.1:$PORT over CURVES_DIR + a Cloudflare quick tunnel.
# The public URL is written to $CURVES_DIR/TUNNEL_URL; the page is <URL>/stage3_<PAGE>.html. Plain URL, no auth: anyone with the link can read it.
#   CURVES_DIR=$PWD sbatch -w <cpu node> serve.sbatch       (cloudflared binary: CLOUDFLARED, default ~/bin/cloudflared)
: "${CURVES_DIR:?export CURVES_DIR=<this curves/ folder (the web root)>}"
PORT=${PORT:-8787}; CF=${CLOUDFLARED:-$HOME/bin/cloudflared}; PY=${PY:-python3}
cd "$CURVES_DIR" || exit 1
$PY $CURVES_DIR/range_http.py $PORT $CURVES_DIR >/dev/null 2>&1 &
sleep 2
while true; do
  $CF tunnel --no-autoupdate --url http://127.0.0.1:$PORT 2>&1 | while read -r l; do u=$(echo "$l" | grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com'); [ -n "$u" ] && { echo "$u" > $CURVES_DIR/TUNNEL_URL; echo "tunnel $u $(date)"; }; done
  sleep 5
done
