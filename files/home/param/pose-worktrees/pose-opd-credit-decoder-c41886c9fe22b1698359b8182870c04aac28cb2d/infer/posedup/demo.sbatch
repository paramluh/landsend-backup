#!/bin/bash
#SBATCH -J posedup-live
#SBATCH -N1 -n1 -c 8
#SBATCH --gres=gpu:2
#SBATCH --mem=100G
#SBATCH -t 3-00:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-posedup-live/logs/demo-%j.out
# ALOE POSE-DUPLEX LIVE DEMO (job 30074's exports; aloe-posedup-live): model server (2 GPUs; pose branch on the first) + always-up no-auth
# front proxy (injects ~/.s4live_token upstream) + cloudflared quick tunnel -> the front, all in restart loops inside this job.
# The quick-tunnel URL is NEW on every job start: aloe-posedup-live/URL.txt (plain URL, no token). Separate from aloe-m1vox-live (untouched).
# CK = the export served at start; the server hot-swaps in-process (between sessions) to the newest COMPLETE dir among PPMLIVE_SWAP_TO and the
# lines of $PPMLIVE_STATE_DIR/SWAP_TO.txt. Feed for videort: FEED_URL.txt (first line; 'off' disables) else PPMLIVE_FEED_URL else
# ws://slinky-10:8990/feed (re-read at every session start). Avatar iframe URL: AVATAR_URL.txt; server default avatar person: PERSON.txt.
#   sbatch -w slinky-10 demo.sbatch          DRY=1 bash demo.sbatch (checks only)
#   PRETEST=1: first the GPU engine check tools/pose_ce_test.py (graph vs eager, teacher-forced pose / speech CE vs the run's eval) -> logs/pretest-<job>.json
# Restart only the backend (proxy + tunnel stay up, same URL): kill the server.py python process of this job (restarted in 10 s).
set -u
H=/data/home/amartyavarma/ppm/aloe-posedup-live
CK=${CK:-/scratch/amartyavarma/aloe-posedup-live/export-498-dcp}   # the FINAL export (consolidated bf16 DCP on /data)
export PPMLIVE_SWAP_TO=${PPMLIVE_SWAP_TO:-$CK}
TOK=$HOME/.s4live_token
CF=/home/amartyavarma/bin/cloudflared
FRONT=${FRONT:-9105}; BACK=${BACK:-9106}
export PPMLIVE_STATE_DIR=${PPMLIVE_STATE_DIR:-$H}
URLF=${URLF:-$H/URL.txt}
ID=${SLURM_JOB_ID:-dry}
source $H/env_live.sh

checks() {
  local ok=1
  [ -f "$CK/COMPLETE" ] && echo "ok  checkpoint $CK (COMPLETE)" || { echo "BAD checkpoint $CK has no COMPLETE"; ok=0; }
  [ -s "$TOK" ] && [ "$(stat -c %a "$TOK")" = 600 ] && echo "ok  token file (mode 600)" || { echo "BAD token file $TOK"; ok=0; }
  [ -x "$CF" ] && echo "ok  cloudflared" || { echo "BAD cloudflared $CF"; ok=0; }
  for f in server.py live_ppm_pose.py feed.py prompts_ppm.py proxy.py a_level.py index.html levelfix/prior.json DEFAULTS.json CK_LABELS.json; do [ -f "$H/$f" ] || { echo "BAD missing $H/$f"; ok=0; }; done
  $PY -m py_compile $H/server.py $H/live_ppm_pose.py $H/feed.py $H/prompts_ppm.py $H/proxy.py $H/a_level.py && echo "ok  py_compile" || ok=0
  $PY -m pyflakes $H/server.py $H/live_ppm_pose.py $H/feed.py $H/proxy.py >/dev/null 2>&1 && echo "ok  pyflakes" || echo "warn pyflakes notes"
  [ $ok = 1 ]
}
if [ "${DRY:-0}" = 1 ]; then checks && echo "== DRY: checks passed" || echo "== DRY: checks FAILED"; exit 0; fi
checks || { echo "ABORT: checks failed"; exit 1; }
mkdir -p $PPMLIVE_STATE_DIR/sessions
nvidia-smi --query-gpu=index,name,memory.total,memory.used,clocks.sm,temperature.gpu --format=csv,noheader
echo "CVD=${CUDA_VISIBLE_DEVICES:-} ck $CK swap targets: $PPMLIVE_SWAP_TO ports front $FRONT back $BACK state $PPMLIVE_STATE_DIR url $URLF defaults $PPMLIVE_DEFAULTS"
( while true; do $PY -u $H/proxy.py --port $FRONT --backend $BACK --token-file $TOK --sessions $PPMLIVE_STATE_DIR/sessions >> $H/logs/proxy-$ID.log 2>&1; sleep 0.5; done ) &
( while true; do
    L=$H/logs/cloudflared-$ID-$(date +%s).log
    $CF tunnel --no-autoupdate --url http://127.0.0.1:$FRONT > $L 2>&1 &
    CP=$!
    for i in $(seq 90); do u=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' $L 2>/dev/null | head -1); [ -n "$u" ] && { echo "$u" > $URLF; echo "tunnel URL $u $(date)"; break; }; sleep 2; done
    wait $CP; sleep 5
  done ) &
$H/tools/tunnel2.sh 2592000 $FRONT >> $H/logs/tunnel2-$ID.out 2>&1 &   # 10-25: second tunnel (http2) for the page's auto-reconnect -> ALT_URLS.txt
if [ "${PRETEST:-0}" = 1 ]; then
  $PY -u $H/tools/pose_ce_test.py "$CK" $H/logs/pretest-$ID.json > $H/logs/pretest-$ID.log 2>&1; echo "pretest rc=$? $(grep -o 'PRETEST.*' $H/logs/pretest-$ID.log | tail -1) $(date)"
fi
while true; do
  $PY -u $H/server.py --ck "$CK" --port $BACK --token-file $TOK
  echo "server exited rc=$? $(date)"; sleep 10
done
