#!/bin/bash
# 10-26 (main: zero-downtime genweb): runs the always-up front (front.py on :9205, own restart loop) and supervises the APP server
# (server.py) on 9206 / 9207. DEPLOY = touch genweb/DEPLOY: start the new app on the other port, wait until it answers /api/schema 200
# (max 120 s), rewrite APP_PORT (the front switches on its next request), drain the old app 30 s, stop it. A crashed current app is
# restarted on its own port. Never restart the front for a deploy. Log: genweb/logs/app-<job>.log
#   sbatch -w slinky-18 -x slinky-9 -c 4 --mem=8G -J genweb-app genweb/app_sup.sh   (wrapped by app.sbatch)
G=/data/home/amartyavarma/ppm/aloe-posedup-live/genweb; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
LOG=$G/logs/app-${SLURM_JOB_ID:-x}.log
log(){ echo "[app_sup $(date -u +%T)] $*" >> $LOG; }
( while true; do $PY -u $G/front.py --port 9205 >> $G/logs/front-${SLURM_JOB_ID:-x}.log 2>&1; echo "[front] exited rc=$? $(date)" >> $LOG; sleep 1; done ) &
start_app(){ $PY -u $G/server.py --port $1 >> $G/logs/web-app$1.log 2>&1 & echo $!; }
healthy(){ [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:$1/api/schema)" = 200 ]; }
cur=$(cat $G/APP_PORT 2>/dev/null || echo 9206); case $cur in 9206|9207) ;; *) cur=9206;; esac
pid=$(start_app $cur); log "app on :$cur pid $pid"
for i in $(seq 120); do healthy $cur && break; sleep 1; done
healthy $cur && { echo $cur > $G/APP_PORT.tmp; mv -f $G/APP_PORT.tmp $G/APP_PORT; log "app :$cur healthy; APP_PORT=$cur"; }
last=$(stat -c %Y $G/DEPLOY 2>/dev/null || echo 0)
while true; do
  sleep 2
  if ! kill -0 $pid 2>/dev/null; then log "app :$cur died; restarting"; pid=$(start_app $cur); for i in $(seq 120); do healthy $cur && break; sleep 1; done; continue; fi
  d=$(stat -c %Y $G/DEPLOY 2>/dev/null || echo 0)
  [ "$d" -gt "$last" ] || continue
  last=$d; new=$([ $cur = 9206 ] && echo 9207 || echo 9206)
  log "DEPLOY: new app on :$new"; npid=$(start_app $new); ok=0
  for i in $(seq 120); do healthy $new && { ok=1; break; }; kill -0 $npid 2>/dev/null || break; sleep 1; done
  if [ $ok = 1 ]; then
    echo $new > $G/APP_PORT.tmp; mv -f $G/APP_PORT.tmp $G/APP_PORT; log "switched APP_PORT=$new; draining :$cur (pid $pid) 30 s"
    sleep 30; kill $pid 2>/dev/null; wait $pid 2>/dev/null; log "old app :$cur stopped"; cur=$new; pid=$npid
  else
    log "DEPLOY FAILED: new app :$new never healthy (see logs/web-app$new.log); keeping :$cur"; kill $npid 2>/dev/null
  fi
done
