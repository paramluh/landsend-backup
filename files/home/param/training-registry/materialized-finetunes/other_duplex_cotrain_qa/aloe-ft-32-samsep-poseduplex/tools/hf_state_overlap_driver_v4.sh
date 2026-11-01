#!/bin/bash
# v4 of the backup driver (10-25 08:00 PT, main's go): up to 3 hosts at once as srun --overlap steps inside the MoT job (host script v4:
# per-host cooperative pause file PAUSE-hfstate-<host>). Each step: -c 1 --mem=2G --gres=none --cpu-bind=none, nice 19 + ionice idle, timeout 6h.
# Gate = MoT's median step time vs a FRESH IDLE BASELINE (median of >= 12 MoT steps while nothing of ours is active):
#   - measured at start, and refreshed every 60 min (all steps paused at their next pause point, 5 min drain, then >= 12 idle steps).
#   - every minute: median of the MoT steps since the last change (>= 6, last 20) > 1.03 x baseline -> pause the newest active step and
#     lower the cap (3 -> 2 -> 1 -> 0); <= 1.03 x with >= 10 steps since the last change -> resume a paused step or start the next part, up to cap.
#     After 30 min at a lowered cap with the gate clear, the cap goes back up by 1. All paused + MoT still slow after 30 min = MoT's own drift ->
#     re-baseline on those idle steps.
# Parts: A (v5b dcp-18163) left first, then B (merged v5 dcp-18100); never 2 steps on one host. Nothing inside the MoT job is ever killed or
# signalled; steps self-exit. STOP-hfstate: no new starts, every step exits at its next pause point. Orchestration only, on the login node.
#   hf_state_overlap_driver_v4.sh [pid of the previous driver] [its last running host label: take over after its END line]
J=${MOT_JOB:-30289}; ML=/data/home/amartyavarma/ppm/mot/logs/ppm-mot-$J.log
T=/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/tools; S=$T/hf_state_host_v4.sbatch
LD=/data/home/amartyavarma/ppm/ppmvox/companions/logs/hfstate-1025; DL=$LD/overlap-driver.log; ST=$LD/overlap-status.tsv; STOP=$LD/STOP-hfstate
A_SRC=/scratch/amartyavarma/ppmvox/ckpt/ppmvox-v5b/dcp-18163; B_SRC=/scratch/amartyavarma/ppmvox/ckpt/v5-18100-merged-keep/dcp-18100
RA=InternalCan/ppmvox-v5b-18163-state; RB=InternalCan/ppmvox-v5-18100-merged-state
CAPMAX=${HFS_CAP:-3}
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
log() { echo "[$(pt)] [v4] $*" >> $DL; }
lastn() { grep -oE '\] step [0-9]+/[0-9]+ [0-9.]+s' $ML | tail -1 | awk '{split($3,a,"/"); print a[1]+0}'; }
since() { grep -oE '\] step [0-9]+/[0-9]+ [0-9.]+s' $ML | tail -400 | awk -v s=$1 '{split($3,a,"/"); if (a[1]+0 > s) {gsub("s","",$4); print $4}}' | tail -${2:-20}; }
medn() { sort -n | awk '{a[NR]=$1} END {if (NR) printf "%s %d\n", (NR%2 ? a[(NR+1)/2] : (a[NR/2]+a[NR/2+1])/2), NR; else print "0 0"}'; }
over() { awk -v c=$1 -v b=$2 'BEGIN {exit !(c > 1.03 * b)}'; }
mot_up() { squeue -h -j $J -t R 2>/dev/null | grep -q .; }
mk() { [ $1 = A ] && echo $LD/ok-ppmvox-v5b-18163-state-dcp-18163-$2 || echo $LD/ok-ppmvox-v5-18100-merged-state-dcp-18100-$2; }
idx() { case $1 in slinky-0) echo 0;; slinky-3) echo 1;; slinky-5) echo 2;; slinky-8) echo 3;; slinky-11) echo 4;; slinky-12) echo 5;; slinky-16) echo 6;; slinky-19) echo 7;; esac; }
args() { [ $1 = A ] && echo "$A_SRC 18163 $(idx $2) $RA $2" || echo "$B_SRC 18100 $(idx $2) $RB $2 RESHARD_VERIFIED.json"; }
# ---- take over from the previous driver after its current host ends
if [ -n "${1:-}" ]; then
  if [ -n "${2:-}" ]; then until grep -qE "\[v3\] END [AB] $2 " $DL || ! kill -0 $1 2>/dev/null; do sleep 5; done; fi
  echo "v4 handover $(pt)" > $STOP
  while kill -0 $1 2>/dev/null; do sleep 10; done
  log "previous driver $1 exited"; rm -f $STOP
fi
rm -f $LD/PAUSE-hfstate $LD/PAUSE-hfstate-*
declare -A SPID PAUSED XOF T0H OUTF
QUEUE=""
for X in A B; do for h in slinky-0 slinky-3 slinky-5 slinky-8 slinky-11 slinky-12 slinky-16 slinky-19; do
  [ $X = A ] && [ $h = slinky-8 ] && continue                      # s8's A part was done from /data (job 30297)
  [ -f $(mk $X $h) ] || QUEUE="$QUEUE $X:$h"; done; done
log "driver v4 start pid $$: queue [$QUEUE ], cap $CAPMAX, MoT at step $(lastn)"
nrun() { echo ${#SPID[@]}; }
nact() { local n=0 h; for h in "${!SPID[@]}"; do [ "${PAUSED[$h]}" = 0 ] && n=$((n+1)); done; echo $n; }
measure_idle() {                                                    # all steps paused already: drain 5 min, then >= 12 idle steps
  local s0; sleep ${1:-300}; s0=$(lastn)
  until [ "$(since $s0 20 | wc -l)" -ge 12 ] || ! mot_up; do sleep 30; done
  read BASE NB <<< "$(since $s0 20 | medn)"; LAST_BASE=$(date +%s)
}
start_part() {                                                      # $1 = X:host
  local X=${1%%:*} h=${1#*:} of
  of=$LD/ov4-$X-$h-$(date +%s).out
  srun --overlap --jobid=$J -w $h -N1 -n1 -c 1 --mem=2G --gres=none --cpu-bind=none --kill-on-bad-exit=0 -J hfst-$X-$h \
    -o $of timeout -k 120 6h bash $S $(args $X $h) > /dev/null 2>&1 &
  SPID[$h]=$!; PAUSED[$h]=0; XOF[$h]=$X; T0H[$h]=$(date +%s); OUTF[$h]=$of
  log "START $X $h (step output $(basename $of)); active $(nact)/$(nrun), cap $CAP, baseline ${BASE}s, MoT at step $(lastn)"
}
# ---- initial idle baseline (nothing of ours runs now)
if mot_up; then measure_idle 0; log "idle baseline ${BASE}s (median of $NB steps)"; fi
CAP=$CAPMAX; SCH=$(lastn); TTRIP=0; fails=0; LAST_BASE=${LAST_BASE:-$(date +%s)}
while true; do
  # reap finished steps
  for h in "${!SPID[@]}"; do
    if ! kill -0 ${SPID[$h]} 2>/dev/null; then
      wait ${SPID[$h]}; rc=$?; X=${XOF[$h]}; res=FAIL; [ -f $(mk $X $h) ] && res=OK
      rm -f $LD/PAUSE-hfstate-$h
      log "END $X $h $res (rc $rc, $(( $(date +%s) - ${T0H[$h]} )) s)"
      echo -e "$(pt)\t$X\t$h\t$res\t$rc\tv4 overlap $J\t$(( $(date +%s) - ${T0H[$h]} ))" >> $ST
      unset SPID[$h] PAUSED[$h] XOF[$h] T0H[$h] OUTF[$h]
      if [ $res = FAIL ]; then fails=$((fails + 1)); QUEUE="$QUEUE $X:$h"; [ $fails -ge 3 ] && { log "STOPPED: $fails failures"; touch $STOP; }; fi
      SCH=$(lastn)
    fi
  done
  [ -z "${QUEUE// /}" ] && [ $(nrun) = 0 ] && break
  if [ -f $STOP ]; then [ $(nrun) = 0 ] && break; sleep 30; continue; fi
  if ! mot_up; then                                                 # MoT gone: hosts free -> remaining parts as standalone CPU jobs
    if [ $(nrun) = 0 ]; then
      log "MoT job $J not running: remaining parts [$QUEUE ] as standalone CPU jobs (A before B per host)"
      for it in $QUEUE; do X=${it%%:*}; h=${it#*:}
        dep=""; [ $X = B ] && pj=$(squeue -h -u $USER -n hsA-$h -o %i | head -1) && [ -n "$pj" ] && dep="-d afterany:$pj"
        sbatch --parsable -w $h -x slinky-9 $dep -J hs$X-$h $S $(args $X $h) >> $DL 2>&1; done
      QUEUE=""; break
    fi
    sleep 30; continue
  fi
  # hourly fresh idle baseline
  if [ $(nrun) -gt 0 ] && [ $(( $(date +%s) - LAST_BASE )) -ge 3600 ]; then
    for h in "${!SPID[@]}"; do touch $LD/PAUSE-hfstate-$h; done
    log "REFRESH baseline: all $(nrun) steps pausing at their next pause point"
    measure_idle 300
    for h in "${!SPID[@]}"; do [ "${PAUSED[$h]}" = 0 ] && rm -f $LD/PAUSE-hfstate-$h; done
    log "REFRESH done: idle baseline ${BASE}s ($NB steps); resumed $(nact) step(s)"; SCH=$(lastn); continue
  fi
  read c n <<< "$(since $SCH 20 | medn)"
  if [ "$(nact)" -gt 0 ] && [ "$n" -ge 6 ] && over $c $BASE; then           # gate tripped: pause the newest active step, lower the cap
    nh=""; nt=0; for h in "${!SPID[@]}"; do [ "${PAUSED[$h]}" = 0 ] && [ ${T0H[$h]} -ge $nt ] && { nt=${T0H[$h]}; nh=$h; }; done
    touch $LD/PAUSE-hfstate-$nh; PAUSED[$nh]=1; CAP=$(nact); TTRIP=$(date +%s); SCH=$(lastn)
    log "GATE TRIP: MoT median ${c}s ($n steps) > 1.03 x ${BASE}s -> paused $nh (${XOF[$nh]}); cap now $CAP"
  elif [ "$n" -ge 10 ] && ! over $c $BASE; then
    if [ $CAP -lt $CAPMAX ] && [ $TTRIP -gt 0 ] && [ $(( $(date +%s) - TTRIP )) -ge 1800 ]; then CAP=$((CAP + 1)); TTRIP=$(date +%s); log "cap back up to $CAP (gate clear 30 min)"; fi
    if [ "$(nact)" -lt $CAP ]; then
      ph=""; for h in "${!SPID[@]}"; do [ "${PAUSED[$h]}" = 1 ] && { ph=$h; break; }; done
      if [ -n "$ph" ]; then rm -f $LD/PAUSE-hfstate-$ph; PAUSED[$ph]=0; SCH=$(lastn); log "RESUME $ph (${XOF[$ph]}): MoT median ${c}s <= 1.03 x ${BASE}s; active $(nact)/$(nrun), cap $CAP"
      else
        for it in $QUEUE; do h=${it#*:}; [ -n "${SPID[$h]:-}" ] && continue
          QUEUE=$(echo " $QUEUE " | sed "s/ $it / /"); start_part $it; SCH=$(lastn); break; done
      fi
    fi
  elif [ "$(nact)" = 0 ] && [ "$(nrun)" -gt 0 ] && [ $TTRIP -gt 0 ] && [ $(( $(date +%s) - TTRIP )) -ge 1800 ] && [ "$n" -ge 10 ]; then
    BASE=$c; LAST_BASE=$(date +%s); CAP=1; TTRIP=$(date +%s); SCH=$(lastn)
    log "RE-BASELINE: all paused 30 min and MoT still at ${c}s = its own drift; new idle baseline ${BASE}s, cap 1"
  fi
  sleep 60
done
log "driver v4 done; queue left [$QUEUE ]"
