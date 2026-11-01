#!/bin/bash
# v3 of the backup driver (10-25 06:05 PT, lead's gate): one host at a time as srun --overlap steps inside the MoT job, host script v3
# (cooperative pause points). Gate = a FRESH IDLE BASELINE per host: the median of >= 12 (<= 20) MoT steps completed after our previous step
# ended (nothing of ours running), measured right before the host starts. DURING the host's step: if the median of the MoT steps since it
# started (>= 6, last 20) exceeds 1.03 x that baseline -> touch PAUSE-hfstate (the step idles at its next pause point, <= one file ~2-3 min);
# resume when the median of >= 10 steps since the pause (last 8) is back <= 1.03 x baseline; if MoT stays slow 30 min while we idle, that is
# MoT's own drift -> re-baseline on those idle steps and resume. Nothing inside the MoT job is ever killed or signalled; every step self-exits
# (timeout 6h + the host script's own exits). STOP-hfstate stops new hosts. Orchestration only (srun / squeue / grep) on the login node.
#   hf_state_overlap_driver_v3.sh [pid of a previous driver to wait for]
J=${MOT_JOB:-30289}; ML=/data/home/amartyavarma/ppm/mot/logs/ppm-mot-$J.log
T=/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/tools; S=$T/hf_state_host_v3.sbatch
LD=/data/home/amartyavarma/ppm/ppmvox/companions/logs/hfstate-1025; DL=$LD/overlap-driver.log; ST=$LD/overlap-status.tsv
PZ=$LD/PAUSE-hfstate; STOP=$LD/STOP-hfstate
A_SRC=/scratch/amartyavarma/ppmvox/ckpt/ppmvox-v5b/dcp-18163; B_SRC=/scratch/amartyavarma/ppmvox/ckpt/v5-18100-merged-keep/dcp-18100
RA=InternalCan/ppmvox-v5b-18163-state; RB=InternalCan/ppmvox-v5-18100-merged-state
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
log() { echo "[$(pt)] [v3] $*" >> $DL; }
lastn() { grep -oE '\] step [0-9]+/[0-9]+ [0-9.]+s' $ML | tail -1 | awk '{split($3,a,"/"); print a[1]+0}'; }
since() { grep -oE '\] step [0-9]+/[0-9]+ [0-9.]+s' $ML | tail -400 | awk -v s=$1 '{split($3,a,"/"); if (a[1]+0 > s) {gsub("s","",$4); print $4}}' | tail -${2:-20}; }
medn() { sort -n | awk '{a[NR]=$1} END {if (NR) printf "%s %d\n", (NR%2 ? a[(NR+1)/2] : (a[NR/2]+a[NR/2+1])/2), NR; else print "0 0"}'; }
over() { awk -v c=$1 -v b=$2 'BEGIN {exit !(c > 1.03 * b)}'; }
mot_up() { squeue -h -j $J -t R 2>/dev/null | grep -q .; }
if [ -n "${1:-}" ]; then while kill -0 $1 2>/dev/null; do sleep 30; done; log "previous driver $1 exited"; rm -f $STOP; fi
rm -f $PZ
LASTEND=$(lastn)
log "driver v3 start pid $$: MoT job $J at step $LASTEND; fresh idle baseline per host (>= 12 steps after our last step), gate 1.03 x"
ITEMS="A:slinky-3:1 A:slinky-5:2 A:slinky-11:4 A:slinky-12:5 A:slinky-16:6 A:slinky-19:7 B:slinky-0:0 B:slinky-3:1 B:slinky-5:2 B:slinky-8:3 B:slinky-11:4 B:slinky-12:5 B:slinky-16:6 B:slinky-19:7"
fails=0
for it in ${ITEMS}; do
  X=${it%%:*}; r=${it#*:}; h=${r%%:*}; i=${r#*:}
  if [ $X = A ]; then ARGS="$A_SRC 18163 $i $RA $h"; MK=$LD/ok-ppmvox-v5b-18163-state-dcp-18163-$h
  else ARGS="$B_SRC 18100 $i $RB $h RESHARD_VERIFIED.json"; MK=$LD/ok-ppmvox-v5-18100-merged-state-dcp-18100-$h; fi
  if [ -f $MK ]; then log "$X $h already ok ($(cat $MK)), skip"; continue; fi
  [ -f $STOP ] && { log "STOP file: not starting $X $h"; break; }
  t0=$(date +%s)
  if mot_up; then
    until [ "$(since $LASTEND 20 | wc -l)" -ge 12 ] || ! mot_up; do sleep 60; done
    read BH NB <<< "$(since $LASTEND 20 | medn)"
    S0=$(lastn)
    log "START $X $h (index $i) as overlap step of $J: fresh idle baseline ${BH}s (median of $NB steps after step $LASTEND), at step $S0"
    srun --overlap --jobid=$J -w $h -N1 -n1 -c 1 --mem=2G --gres=none --cpu-bind=none --kill-on-bad-exit=0 -J hfst-$X-$h \
      -o $LD/ov3-$X-$h-%j.%s.out timeout -k 120 6h bash $S $ARGS &
    SPID=$!; paused=0; npause=0
    while kill -0 $SPID 2>/dev/null; do
      sleep 60
      if [ $paused = 0 ]; then
        read c n <<< "$(since $S0 20 | medn)"
        if [ "$n" -ge 6 ] && over $c $BH; then
          touch $PZ; paused=1; npause=$((npause + 1)); PS0=$(lastn); PT0=$(date +%s)
          log "PAUSE $X $h: MoT median ${c}s over $n steps since step $S0 > 1.03 x idle baseline ${BH}s"
        fi
      else
        read c n <<< "$(since $PS0 8 | medn)"; nall=$(since $PS0 400 | wc -l)
        if [ "$nall" -ge 10 ] && ! over $c $BH; then
          rm -f $PZ; paused=0; S0=$(lastn); log "RESUME $X $h: MoT median ${c}s (last $n steps while paused) <= 1.03 x ${BH}s"
        elif [ $(( $(date +%s) - PT0 )) -ge 1800 ] && [ "$nall" -ge 10 ]; then
          read BH NB <<< "$(since $(( PS0 + 3 )) 20 | medn)"
          rm -f $PZ; paused=0; S0=$(lastn); log "RE-BASELINE + RESUME $X $h: MoT stayed slow 30 min while we idled = its own drift; new idle baseline ${BH}s ($NB steps)"
        fi
      fi
    done
    wait $SPID; rc=$?; how="overlap $J"; rm -f $PZ
    LASTEND=$(lastn)
  else
    log "START $X $h (index $i) as a standalone CPU job (MoT job $J not running)"
    sbatch --wait -w $h -x slinky-9 -J hs$X-$h $S $ARGS > /dev/null 2>&1
    rc=$?; how="standalone"; npause=0
  fi
  res=FAIL; [ -f $MK ] && res=OK
  log "END $X $h $res (rc $rc, $how, $(( $(date +%s) - t0 )) s, $npause pauses); MoT at step $(lastn), median of last 10 $(since 0 10 | medn | cut -d' ' -f1)s"
  echo -e "$(pt)\t$X\t$h\t$res\t$rc\t$how\t$(( $(date +%s) - t0 ))\t$npause" >> $ST
  if [ $res = OK ]; then fails=0; else fails=$((fails + 1)); [ $fails -ge 2 ] && { log "STOPPED: 2 failures in a row"; break; }; fi
done
log "driver v3 done"
