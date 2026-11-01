#!/bin/bash
# v2 of hf_state_overlap_driver.sh (10-25 05:50 PT): the same one-host-at-a-time overlap steps inside the MoT job, with hf_state_host_v2.sbatch
# (2 tokio / rayon threads, 1 upload worker, xet fixed upload concurrency 2: anon RSS ~0.45 GB under the 2G step cap; v1 OOMed at 64 threads).
# Success = the host script's marker ok-<repo>-dcp-<step>-<host> (srun's rc is != 0 whenever the step saw a contained OOM event).
# Gate: before each host, MoT's median step over its last 20 steps <= 1.03 x the PRE-UPLOAD baseline (20.95 s at step 5839), else wait 5 min.
# Stops starting new steps on: $LD/STOP-hfstate, or 2 failures in a row. Never kills / signals anything inside the MoT job; each step self-exits
# (timeout 5h + the script's own early exits). Orchestration only (srun / squeue / grep) on the login node.
J=${MOT_JOB:-30289}; ML=/data/home/amartyavarma/ppm/mot/logs/ppm-mot-$J.log
T=/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/tools; S=$T/hf_state_host_v2.sbatch
LD=/data/home/amartyavarma/ppm/ppmvox/companions/logs/hfstate-1025; DL=$LD/overlap-driver.log; ST=$LD/overlap-status.tsv
A_SRC=/scratch/amartyavarma/ppmvox/ckpt/ppmvox-v5b/dcp-18163; B_SRC=/scratch/amartyavarma/ppmvox/ckpt/v5-18100-merged-keep/dcp-18100
RA=InternalCan/ppmvox-v5b-18163-state; RB=InternalCan/ppmvox-v5-18100-merged-state
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
log() { echo "[$(pt)] [v2] $*" >> $DL; }
med() { grep -oE '\] step [0-9]+/[0-9]+ [0-9.]+s' $ML | tail -20 | awk '{gsub("s","",$4); print $4}' | sort -n | awk '{a[NR]=$1} END {if (NR) print (NR%2 ? a[(NR+1)/2] : (a[NR/2]+a[NR/2+1])/2); else print 0}'; }
lastn() { grep -oE '\] step [0-9]+/[0-9]+ [0-9.]+s' $ML | tail -1 | awk '{print $3}'; }
mot_up() { squeue -h -j $J -t R 2>/dev/null | grep -q .; }
BASE=${MOT_BASE:-20.95}
log "driver v2 start pid $$: MoT job $J pre-upload baseline ${BASE}s; now median $(med)s at step $(lastn); gate 1.03 x"
ITEMS="A:slinky-3:1 A:slinky-5:2 A:slinky-11:4 A:slinky-12:5 A:slinky-16:6 A:slinky-19:7 B:slinky-0:0 B:slinky-3:1 B:slinky-5:2 B:slinky-8:3 B:slinky-11:4 B:slinky-12:5 B:slinky-16:6 B:slinky-19:7"
fails=0
for it in ${ITEMS}; do
  X=${it%%:*}; r=${it#*:}; h=${r%%:*}; i=${r#*:}
  if [ $X = A ]; then ARGS="$A_SRC 18163 $i $RA $h"; MK=$LD/ok-ppmvox-v5b-18163-state-dcp-18163-$h
  else ARGS="$B_SRC 18100 $i $RB $h RESHARD_VERIFIED.json"; MK=$LD/ok-ppmvox-v5-18100-merged-state-dcp-18100-$h; fi
  if [ -f $MK ]; then log "$X $h already ok ($(cat $MK)), skip"; continue; fi
  [ -f $LD/STOP-hfstate ] && { log "STOP file: not starting $X $h"; break; }
  while mot_up; do                                                  # back-off gate (only while we would share the MoT job)
    c=$(med); ok=$(awk -v c=$c -v b=$BASE 'BEGIN {print (c <= 1.03 * b) ? 1 : 0}')
    [ "$ok" = 1 ] && break
    log "PAUSE before $X $h: MoT median step ${c}s > 1.03 x baseline ${BASE}s (at step $(lastn)); re-check in 5 min"
    sleep 300
    [ -f $LD/STOP-hfstate ] && break
  done
  [ -f $LD/STOP-hfstate ] && { log "STOP file: not starting $X $h"; break; }
  t0=$(date +%s)
  if mot_up; then
    log "START $X $h (index $i) as overlap step of $J: MoT median $(med)s (baseline ${BASE}s, step $(lastn))"
    srun --overlap --jobid=$J -w $h -N1 -n1 -c 1 --mem=2G --gres=none --cpu-bind=none --kill-on-bad-exit=0 -J hfst-$X-$h \
      -o $LD/ov2-$X-$h-%j.%s.out timeout -k 120 5h bash $S $ARGS
    rc=$?; how="overlap $J"
  else
    log "START $X $h (index $i) as a standalone CPU job (MoT job $J not running)"
    sbatch --wait -w $h -x slinky-9 -J hs$X-$h $S $ARGS > /dev/null 2>&1
    rc=$?; how="standalone"
  fi
  res=FAIL; [ -f $MK ] && res=OK
  log "END $X $h $res (srun/sbatch rc $rc, $how, $(( $(date +%s) - t0 )) s); MoT median now $(med)s (baseline ${BASE}s, step $(lastn))"
  echo -e "$(pt)\t$X\t$h\t$res\t$rc\t$how\t$(( $(date +%s) - t0 ))" >> $ST
  if [ $res = OK ]; then fails=0; else fails=$((fails + 1)); [ $fails -ge 2 ] && { log "STOPPED: 2 failures in a row"; break; }; fi
done
log "driver v2 done"
