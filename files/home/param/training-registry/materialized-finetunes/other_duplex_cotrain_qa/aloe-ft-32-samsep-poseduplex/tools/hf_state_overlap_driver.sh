#!/bin/bash
# 10-25 (lead + main order): back up PPMVOX full states to private HF ONE HOST AT A TIME as srun --overlap steps inside the running MoT job
# (no free CPUs outside it). Each step: -c 1 --mem=2G --gres=none --cpu-bind=none, nice 19 + ionice idle (inside hf_state_host.sbatch),
# 2 upload workers, streaming low-RSS digest check, a hard `timeout` so it self-exits. NOTHING inside the MoT job is ever killed / signalled.
# Back-off: before each host, MoT's median step time over its last 20 steps must be <= 1.03 x the pre-upload baseline; else wait (re-check 5 min).
# If the MoT job is gone, the remaining items run as standalone CPU jobs (sbatch --wait, -c 2 --mem=8G, same script) one at a time.
# Order: A = v5b dcp-18163 on the 7 node-local hosts (s8's part is done separately from /data), then B = merged v5 dcp-18100 on all 8 hosts.
# Orchestration only (srun / squeue / grep): runs detached on the login node; no compute here.
J=${MOT_JOB:-30289}; ML=/data/home/amartyavarma/ppm/mot/logs/ppm-mot-$J.log
T=/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/tools; S=$T/hf_state_host.sbatch
LD=/data/home/amartyavarma/ppm/ppmvox/companions/logs/hfstate-1025; DL=$LD/overlap-driver.log; ST=$LD/overlap-status.tsv
A_SRC=/scratch/amartyavarma/ppmvox/ckpt/ppmvox-v5b/dcp-18163; B_SRC=/scratch/amartyavarma/ppmvox/ckpt/v5-18100-merged-keep/dcp-18100
RA=InternalCan/ppmvox-v5b-18163-state; RB=InternalCan/ppmvox-v5-18100-merged-state
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
log() { echo "[$(pt)] $*" >> $DL; }
med() { grep -oE '\] step [0-9]+/[0-9]+ [0-9.]+s' $ML | tail -20 | awk '{gsub("s","",$4); print $4}' | sort -n | awk '{a[NR]=$1} END {if (NR) print (NR%2 ? a[(NR+1)/2] : (a[NR/2]+a[NR/2+1])/2); else print 0}'; }
lastn() { grep -oE '\] step [0-9]+/[0-9]+ [0-9.]+s' $ML | tail -1 | awk '{print $3}'; }
mot_up() { squeue -h -j $J -t R 2>/dev/null | grep -q .; }
BASE=${MOT_BASE:-$(med)}
log "driver start pid $$: MoT job $J baseline median step ${BASE}s (last 20 steps, at step $(lastn)); gate 1.03 x"
ITEMS="A:slinky-0:0 A:slinky-3:1 A:slinky-5:2 A:slinky-11:4 A:slinky-12:5 A:slinky-16:6 A:slinky-19:7 B:slinky-0:0 B:slinky-3:1 B:slinky-5:2 B:slinky-8:3 B:slinky-11:4 B:slinky-12:5 B:slinky-16:6 B:slinky-19:7"
for it in ${ITEMS}; do
  X=${it%%:*}; r=${it#*:}; h=${r%%:*}; i=${r#*:}
  if [ -f $LD/done-$X-$h ]; then log "$X $h already done, skip"; continue; fi
  if [ $X = A ]; then ARGS="$A_SRC 18163 $i $RA $h"; else ARGS="$B_SRC 18100 $i $RB $h RESHARD_VERIFIED.json"; fi
  while mot_up; do                                                  # back-off gate (only while we would share the MoT job)
    c=$(med); ok=$(awk -v c=$c -v b=$BASE 'BEGIN {print (c <= 1.03 * b) ? 1 : 0}')
    [ "$ok" = 1 ] && break
    log "PAUSE before $X $h: MoT median step ${c}s > 1.03 x baseline ${BASE}s (at step $(lastn)); re-check in 5 min"
    sleep 300
  done
  t0=$(date +%s)
  if mot_up; then
    log "START $X $h (index $i) as overlap step of $J: MoT median $(med)s (baseline ${BASE}s, step $(lastn))"
    srun --overlap --jobid=$J -w $h -N1 -n1 -c 1 --mem=2G --gres=none --cpu-bind=none --kill-on-bad-exit=0 -J hfst-$X-$h \
      -o $LD/ov-$X-$h-%j.%s.out timeout -k 120 5h bash $S $ARGS
    rc=$?; how="overlap $J"
  else
    log "START $X $h (index $i) as a standalone CPU job (MoT job $J not running)"
    sbatch --wait -w $h -x slinky-9 -J hs$X-$h $S $ARGS > /dev/null 2>&1
    rc=$?; how="standalone"
  fi
  log "END $X $h rc $rc ($how, $(( $(date +%s) - t0 )) s); MoT median now $(med)s (baseline ${BASE}s, step $(lastn))"
  echo -e "$(pt)\t$X\t$h\t$rc\t$how\t$(( $(date +%s) - t0 ))" >> $ST
  [ $rc = 0 ] && touch $LD/done-$X-$h
done
log "driver done"
