#!/bin/bash
# BOTH scaling FTs CHAIN (10-22; user 12:22 PT via coordinator: "kill it" + "the 2 fine-tunes on our 2 checkpoints run straight after, on 8 nodes
# [each]"). Runs in ftb_wait.sbatch (CPU-only, slinky-15) once the 8,1,1 delivery-2 append train job (FTB_TRAIN_JOB, first 27204) has ended
# (afterany). Log: logs/ftb_chain.log; markers in control/. Both FTs = aloe-v2pure-16610 (job 26089) verbatim except the init: world 64 on 8 PPM
# hosts each (8 ranks per host = 26089's layout), separate run dirs / ports (FT-(a) aloe-v2pure-17352 init local-17352, port 29864; FT-(b)
# aloe-v2pure-17641 init local-17641, port 29865).
#  1. the train job must have logged "done <M> steps" (M = LAUNCHED.txt field 4 = 17641); else follow a watchdog relaunch (<= 6 hops) or FTB_BLOCKED.
#  2. per-host /scratch free: ftb_convert.sbatch (16 hosts; protect dcp-M as ckpt-keep-ft hard links + convert it to /data ckpt-infer/ppm-v1/local-M)
#     prints SCRATCHFREE0 at its start; if /data < 386 GB it is not started yet (DISK_LOW wait) and ftb_dfprobe.sbatch gives the free space instead.
#  3. SPLIT: the 16 hosts by free /scratch, descending, alternately to FT-(a) / FT-(b). Per group: final resumable save ON if every host has
#     >= 144 GB (+7 on its rank 0 host) = final save 79 + final export 11 + 2 intermediate exports in flight 22 (tools/xfree_run.sh moves each
#     intermediate to private HF during the run) + caches 2 + 30 left; else OFF (operational, ALOE_FINAL_SAVE=0) if every host has >= 65 (+7);
#     else that FT goes to the SPARE nodes (slinky-4 GPUs 0-5 + slinky-15 + slinky-18 GPUs 2-7, world 20 het job, as the killed FT-(a)).
#  4. FT-(a) launches at once; FT-(b) once local-M is COMPLETE. Each: + xfree (after:<FT>), final-export HF backup (afterok) and the resumable
#     save cleanup after HF (afterok). control/FT_PLAN.txt = the split; control/FTB_LAUNCHED = "<FT-(a) job> <FT-(b) job>" when both launched.
set -u
Q=/data/home/amartyavarma/ppm/aloe-v2pure-17641; T=$Q/tools; CTL=$Q/control; LOG=$Q/logs/ftb_chain.log
A=/data/home/amartyavarma/ppm/aloe-v2pure-17352
C811=/data/home/amartyavarma/ppm/code-811; NODES16=$(cat $C811/nodes16_s7.txt)
J=${FTB_TRAIN_JOB:-27204}; HOP=${FTB_HOP:-0}
M=$(awk '{print $4}' $C811/fresh/append/LAUNCHED.txt); [[ "$M" =~ ^[0-9]+$ ]] || M=17641
mkdir -p $CTL
say() { echo "[$(TZ=America/Los_Angeles date '+%m-%d %T PT')] $*" | tee -a $LOG; }
clean_sb() { env $(env | grep -oE '^(SLURM|SBATCH|SRUN|SALLOC)_[A-Za-z0-9_]*=' | tr -d = | sed 's/^/-u /') sbatch "$@"; }
dfree() { df -BG --output=avail /data | tail -1 | tr -dc 0-9; }
say "chain start: train job $J (hop $HOP), target M=$M"
# ---- 1. outcome
L=/data/home/amartyavarma/ppm/logs/ppm-$J.log
if ! grep -aq "done $M steps" $L 2>/dev/null; then
  say "train job $J ended WITHOUT 'done $M steps' (last: $(grep -a 'step [0-9]*/' $L 2>/dev/null | tail -1 | cut -c1-60))"
  for i in $(seq 1 90); do
    N=$(cat /data/home/amartyavarma/ppm/control/ppm-v1/CURRENT_JOB_811 2>/dev/null)
    [ -n "$N" ] && [ "$N" != "$J" ] && squeue -h -j "$N" -o %T 2>/dev/null | grep -qE "RUNNING|PENDING|CONFIGURING" && break
    N=$(squeue -h -u amartyavarma -n ppm811 -o %i | sort -n | tail -1)
    [ -n "$N" ] && [ "$N" -gt "$J" ] && break
    N=""; sleep 30
  done
  if [ -n "$N" ] && [ "$HOP" -lt 6 ]; then
    W=$(FTB_TRAIN_JOB=$N FTB_HOP=$((HOP + 1)) clean_sb --parsable --dependency=afterany:$N $T/ftb_wait.sbatch)
    say "following the relaunched train job $N: waiter $W (afterany:$N)"; echo "$W" > $CTL/FTB_WAITER; exit 0
  fi
  say "BLOCKED: no relaunch of the append run found; FTs NOT launched"; echo "$(date) train job $J ended without done $M" > $CTL/FTB_BLOCKED; exit 1
fi
say "train job $J: done $M steps"
# ---- 2. per-host free (+ start the init conversion when /data has room)
OUT=/data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-$M; CJ=""
if [ -f $OUT/COMPLETE ]; then say "init $OUT already COMPLETE"; fi
if [ ! -f $OUT/COMPLETE ] && [ "$(dfree)" -ge 386 ]; then
  CJ=$(FTB_STEP=$M clean_sb --parsable -w $NODES16 $T/ftb_convert.sbatch); PROBE=$Q/logs/ftb_convert-$CJ.log; TAG=SCRATCHFREE0
  say "convert job $CJ (dcp-$M -> $OUT; /data $(dfree) GB free)"
else
  [ -f $OUT/COMPLETE ] || say "DISK_LOW: /data $(dfree) GB < 386: conversion waits; free space from a probe"
  PJ=$(clean_sb --parsable -w $NODES16 $T/ftb_dfprobe.sbatch); PROBE=$Q/logs/ftb_dfprobe-$PJ.log; TAG=SCRATCHFREE
fi
for i in $(seq 1 180); do [ "$(grep -hc " $TAG " $PROBE 2>/dev/null)" -ge 16 ] && break; sleep 10; done
[ "$(grep -hc " $TAG " $PROBE 2>/dev/null)" -ge 16 ] || { say "BLOCKED: no per-host free space after 30 min ($PROBE)"; echo "$(date) no probe" > $CTL/FTB_BLOCKED; exit 1; }
FREE=$(grep -h " $TAG " $PROBE | awk '{print $1, $3}' | grep -v "^slinky-10 " | sort -k2 -n -r)
say "per-host /scratch free: $(echo "$FREE" | sort -V | awk '{print $1"="$2}' | paste -sd' ')"
# ---- 3. split
GA=$(echo "$FREE" | awk 'NR % 2 == 1 {print $1}' | sort -V | paste -sd,); GB=$(echo "$FREE" | awk 'NR % 2 == 0 {print $1}' | sort -V | paste -sd,)
fs_of() {   # $1 = comma host list -> 1 / 0 / spares
  local r0 h f ok1=1 ok0=1; r0=$(echo $1 | cut -d, -f1)
  for h in $(echo $1 | tr , ' '); do f=$(echo "$FREE" | awk -v h=$h '$1==h {print $2}'); x=0; [ $h = $r0 ] && x=7
    [ "${f:-0}" -ge $((144 + x)) ] || ok1=0; [ "${f:-0}" -ge $((65 + x)) ] || ok0=0; done
  [ $ok1 = 1 ] && echo 1 || { [ $ok0 = 1 ] && echo 0 || echo spares; }
}
[ "$(echo $GA | tr , '\n' | wc -l)" = 8 ] && [ "$(echo $GB | tr , '\n' | wc -l)" = 8 ] || { say "BLOCKED: not 16 PPM hosts in the probe"; echo "$(date) hosts" > $CTL/FTB_BLOCKED; exit 1; }
FA=$(fs_of $GA); FB=$(fs_of $GB)
if [ $FA = spares ] && [ $FB = spares ]; then   # both short: FT-(a) on the 8 most-free hosts if they fit, FT-(b) on the spares
  GA=$(echo "$FREE" | head -8 | awk '{print $1}' | sort -V | paste -sd,); FA=$(fs_of $GA); FB=spares
fi
echo "FT-(a) 17352: $([ $FA = spares ] && echo "spares slinky-4,slinky-15,slinky-18 (world 20)" || echo "$GA (world 64, final save $FA)")
FT-(b) $M: $([ $FB = spares ] && echo "spares slinky-4,slinky-15,slinky-18 (world 20)" || echo "$GB (world 64, final save $FB)")
free: $(echo "$FREE" | sort -V | awk '{print $1"="$2}' | paste -sd' ')" > $CTL/FT_PLAN.txt
say "SPLIT: $(tr '\n' ' ' < $CTL/FT_PLAN.txt)"
[ $FA = spares ] && [ $FB = spares ] && { say "BLOCKED: neither FT fits 8 PPM hosts even without the final save"; echo "$(date) no fit" > $CTL/FTB_BLOCKED; exit 1; }
launch() {   # $1 run dir, $2 hosts|spares, $3 final save -> FT job id; chains xfree + HF backup + save cleanup
  local R=$1 H=$2 FS=$3 FJ XJ HJ RJ NH NT HS
  if [ "$H" = spares ]; then
    HS=slinky-4,slinky-15,slinky-18; NT=6,8,6; NH=3
    if [ $R = aloe-v2pure-17352 ]; then FJ=$(clean_sb --parsable $A/run_het.sbatch); else FJ=$(clean_sb --parsable $Q/run_spares.sbatch); fi
  else
    HS=$H; NT=8,8,8,8,8,8,8,8; NH=8
    FJ=$(FTB_FINAL_SAVE=$FS clean_sb --parsable -w $HS /data/home/amartyavarma/ppm/$R/run.sbatch)
  fi
  [ -n "$FJ" ] || return 1
  XJ=$(XF_RUN=$R XF_REPO=InternalCan/ppm-$R XF_STEPS="33 67 100 133 167" XF_FINAL=200 XF_HOSTS=$HS XF_NTASKS=$NT \
       XF_TLOG=/data/home/amartyavarma/ppm/$R/logs/aloev2p${R#aloe-v2pure-}-$FJ.log XF_TJOB=$FJ clean_sb --parsable --dependency=after:$FJ -N$NH -w $HS $T/xfree_run.sbatch)
  HJ=$(clean_sb --parsable --dependency=afterok:$FJ -N$NH -w $HS /data/home/amartyavarma/ppm/$R/hf_final_export.sbatch)
  RJ=$(HFJ=$HJ clean_sb --parsable --dependency=afterok:$HJ -N$NH -w $HS /data/home/amartyavarma/ppm/$R/rm_saves_after_hf.sbatch)
  printf '{"train_job": "%s", "hosts": "%s", "final_save": "%s", "xfree_job": "%s", "backup_job": "%s", "rm_saves_job": "%s", "repo_final": "InternalCan/ppm-%s-final", "repo_intermediates": "InternalCan/ppm-%s", "time": "%s"}\n' \
    "$FJ" "$HS" "$FS" "$XJ" "$HJ" "$RJ" "$R" "$R" "$(TZ=America/Los_Angeles date '+%F %T PT')" > /data/home/amartyavarma/ppm/$R/BACKUP_JOB
  say "LAUNCHED $R: job $FJ on $HS (final save $FS); xfree $XJ, HF backup $HJ, save cleanup $RJ"
  echo $FJ
}
# ---- 4a. FT-(a) now (its init local-17352 is on /data); a spares FT-(b) would have to wait for it, so FT-(a) never takes the spares if FT-(b) does
FJA=$(launch aloe-v2pure-17352 $([ $FA = spares ] && echo spares || echo $GA) $([ $FA = spares ] && echo 1 || echo $FA) | tail -1)
[ -n "$FJA" ] || { say "BLOCKED: FT-(a) sbatch failed"; echo "$(date) sbatch A" > $CTL/FTB_BLOCKED; exit 1; }
# ---- 4b. FT-(b) after its init
if [ ! -f $OUT/COMPLETE ]; then
  if [ -z "$CJ" ]; then   # /data was short: wait (5 min polls, <= 12 h), then convert
    for i in $(seq 1 144); do [ "$(dfree)" -ge 386 ] && break; touch $CTL/FTB_DISK_LOW; sleep 300; done
    [ "$(dfree)" -ge 386 ] || { say "BLOCKED: /data $(dfree) GB < 386 after 12 h: FT-(b) init not converted"; echo "$(date) /data low" > $CTL/FTB_BLOCKED; exit 1; }
    rm -f $CTL/FTB_DISK_LOW
    CJ=$(FTB_STEP=$M clean_sb --parsable -w $NODES16 $T/ftb_convert.sbatch); say "convert job $CJ (/data $(dfree) GB free)"
  fi
  while squeue -h -j $CJ 2>/dev/null | grep -q .; do sleep 15; done
  [ "$(cat $OUT/COMPLETE 2>/dev/null)" = "$M" ] || { say "BLOCKED: convert job $CJ did not finish local-$M (logs/ftb_convert-$CJ.log)"; echo "$(date) convert failed $CJ" > $CTL/FTB_BLOCKED; exit 1; }
fi
say "init $OUT COMPLETE ($(du -sBG $OUT | cut -f1)); /data $(dfree) GB free"
if [ $FB = spares ]; then   # the spares must be free (FT-(a) not there) and have the room
  for i in $(seq 1 144); do
    if ! squeue -h -u amartyavarma -o "%j %N" | grep -E "^aloev2p17352 " | grep -qE "slinky-(4|15|18)"; then
      P3=$(clean_sb --parsable --wait -N3 -w slinky-4,slinky-15,slinky-18 $T/ftb_dfprobe.sbatch)
      awk '/SCRATCHFREE/ {n[$1]=$3} END {exit !(n["slinky-4"] >= 6*39+46+14 && n["slinky-15"] >= 8*39+46 && n["slinky-18"] >= 6*39+46)}' $Q/logs/ftb_dfprobe-$P3.log && break
    fi
    touch $CTL/FTB_WAIT_SPARES; sleep 300
  done
  rm -f $CTL/FTB_WAIT_SPARES
fi
FJB=$(launch aloe-v2pure-17641 $([ $FB = spares ] && echo spares || echo $GB) $([ $FB = spares ] && echo 1 || echo $FB) | tail -1)
[ -n "$FJB" ] || { say "BLOCKED: FT-(b) sbatch failed"; echo "$(date) sbatch B" > $CTL/FTB_BLOCKED; exit 1; }
echo "$FJA $FJB" > $CTL/FTB_LAUNCHED
say "BOTH LAUNCHED: FT-(a) $FJA, FT-(b) $FJB"
