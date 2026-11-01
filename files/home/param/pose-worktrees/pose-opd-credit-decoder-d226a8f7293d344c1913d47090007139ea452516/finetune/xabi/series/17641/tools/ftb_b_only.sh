#!/bin/bash
# FT-(b) ONLY re-run of ftb_chain.sh steps 2 + 4b (10-22 17:3x PT): convert 27548 failed on slinky-7 (bare grep slot bug, fixed in ftb_convert.sbatch
# line 23). FT-(a) 27549 already launched. Uses the chain's SPLIT (control/FT_PLAN.txt): FT-(b) hosts + final save as planned. Same launch() as the chain.
set -u
Q=/data/home/amartyavarma/ppm/aloe-v2pure-17641; T=$Q/tools; CTL=$Q/control; LOG=$Q/logs/ftb_chain.log
C811=/data/home/amartyavarma/ppm/code-811; NODES16=$(cat $C811/nodes16_s7.txt); M=17641
say() { echo "[$(TZ=America/Los_Angeles date '+%m-%d %T PT')] $*" | tee -a $LOG; }
clean_sb() { env $(env | grep -oE '^(SLURM|SBATCH|SRUN|SALLOC)_[A-Za-z0-9_]*=' | tr -d = | sed 's/^/-u /') sbatch "$@"; }
dfree() { df -BG --output=avail /data | tail -1 | tr -dc 0-9; }
GB=$(grep '^FT-(b)' $CTL/FT_PLAN.txt | awk '{print $3}'); FB=$(grep '^FT-(b)' $CTL/FT_PLAN.txt | grep -oE 'final save [01]' | awk '{print $3}')
[ "$(echo $GB | tr , '\n' | wc -l)" = 8 ] && [ -n "$FB" ] || { say "BLOCKED(b-only): bad FT_PLAN"; exit 1; }
OUT=/data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-$M
rm -f $CTL/FTB_BLOCKED
if [ ! -f $OUT/COMPLETE ]; then
  [ "$(dfree)" -ge 386 ] || { say "BLOCKED(b-only): /data $(dfree) GB < 386"; echo "$(date) /data low" > $CTL/FTB_BLOCKED; exit 1; }
  CJ=$(FTB_STEP=$M clean_sb --parsable -w $NODES16 $T/ftb_convert.sbatch); say "b-only: convert job $CJ (/data $(dfree) GB free)"
  sleep 20; while squeue -h -j $CJ 2>/dev/null | grep -q .; do sleep 15; done
  [ "$(cat $OUT/COMPLETE 2>/dev/null)" = "$M" ] || { say "BLOCKED(b-only): convert $CJ did not finish local-$M"; echo "$(date) convert failed $CJ" > $CTL/FTB_BLOCKED; exit 1; }
fi
say "init $OUT COMPLETE ($(du -sBG $OUT | cut -f1)); /data $(dfree) GB free"
R=aloe-v2pure-17641; HS=$GB; NT=8,8,8,8,8,8,8,8; NH=8
FJ=$(FTB_FINAL_SAVE=$FB clean_sb --parsable -w $HS /data/home/amartyavarma/ppm/$R/run.sbatch)
[ -n "$FJ" ] || { say "BLOCKED(b-only): FT-(b) sbatch failed"; echo "$(date) sbatch B" > $CTL/FTB_BLOCKED; exit 1; }
XJ=$(XF_RUN=$R XF_REPO=InternalCan/ppm-$R XF_STEPS="33 67 100 133 167" XF_FINAL=200 XF_HOSTS=$HS XF_NTASKS=$NT \
     XF_TLOG=/data/home/amartyavarma/ppm/$R/logs/aloev2p${R#aloe-v2pure-}-$FJ.log XF_TJOB=$FJ clean_sb --parsable --dependency=after:$FJ -N$NH -w $HS $T/xfree_run.sbatch)
HJ=$(clean_sb --parsable --dependency=afterok:$FJ -N$NH -w $HS /data/home/amartyavarma/ppm/$R/hf_final_export.sbatch)
RJ=$(HFJ=$HJ clean_sb --parsable --dependency=afterok:$HJ -N$NH -w $HS /data/home/amartyavarma/ppm/$R/rm_saves_after_hf.sbatch)
printf '{"train_job": "%s", "hosts": "%s", "final_save": "%s", "xfree_job": "%s", "backup_job": "%s", "rm_saves_job": "%s", "repo_final": "InternalCan/ppm-%s-final", "repo_intermediates": "InternalCan/ppm-%s", "time": "%s"}\n' \
  "$FJ" "$HS" "$FB" "$XJ" "$HJ" "$RJ" "$R" "$R" "$(TZ=America/Los_Angeles date '+%F %T PT')" > /data/home/amartyavarma/ppm/$R/BACKUP_JOB
say "LAUNCHED $R: job $FJ on $HS (final save $FB); xfree $XJ, HF backup $HJ, save cleanup $RJ"
FJA=$(grep -oE 'LAUNCHED aloe-v2pure-17352: job [0-9]+' $LOG | tail -1 | awk '{print $NF}')
echo "$FJA $FJ" > $CTL/FTB_LAUNCHED; say "BOTH LAUNCHED: FT-(a) $FJA, FT-(b) $FJ"
