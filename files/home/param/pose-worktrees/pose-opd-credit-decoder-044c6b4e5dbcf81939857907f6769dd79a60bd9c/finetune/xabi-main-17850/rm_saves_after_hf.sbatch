#!/bin/bash
#SBATCH -J aloev2pm17850-rmsaves -p all --ntasks-per-node=1 -c 1 --mem=2G --gres=none -t 0:30:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-v2pure-main-17850/logs/rm_saves-%j.log
# aloe-v2pure-main-17850 (MONOFT) = aloe-v2pure-17641/rm_saves_after_hf.sbatch (job 27625) with this run's names / dirs, + one MONOFT line: after the
# FINAL export-200 is on private HF (job $HFJ, VERIFY PASS on EVERY one of the run hosts), delete this finished run's node-local RESUMABLE saves
# (saves/step-*) AND its node-local init copy (/scratch/amartyavarma/aloe-v2pure-main-17850-init: bf16 shards derived from dcp-17850; deleted only while
# dcp-17850 itself is still COMPLETE on the host, else kept + logged) via scratch_guard guard_rm, one line each -> scratch-inspect/monoft-cleanup/DELETED.log.
# The node-local export-200 is kept.
# usage: HFJ=<hf job> sbatch --dependency=afterok:<hf job> -N<n> -w <the FT hosts> rm_saves_after_hf.sbatch
: "${HFJ:?}"
export HFJ
mkdir -p /data/home/amartyavarma/scratch-inspect/monoft-cleanup
srun -N$SLURM_NNODES -n$SLURM_NNODES --ntasks-per-node=1 -c1 --mem=1G bash -c '
source /data/home/amartyavarma/ppm/scratch_guard.sh
H=$(hostname -s); D=/data/home/amartyavarma/ppm/aloe-v2pure-main-17850; LOG=/data/home/amartyavarma/scratch-inspect/monoft-cleanup/DELETED.log
R=InternalCan__ppm-aloe-v2pure-main-17850-final; S=/scratch/amartyavarma/aloe-v2pure-main-17850/saves
n=$(grep -F "job $HFJ" $D/logs/BACKUP_LOG.txt | grep -c "export-200 HF upload rc 0 (0 = VERIFY PASS)")
[ "$n" = "$SLURM_NNODES" ] || { echo "$H ABORT: $n of $SLURM_NNODES hosts VERIFY PASS for export-200 in job $HFJ"; exit 1; }
VJ=$D/logs/hf_verify/${R}__export-200__$H.json
python3 -c "import json,sys; j=json.load(open(sys.argv[1])); sys.exit(0 if j.get(\"PASS\") is True and j.get(\"private\") is True and j.get(\"n_missing\")==0 and not j.get(\"size_mismatch\") and not j.get(\"sha_mismatch\") else 1)" $VJ \
  || { echo "$H ABORT: verify json $VJ not clean"; exit 1; }
ls /scratch/amartyavarma/aloe-v2pure-main-17850/exports/export-200/rank*.pt > /dev/null 2>&1 || { echo "$H ABORT: export-200 rank files missing"; exit 1; }
for d in $S/step-*; do
  [ -d "$d" ] || continue
  sz=$(du -s --block-size=1G "$d" | cut -f1)
  if guard_rm "$d"; then
    echo "$(TZ=America/Los_Angeles date "+%m-%d %H:%M PT") $H deleted $d (${sz}G; resumable save of the FINISHED aloe-v2pure-main-17850 FT; final export-200 node-local + private HF InternalCan/ppm-aloe-v2pure-main-17850-final/export-200/ VERIFY PASS on all the run hosts, job $HFJ, verify $VJ) [MONOFT]; free now $(df -BG --output=avail /scratch | tail -1 | tr -d " ")" | tee -a $LOG
  else echo "$H guard refused $d"; fi
done
I=/scratch/amartyavarma/aloe-v2pure-main-17850-init; src=""
for x in /scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-17850 /scratch/amartyavarma/ppm/ckpt-keep*/dcp-17850; do [ -f $x/COMPLETE ] && { src=$x; break; }; done
if [ -d $I ] && [ -n "$src" ]; then
  sz=$(du -s --block-size=1G "$I" | cut -f1)
  if guard_rm "$I"; then
    echo "$(TZ=America/Los_Angeles date "+%m-%d %H:%M PT") $H deleted $I (${sz}G; node-local bf16 init copy of the FINISHED aloe-v2pure-main-17850 FT, derived from $src which is still COMPLETE on this host) [MONOFT]; free now $(df -BG --output=avail /scratch | tail -1 | tr -d " ")" | tee -a $LOG
  else echo "$H guard refused $I"; fi
elif [ -d $I ]; then echo "$H KEPT $I: dcp-17850 no longer COMPLETE on this host (the init copy may be its only local copy)"; fi
echo "$H done; /scratch free $(df -BG --output=avail /scratch | tail -1 | tr -d " ")"' | sort -V
