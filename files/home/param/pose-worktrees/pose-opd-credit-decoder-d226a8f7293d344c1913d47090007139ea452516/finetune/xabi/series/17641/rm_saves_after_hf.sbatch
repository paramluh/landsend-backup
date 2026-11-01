#!/bin/bash
#SBATCH -J aloev2p17641-rmsaves -p all --ntasks-per-node=1 -c 1 --mem=2G --gres=none -t 0:30:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-v2pure-17641/logs/rm_saves-%j.log
# aloe-v2pure-17641 (10-22) = aloe-v2pure-13533/rm_saves_after_hf.sbatch's rule: after the FINAL export-200 is on private HF (job $HFJ, VERIFY PASS on
# EVERY one of the run hosts), delete this finished run's node-local RESUMABLE saves (saves/step-*) via scratch_guard guard_rm, one line each ->
# scratch-inspect/ft17352-cleanup-1022/DELETED.log. The node-local export-200 is kept.
# usage: HFJ=<hf job> sbatch --dependency=afterok:<hf job> -N<n> -w <the FT hosts> rm_saves_after_hf.sbatch
: "${HFJ:?}"
export HFJ
srun -N$SLURM_NNODES -n$SLURM_NNODES --ntasks-per-node=1 -c1 --mem=1G bash -c '
source /data/home/amartyavarma/ppm/scratch_guard.sh
H=$(hostname -s); D=/data/home/amartyavarma/ppm/aloe-v2pure-17641; LOG=/data/home/amartyavarma/scratch-inspect/ft17352-cleanup-1022/DELETED.log
R=InternalCan__ppm-aloe-v2pure-17641-final; S=/scratch/amartyavarma/aloe-v2pure-17641/saves
n=$(grep -F "job $HFJ" $D/logs/BACKUP_LOG.txt | grep -c "export-200 HF upload rc 0 (0 = VERIFY PASS)")
[ "$n" = "$SLURM_NNODES" ] || { echo "$H ABORT: $n of $SLURM_NNODES hosts VERIFY PASS for export-200 in job $HFJ"; exit 1; }
VJ=$D/logs/hf_verify/${R}__export-200__$H.json
python3 -c "import json,sys; j=json.load(open(sys.argv[1])); sys.exit(0 if j.get(\"PASS\") is True and j.get(\"private\") is True and j.get(\"n_missing\")==0 and not j.get(\"size_mismatch\") and not j.get(\"sha_mismatch\") else 1)" $VJ \
  || { echo "$H ABORT: verify json $VJ not clean"; exit 1; }
ls /scratch/amartyavarma/aloe-v2pure-17641/exports/export-200/rank*.pt > /dev/null 2>&1 || { echo "$H ABORT: export-200 rank files missing"; exit 1; }
for d in $S/step-*; do
  [ -d "$d" ] || continue
  sz=$(du -s --block-size=1G "$d" | cut -f1)
  if guard_rm "$d"; then
    echo "$(TZ=America/Los_Angeles date "+%m-%d %H:%M PT") $H deleted $d (${sz}G; resumable save of the FINISHED aloe-v2pure-17641 FT; final export-200 node-local + private HF InternalCan/ppm-aloe-v2pure-17641-final/export-200/ VERIFY PASS on all the run hosts, job $HFJ, verify $VJ) [10-22 scaling series]; free now $(df -BG --output=avail /scratch | tail -1 | tr -d " ")" | tee -a $LOG
  else echo "$H guard refused $d"; fi
done
echo "$H done; /scratch free $(df -BG --output=avail /scratch | tail -1 | tr -d " ")"' | sort -V
