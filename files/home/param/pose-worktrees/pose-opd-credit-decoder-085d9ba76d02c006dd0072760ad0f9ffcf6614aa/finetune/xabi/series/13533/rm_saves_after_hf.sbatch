#!/bin/bash
#SBATCH -J aloev2p13533-rmsaves -p all --ntasks-per-node=1 -c 1 --mem=2G --gres=none -t 0:30:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-v2pure-13533/logs/rm_saves-%j.log
# aloe-v2pure-13533 (10-20, coordinator): after the FINAL export-200 is on private HF (job $HFJ, VERIFY PASS on EVERY host), delete this finished run's
# node-local RESUMABLE saves (saves/step-*) on its 8 hosts (scratch_guard guard_rm), one line each -> DELETED-1019-hfb-hold.log. Intermediate exports kept.
# usage: HFJ=<hf job> sbatch --dependency=afterok:<hf job> -N8 -w <set B> rm_saves_after_hf.sbatch
: "${HFJ:?}"
export HFJ
srun -N$SLURM_NNODES -n$SLURM_NNODES --ntasks-per-node=1 -c1 --mem=1G bash -c '
source /data/home/amartyavarma/ppm/scratch_guard.sh
H=$(hostname -s); D=/data/home/amartyavarma/ppm/aloe-v2pure-13533; LOG=/data/home/amartyavarma/scratch-inspect/DELETED-1019-hfb-hold.log
R=InternalCan__ppm-aloe-v2pure-13533-final; S=/scratch/amartyavarma/aloe-v2pure-13533/saves
n=$(grep -F "job $HFJ" $D/logs/BACKUP_LOG.txt | grep -c "export-200 HF upload rc 0 (0 = VERIFY PASS)")
[ "$n" = 8 ] || { echo "$H ABORT: $n of 8 hosts VERIFY PASS for export-200 in job $HFJ"; exit 1; }
VJ=$D/logs/hf_verify/${R}__export-200__$H.json
python3 -c "import json,sys; j=json.load(open(sys.argv[1])); sys.exit(0 if j.get(\"PASS\") is True and j.get(\"private\") is True and j.get(\"n_missing\")==0 and not j.get(\"size_mismatch\") and not j.get(\"sha_mismatch\") else 1)" $VJ \
  || { echo "$H ABORT: verify json $VJ not clean"; exit 1; }
[ -f /scratch/amartyavarma/aloe-v2pure-13533/exports/export-200/rank$(( 8 * ($(echo slinky-5,slinky-6,slinky-8,slinky-10,slinky-11,slinky-16,slinky-17,slinky-19 | tr , "\n" | grep -nx $H | cut -d: -f1) - 1) )).pt ] || { echo "$H ABORT: export-200 rank file missing"; exit 1; }
for d in $S/step-*; do
  [ -d "$d" ] || continue
  sz=$(du -s --block-size=1G "$d" | cut -f1)
  if guard_rm "$d"; then
    echo "$(TZ=America/Los_Angeles date "+%m-%d %H:%M PT") $H deleted $d (${sz}G; resumable save of the FINISHED aloe-v2pure-13533 FT job 26187 rc 0; final export-200 node-local + private HF InternalCan/ppm-aloe-v2pure-13533-final/export-200/ VERIFY PASS on all 8 hosts, job $HFJ, verify $VJ) coordinator 10-20; free now $(df -BG --output=avail /scratch | tail -1 | tr -d " ")" | tee -a $LOG
  else echo "$H guard refused $d"; fi
done
echo "$H done; /scratch free $(df -BG --output=avail /scratch | tail -1 | tr -d " ")"' | sort -V
