#!/bin/bash
#SBATCH -p all -N1 -n1 -c 4 --mem=16G --gres=none -t 1-00:00:00 -J si20ktxt-hf --exclude=slinky-4,slinky-7,slinky-18
# si20k-txt copy of si20k-vox-ppm/tools/hf_up.sbatch (-c 4 / 16G: fits the 8 spare CPUs of a training node during the runs). HF-ONLY backup of one export: <src dir> -> PRIVATE HF model repo
# InternalCan/<run>, path <tag>/ (hfb_up.py: node-scratch symlink farm, upload_large_folder, verify per-file size + HF LFS sha256 vs the
# upload-time local sha256 of EVERY file + 3 files streamed back, private == True), retry / back-off. NEVER GCS. On VERIFY PASS writes
# <src>/HF_BACKUP.json; with 'delete-after-verify' (intermediate exports only: their /data copy exists only to consolidate the node-local
# shards for the upload) it then deletes the /data copy (the node-local shards stay on the training nodes; HF holds the verified copy).
# CPU only, on slinky-15 (not a training node; never slinky-4 / slinky-18).   usage: sbatch -o LOG hf_up.sbatch <run dir> <src dir> <tag> [delete-after-verify]
Q=$1; W=$2; TAG=$3; DEL=${4:-}
RUN=$(basename $Q); R=InternalCan/$RUN; NAME=$RUN-$TAG; D=/data/home/amartyavarma/setup-logs/hfb; PY=/data/home/amartyavarma/venvs/duplex/bin/python
O=$Q/hf_backup; mkdir -p $O; L=$O/hf_$TAG-$SLURM_JOB_ID.log; F=/scratch/amartyavarma/hfb/$NAME; S=$O/spec-$TAG.json
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
[ -f $W/COMPLETE ] || { echo "[$(pt)] ABORT: $W/COMPLETE missing" >> $L; exit 1; }
mkdir -p $F /scratch/amartyavarma/hfb/xet-cache
printf '{"dirs": [["%s", "%s"]], "exclude": ["HF_BACKUP.json"]}\n' "$W" "$TAG" > $S
echo "[$(pt)] start job $SLURM_JOB_ID on $(hostname -s): $W ($(ls -A $W | wc -l) files, $(du -sb $W | cut -f1) B) -> private $R/$TAG/" >> $L
export PYTHONUNBUFFERED=1 HF_XET_HIGH_PERFORMANCE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 HF_XET_CACHE=/scratch/amartyavarma/hfb/xet-cache HFB_OWNER=$SLURM_JOB_ID
UP="$PY $D/hfb_up.py --repo $R --type model --farm $F --spec $S --manifest-name MANIFEST.$TAG.tsv --tag $TAG --require $W/COMPLETE --workers 16"
bo=60; att=0; vfail=0; hrc=0
while true; do
  att=$((att+1))
  if ! nice -n 10 $UP --verify-only --no-verify >> $L 2>&1; then echo "[$(pt)] attempt $att: repo create / farm failed; retry ${bo}s" >> $L; [ $att -ge 8 ] && { hrc=2; break; }; sleep $bo; bo=$((bo*2>600?600:bo*2)); continue; fi
  nice -n 10 $PY $D/hfb_up.py --child --repo $R --type model --farm $F --spec $S --workers 16 >> $L 2>&1; rc=$?
  if [ $rc -ne 0 ]; then echo "[$(pt)] attempt $att: upload rc $rc; retry ${bo}s" >> $L; [ $att -ge 8 ] && { hrc=1; break; }; sleep $bo; bo=$((bo*2>600?600:bo*2)); continue; fi
  nice -n 10 $UP --verify-only >> $L 2>&1; vrc=$?
  [ $vrc -eq 0 ] && break
  vfail=$((vfail+1)); echo "[$(pt)] VERIFY FAIL rc $vrc ($vfail)" >> $L; [ $vfail -ge 4 ] && { hrc=3; break; }; sleep $bo
done
if [ $hrc -eq 0 ]; then
  V=$(ls -t $D/verify/InternalCan__${RUN}__${TAG}__*.json 2>/dev/null | head -1)
  printf '{"repo": "%s", "path": "%s/", "private": true, "verify": "%s", "time": "%s", "job": "%s"}\n' "$R" "$TAG" "$V" "$(pt)" "$SLURM_JOB_ID" > $W/HF_BACKUP.json
  if [ "$DEL" = delete-after-verify ]; then
    case "$W" in $Q/exports/export-*) cp $W/HF_BACKUP.json $O/HF_BACKUP-$TAG.json; rm -rf "$W"; rm -rf $F
      echo "[$(pt)] $TAG: /data copy $W DELETED after HF VERIFY PASS (si20k-txt: the node-local shards were deleted by the watcher once this /data copy verified; HF $R/$TAG/ holds the verified copy)" >> $Q/exports/BACKUP_LOG.txt;;
    *) echo "[$(pt)] $TAG: delete-after-verify refused for $W (not under $Q/exports)" >> $L;; esac
  fi
fi
echo "[$(pt)] $TAG HF upload rc $hrc (0 = VERIFY PASS, private repo $R)" >> $L
mkdir -p $Q/exports; echo "[$(pt)] $TAG HF upload rc $hrc (0 = VERIFY PASS) private $R/$TAG/ job $SLURM_JOB_ID" >> $Q/exports/BACKUP_LOG.txt
exit $hrc
