#!/bin/bash
#SBATCH -p all -N1 -n1 -c 16 --mem=48G --gres=none -t 1-00:00:00 -J nextvox-hf
#SBATCH --exclude=slinky-4,slinky-7,slinky-15,slinky-17,slinky-18,slinky-9,slinky-10,slinky-11,slinky-12,slinky-13,slinky-14,slinky-16,slinky-19
#SBATCH -o /data/home/amartyavarma/ppm/aloe-nextvox-6242/logs/hf-%j.out
# HF-ONLY backup of one aloe-nextvox-6242 export (= next-arch/hf_backup/hf_w.sbatch with the source / repo changed): <src dir> -> PRIVATE HF
# model repo InternalCan/aloe-nextvox-6242, path <tag>/ (hfb_up.py: symlink farm on node scratch, upload_large_folder, verify per-file size + HF
# LFS sha256 vs the upload-time local sha256 of EVERY file + 3 files streamed back, private == True), retry / back-off. Never GCS.
# On VERIFY PASS writes <src>/HF_BACKUP.json. CPU only, never on the training / forbidden nodes.   sbatch hf_up.sbatch <src dir> <tag>
W=$1; TAG=$2
R=InternalCan/aloe-nextvox-6242; NAME=aloe-nextvox-$TAG; D=/data/home/amartyavarma/setup-logs/hfb; PY=/data/home/amartyavarma/venvs/duplex/bin/python
O=/data/home/amartyavarma/ppm/aloe-nextvox-6242/hf_backup; mkdir -p $O; L=$O/hf_$TAG-$SLURM_JOB_ID.log; F=/scratch/amartyavarma/hfb/$NAME; S=$O/spec-$TAG.json
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
  V=$(ls -t $D/verify/InternalCan__aloe-nextvox-6242__${TAG}__*.json 2>/dev/null | head -1)
  printf '{"repo": "%s", "path": "%s/", "private": true, "verify": "%s", "time": "%s", "job": "%s"}\n' "$R" "$TAG" "$V" "$(pt)" "$SLURM_JOB_ID" > $W/HF_BACKUP.json
fi
echo "[$(pt)] $TAG HF upload rc $hrc (0 = VERIFY PASS, private repo $R)" >> $L
echo "[$(pt)] $TAG HF upload rc $hrc (0 = VERIFY PASS) private $R/$TAG/ job $SLURM_JOB_ID" >> /data/home/amartyavarma/ppm/aloe-nextvox-6242/exports/BACKUP_LOG.txt
exit $hrc
