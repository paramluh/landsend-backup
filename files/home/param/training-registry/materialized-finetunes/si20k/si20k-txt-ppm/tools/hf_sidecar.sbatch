#!/bin/bash
#SBATCH -p all -N1 -n1 -c 2 --mem=8G --gres=none -t 2:00:00 -J si20ktxt-hfside --exclude=slinky-4,slinky-7,slinky-18
# si20k-txt: add ONE sidecar file (THETA_FP32.json: the exact fp32 vgain theta of an export that is already HF-verified and no longer on
# disk) into its HF folder InternalCan/<run>/<tag>/ with hfb_up.py (own manifest name MANIFEST.<tag>-thetafix.tsv, so the export's original
# manifest is untouched), then VERIFY (per-file size + LFS sha256 vs the upload-time local sha256). Upload only, nothing downloaded or deleted.
# usage: sbatch -o LOG hf_sidecar.sbatch <run dir> <sidecar file> <tag>
Q=$1; SF=$2; TAG=$3
RUN=$(basename $Q); R=InternalCan/$RUN; D=/data/home/amartyavarma/setup-logs/hfb; PY=/data/home/amartyavarma/venvs/duplex/bin/python
O=$Q/hf_backup; mkdir -p $O; L=$O/hf_${TAG}-thetafix-$SLURM_JOB_ID.log; F=/scratch/amartyavarma/hfb/$RUN-$TAG-thetafix; S=$O/spec-$TAG-thetafix.json
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
mkdir -p $F /scratch/amartyavarma/hfb/xet-cache
printf '{"files": [["%s", "%s/THETA_FP32.json"]]}\n' "$SF" "$TAG" > $S
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1 HF_XET_CACHE=/scratch/amartyavarma/hfb/xet-cache HFB_OWNER=$SLURM_JOB_ID
UP="$PY $D/hfb_up.py --repo $R --type model --farm $F --spec $S --manifest-name MANIFEST.$TAG-thetafix.tsv --tag $TAG-thetafix --workers 2"
rc=1
for att in 1 2 3 4; do
  nice -n 10 $UP --verify-only --no-verify >> $L 2>&1 && nice -n 10 $PY $D/hfb_up.py --child --repo $R --type model --farm $F --spec $S --workers 2 >> $L 2>&1 \
    && nice -n 10 $UP --verify-only >> $L 2>&1 && { rc=0; break; }
  sleep 60
done
rm -rf $F
echo "[$(pt)] $TAG/THETA_FP32.json sidecar HF upload rc $rc (0 = VERIFY PASS) private $R/$TAG/ job $SLURM_JOB_ID" | tee -a $L >> $Q/exports/BACKUP_LOG.txt
exit $rc
