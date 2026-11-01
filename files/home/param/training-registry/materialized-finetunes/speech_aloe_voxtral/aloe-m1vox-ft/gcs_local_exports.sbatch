#!/bin/bash
#SBATCH -J aloem1voxft-gcs-local -p all -N8 --ntasks-per-node=1 -c 8 --mem=32G --gres=none -t 1-00:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-m1vox-ft/logs/gcs_local-%j.log
# After the aloe-m1vox-ft training job (launch.sh submits this with --dependency=afterok:<train job>; = aloe-pose-ft/gcs_local_exports.sbatch):
# every node uploads its node-local export shards (/scratch/amartyavarma/aloe-m1vox-ft/exports/export-N: rank files of its 8 ranks, + rep.pt /
# meta.json / COMPLETE on rank 0's node) to gs://podcast-raw-audio-data-mass/amartyavarma/ckpt-backup/aloe-m1vox-ft-local/export-N/ (disjoint files
# per node), each file verified (size + md5 + crc32c) by gcs_up.py (rc 0 = VERIFY PASS). Then, ONLY for the 5 INTERMEDIATE exports whose upload
# verified on this node, the node-local copy is deleted (frees ~11 GB x 5 per node for PPM's own saves when it resumes); export-366 stays local.
# The consolidated /data export (export-366) is uploaded by node 0 too (prefix .../aloe-m1vox-ft/export-366). Sources only read except the deletions.
srun bash -c '
export GOOGLE_APPLICATION_CREDENTIALS=/data/home/amartyavarma/speechsdk-ebf4e74f9a4c.json GCS_KEY=/data/home/amartyavarma/speechsdk-ebf4e74f9a4c.json PYTHONUNBUFFERED=1 GCS_HASH_THREADS=4 GCS_MULT=2
PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python; UP=/data/home/amartyavarma/setup-logs/hfb/gcs_up.py; H=$(hostname -s); rc=0
S=/scratch/amartyavarma/gcsbk/sums/aloe-m1vox-ft-local; mkdir -p $S
for d in $(ls -d /scratch/amartyavarma/aloe-m1vox-ft/exports/export-* 2>/dev/null | sort -V); do
  T=$(basename $d); r=0
  python3 -c "import json,sys; json.dump({\"dirs\": [[sys.argv[1], sys.argv[2]]], \"exclude\": []}, open(sys.argv[3], \"w\"))" $d $T $S/spec-$T-$H.json
  nice -n 10 $PY -W ignore $UP --spec $S/spec-$T-$H.json --prefix amartyavarma/ckpt-backup/aloe-m1vox-ft-local --tag $T --workers 4 \
    --manifest-name MANIFEST.$T.$H.tsv --verify-out /data/home/amartyavarma/ppm/aloe-m1vox-ft/logs/gcs_verify --sums-dir $S || r=1
  echo "$H $T rc $r"; [ $r = 0 ] || rc=1
  if [ $r = 0 ] && [ "$T" != export-366 ]; then du -sh $d; rm -rf $d && echo "$H deleted node-local $d (GCS verify PASS)"; fi
done
if [ "$SLURM_NODEID" = 0 ] && [ -f /data/home/amartyavarma/ppm/aloe-m1vox-ft/export-366/COMPLETE ]; then
  python3 -c "import json,sys; json.dump({\"dirs\": [[sys.argv[1], sys.argv[2]]], \"exclude\": []}, open(sys.argv[3], \"w\"))" /data/home/amartyavarma/ppm/aloe-m1vox-ft/export-366 export-366 $S/spec-weka-430.json
  nice -n 10 $PY -W ignore $UP --spec $S/spec-weka-430.json --prefix amartyavarma/ckpt-backup/aloe-m1vox-ft --tag export-366 --workers 4 \
    --manifest-name MANIFEST.export-366.tsv --verify-out /data/home/amartyavarma/ppm/aloe-m1vox-ft/logs/gcs_verify --sums-dir $S || rc=1
  echo "$H weka export-366 rc $rc"
fi
df -BG /scratch | tail -1
exit $rc'
