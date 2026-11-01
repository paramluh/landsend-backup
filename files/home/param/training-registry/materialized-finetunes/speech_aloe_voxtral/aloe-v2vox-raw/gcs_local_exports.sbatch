#!/bin/bash
#SBATCH -J aloev2vraw-gcs-local -p all --ntasks-per-node=1 -c 4 --mem=32G --gres=none -t 1-00:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-v2vox-raw/logs/gcs_local-%j.log
# aloe-v2vox-raw (= aloe-v2vox-ft/gcs_local_exports.sbatch, job 22420's; -N = the training job's node count; an export root that is a symlink to
# /data/.../node-exports/<host> is uploaded the same way). After the training job (--dependency=afterok:<train job> -w <the training hosts>):
# every node uploads its node-local export shards (/scratch/amartyavarma/aloe-v2vox-raw/exports/export-N: rank files of its 8 ranks, + rep.pt /
# meta.json / COMPLETE on rank 0's node) to gs://podcast-raw-audio-data-mass/amartyavarma/ckpt-backup/aloe-v2vox-raw-local/export-N/ (disjoint files
# per node), each file verified (size + md5 + crc32c) by gcs_up.py (rc 0 = VERIFY PASS). Then, ONLY for the 5 INTERMEDIATE exports whose upload
# verified on this node, the node-local copy is deleted (frees the intermediate shards for later runs); export-253 stays local.
# The consolidated /data export (export-253) is uploaded by node 0 too (prefix .../aloe-v2vox-raw/export-253). Sources only read except the deletions.
srun bash -c '
export GOOGLE_APPLICATION_CREDENTIALS=/data/home/amartyavarma/speechsdk-ebf4e74f9a4c.json GCS_KEY=/data/home/amartyavarma/speechsdk-ebf4e74f9a4c.json PYTHONUNBUFFERED=1 GCS_HASH_THREADS=4 GCS_MULT=2
PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python; UP=/data/home/amartyavarma/setup-logs/hfb/gcs_up.py; H=$(hostname -s); rc=0
S=/scratch/amartyavarma/gcsbk/sums/aloe-v2vox-raw-local; mkdir -p $S
for d in $(ls -d /scratch/amartyavarma/aloe-v2vox-raw/exports/export-* 2>/dev/null | sort -V); do
  d=$(readlink -f $d)
  T=$(basename $d); r=0
  python3 -c "import json,sys; json.dump({\"dirs\": [[sys.argv[1], sys.argv[2]]], \"exclude\": []}, open(sys.argv[3], \"w\"))" $d $T $S/spec-$T-$H.json
  nice -n 10 $PY -W ignore $UP --spec $S/spec-$T-$H.json --prefix amartyavarma/ckpt-backup/aloe-v2vox-raw-local --tag $T --workers 4 \
    --manifest-name MANIFEST.$T.$H.tsv --verify-out /data/home/amartyavarma/ppm/aloe-v2vox-raw/logs/gcs_verify --sums-dir $S || r=1
  echo "$H $T rc $r"; [ $r = 0 ] || rc=1
  if [ $r = 0 ] && [ "$T" != export-253 ]; then du -sh $d; rm -rf $d && echo "$H deleted node-local $d (GCS verify PASS)"; fi
done
if [ "$SLURM_NODEID" = 0 ] && [ -f /data/home/amartyavarma/ppm/aloe-v2vox-raw/export-253/COMPLETE ]; then
  python3 -c "import json,sys; json.dump({\"dirs\": [[sys.argv[1], sys.argv[2]]], \"exclude\": []}, open(sys.argv[3], \"w\"))" /data/home/amartyavarma/ppm/aloe-v2vox-raw/export-253 export-253 $S/spec-weka-253.json
  nice -n 10 $PY -W ignore $UP --spec $S/spec-weka-253.json --prefix amartyavarma/ckpt-backup/aloe-v2vox-raw --tag export-253 --workers 4 \
    --manifest-name MANIFEST.export-253.tsv --verify-out /data/home/amartyavarma/ppm/aloe-v2vox-raw/logs/gcs_verify --sums-dir $S || rc=1
  echo "$H weka export-253 rc $rc"
fi
df -BG /scratch | tail -1
exit $rc'
