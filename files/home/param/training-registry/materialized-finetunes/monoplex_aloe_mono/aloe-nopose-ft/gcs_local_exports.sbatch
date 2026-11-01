#!/bin/bash
#SBATCH -J aloenoposeft-gcs-local -p all -N8 -w slinky-1,slinky-10,slinky-11,slinky-13,slinky-14,slinky-16,slinky-17,slinky-19 --ntasks-per-node=1 -c 8 --mem=32G --gres=none -t 1-00:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-nopose-ft/logs/gcs_local-%j.log
# After job 22291 (aloe-nopose-ft, the no-pose ablation of 22091): every node uploads its node-local export shards (/scratch/amartyavarma/aloe-nopose-ft/exports/export-N: rank files of its 8 ranks,
# + rep.pt / meta.json / COMPLETE on rank 0's node) to gs://podcast-raw-audio-data-mass/amartyavarma/ckpt-backup/aloe-nopose-ft-local/export-N/
# (disjoint files per node), each file verified (size + md5 + crc32c) by gcs_up.py. Sources only read.
srun bash -c '
export GOOGLE_APPLICATION_CREDENTIALS=/data/home/amartyavarma/speechsdk-ebf4e74f9a4c.json GCS_KEY=/data/home/amartyavarma/speechsdk-ebf4e74f9a4c.json PYTHONUNBUFFERED=1 GCS_HASH_THREADS=4 GCS_MULT=2
PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python; UP=/data/home/amartyavarma/setup-logs/hfb/gcs_up.py; H=$(hostname -s); rc=0
for d in /scratch/amartyavarma/aloe-nopose-ft/exports/export-*; do
  T=$(basename $d); S=/scratch/amartyavarma/gcsbk/sums/aloe-nopose-ft-local; mkdir -p $S
  python3 -c "import json,sys; json.dump({\"dirs\": [[sys.argv[1], sys.argv[2]]], \"exclude\": []}, open(sys.argv[3], \"w\"))" $d $T $S/spec-$T-$H.json
  nice -n 10 $PY -W ignore $UP --spec $S/spec-$T-$H.json --prefix amartyavarma/ckpt-backup/aloe-nopose-ft-local --tag $T --workers 4 \
    --manifest-name MANIFEST.$T.$H.tsv --verify-out /data/home/amartyavarma/ppm/aloe-nopose-ft/logs/gcs_verify --sums-dir $S || rc=1
  echo "$H $T rc $rc"
done
exit $rc'
