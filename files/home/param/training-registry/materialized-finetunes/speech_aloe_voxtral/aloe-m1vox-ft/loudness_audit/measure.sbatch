#!/bin/bash
#SBATCH -J loud-measure -p all -w slinky-4 -N1 -n1 -c 48 --mem=200G --gres=none -t 4:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-m1vox-ft/loudness_audit/logs/measure-%j.log
# loudness audit, measure step (CPU only, slinky-4): every m1vox train item's A waveform (GCS FLACs fetched to node scratch, deleted after use)
D=/data/home/amartyavarma/ppm/aloe-m1vox-ft/loudness_audit
export GOOGLE_APPLICATION_CREDENTIALS=/data/home/amartyavarma/speechsdk-ebf4e74f9a4c.json OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
nice -n 10 /data/home/amartyavarma/venvs/qwen-s2/bin/python -I $D/loudness_audit.py measure --out $D --tmp /scratch/amartyavarma/loudness_audit_tmp \
  --workers 44 --dl 16 --ahead 160 "$@"
rm -rf /scratch/amartyavarma/loudness_audit_tmp
