#!/bin/bash
#SBATCH -J evalvox-stage -p all -N1 -n1 -c 8 --mem=16G -t 02:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-ft-32-samsep/logs/evalvox-stage-%j-%N.log
# usage: sbatch -w <host> stage.sbatch <tag> <host index in subset_8_6.json>. Waits until /scratch keeps >= 30 GB floor after the download.
T=$1; HI=$2; E=/data/home/amartyavarma/ppm/aloe-ft-32-samsep/evalvox; DEST=${3:-/scratch/amartyavarma/evalvox/$T}   # 3rd arg: dest (run-scoped, e.g. .../evalvox/ppmvox-v2/w-N); HF repo from EVALVOX_HF_REPO
NEED=$([ "$HI" = 0 ] && echo 58 || echo 42)
until [ $(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9) -ge $((NEED + 30)) ] || [ -f $DEST/COMPLETE ]; do echo "$(date +%T) waiting for /scratch room (need ${NEED}+30 GB, have $(df -BG --output=avail /scratch | tail -1 | tr -d ' '))"; sleep 120; done
mkdir -p $(dirname $DEST); cd /tmp
nice -n 10 /data/home/amartyavarma/venvs/duplex/bin/python -I $E/stage_export.py $T $E/subset_8_6.json $HI $DEST
