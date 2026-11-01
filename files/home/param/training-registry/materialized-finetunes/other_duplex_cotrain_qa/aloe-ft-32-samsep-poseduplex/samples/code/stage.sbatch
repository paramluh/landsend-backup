#!/bin/bash
#SBATCH -J pds-stage -p all -N1 -n1 -c 24 --mem=32G -t 01:00:00 --gres=none
#SBATCH -o /data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/samples/logs/stage-%j-%N.log
# usage: sbatch -w <host> stage.sbatch <repo> <tag>   -> /scratch/amartyavarma/pdsamples/<tag>. Floor: /scratch keeps >= 64 GiB free after the copy.
R=$1; T=$2; C=/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/samples/code; DEST=/scratch/amartyavarma/pdsamples/$T
NEED=90; AV=$(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9); echo "$(date +%T) $(hostname): /scratch avail ${AV}G, need ${NEED}G + 64G floor"
[ -f $DEST/COMPLETE ] || [ $AV -ge $((NEED + 64)) ] || { echo "FLOOR: not enough /scratch room"; exit 3; }
mkdir -p /scratch/amartyavarma/pdsamples
nice -n 10 /data/home/amartyavarma/venvs/duplex/bin/python -I $C/stage_pd.py $R $T $DEST
df -BG --output=avail /scratch | tail -1
