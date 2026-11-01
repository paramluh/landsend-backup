#!/bin/bash
# pdsamples host-to-host pull of the pose-duplex FINAL export (main 10-25): serve jobs on the 8 pose hosts + pull jobs on slinky-18 / slinky-4.
S=/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/samples; C=$S/code; X=$S/xfer; L=$S/logs; PY=/data/home/amartyavarma/venvs/duplex/bin/python
T=${1:-export-498}; D=/scratch/amartyavarma/aloe-ft-32-samsep/exports-poseduplex-v5/$T
HOSTS=slinky-1,slinky-2,slinky-6,slinky-15,slinky-7,slinky-13,slinky-14,slinky-17
mkdir -p $X; rm -f $X/*.serve $X/*.sha256.json $X/PULL_DONE
for h in ${HOSTS//,/ }; do
  sbatch --parsable -J pds-serve -w $h -n1 -c 4 --mem=4G --gres=none -t 03:00:00 -o $L/xfer-serve-%j-%N.log --wrap "nice -n 10 $PY -I $C/xfer_serve.py $D $X 18741"
done
for h in slinky-18 slinky-4; do
  sbatch --parsable -J pds-pull -w $h -n1 -c 16 --mem=16G --gres=none -t 03:00:00 -o $L/xfer-pull-%j-%N.log \
    --wrap "AV=\$(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9); echo avail \${AV}G; [ \$AV -ge 155 ] || { echo FLOOR; exit 3; }; mkdir -p /scratch/amartyavarma/pdsamples; nice -n 10 $PY -I $C/xfer_pull.py $X /scratch/amartyavarma/pdsamples/$T $HOSTS; df -BG --output=avail /scratch | tail -1"
done
