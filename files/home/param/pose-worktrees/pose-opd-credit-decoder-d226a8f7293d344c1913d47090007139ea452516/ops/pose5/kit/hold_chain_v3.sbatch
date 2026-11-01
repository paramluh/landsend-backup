#!/bin/bash
#SBATCH -J ppm811s2p-holdchain -p all -w slinky-4 -N1 -n1 -c1 --mem=4G -t 4-00:00:00 -o /data/home/amartyavarma/ppm/pose5/kit/logs/hold_chain-%j.log
# pose5 chained hold for PPM_KEEP_CKPT=1 (kit/hold_chain_v3.sh). usage: TARGET=<max steps> PIN_STEP=17850 sbatch [-w <spare>] hold_chain.sbatch
exec bash /data/home/amartyavarma/ppm/pose5/kit/hold_chain_v3.sh
