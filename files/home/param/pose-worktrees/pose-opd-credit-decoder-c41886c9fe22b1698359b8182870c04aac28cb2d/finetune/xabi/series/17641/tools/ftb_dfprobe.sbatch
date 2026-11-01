#!/bin/bash
#SBATCH -J ftb-dfprobe -p all -N16 --ntasks-per-node=1 -c 1 --mem=1G --gres=none -t 0:10:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-v2pure-17641/logs/ftb_dfprobe-%j.log
# aloe-v2pure-17641: print each PPM host's /scratch free (ftb_chain.sh re-picks FT-(b)'s 8 hosts from it when the first pick was short).
srun -N16 -n16 --ntasks-per-node=1 -c 1 --mem=1G --cpu-bind=none bash -c 'echo "$(hostname -s) SCRATCHFREE $(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9)"'
