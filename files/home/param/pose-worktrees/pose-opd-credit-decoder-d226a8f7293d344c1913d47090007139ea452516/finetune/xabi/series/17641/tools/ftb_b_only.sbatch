#!/bin/bash
#SBATCH -J ftb-bonly -p all -N1 -n1 -c 1 --mem=2G --gres=none -t 4:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-v2pure-17641/logs/ftb_bonly-%j.log
exec bash /data/home/amartyavarma/ppm/aloe-v2pure-17641/tools/ftb_b_only.sh
