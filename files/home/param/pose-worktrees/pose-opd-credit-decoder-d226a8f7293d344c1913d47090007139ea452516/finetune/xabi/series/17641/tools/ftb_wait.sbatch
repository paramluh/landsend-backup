#!/bin/bash
#SBATCH -J ftb-wait -p all -N1 -n1 -c 1 --mem=2G --gres=none -t 2-00:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-v2pure-17641/logs/ftb_wait-%j.log
# aloe-v2pure-17641 FT-(b) waiter: submitted with --dependency=afterany:<append train job>; runs tools/ftb_chain.sh (doc there). CPU-only; any node (no host assumptions: /data, squeue, sbatch only).
exec bash /data/home/amartyavarma/ppm/aloe-v2pure-17641/tools/ftb_chain.sh
