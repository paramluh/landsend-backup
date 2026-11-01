#!/bin/bash
#SBATCH -J m1vox-clean -p all -N8 --ntasks-per-node=1 -c 2 --mem=4G --gres=none -t 0:15:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-m1vox-ft/logs/clean-22315-%j.log
# discard job 22315's outputs (superseded by the compiled-decoder relaunch; user: delete its saves): node-local exports + the /data stop export
srun bash -c 'H=$(hostname -s); echo "$H removing $(ls /scratch/amartyavarma/aloe-m1vox-ft/exports 2>/dev/null | tr "\n" " ")"; rm -rf /scratch/amartyavarma/aloe-m1vox-ft/exports; echo "$H free $(df -BG --output=avail /scratch | tail -1)"'
ls -d /data/home/amartyavarma/ppm/aloe-m1vox-ft/export-366-stop* 2>/dev/null && rm -rf /data/home/amartyavarma/ppm/aloe-m1vox-ft/export-366-stop* && echo "removed /data stop export"
exit 0
