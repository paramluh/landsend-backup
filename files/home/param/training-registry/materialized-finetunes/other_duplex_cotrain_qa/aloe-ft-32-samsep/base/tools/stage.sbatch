#!/bin/bash
#SBATCH -J aloeft32-base-stage -p all -N2 -w slinky-4,slinky-18 --ntasks-per-node=1 -c 32 --mem=256G --gres=none -t 3:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-ft-32-samsep/base/logs/stage-%j.out
# ALOEFT32 base: dcp-$STEP (HF $REPO) -> /scratch/amartyavarma/aloe-samsep-$STEP-init on slinky-18 and slinky-4 (CPU only, no GPUs held)
srun -N2 -n2 --ntasks-per-node=1 -c 32 --cpu-bind=none --kill-on-bad-exit=0 bash /data/home/amartyavarma/ppm/aloe-ft-32-samsep/base/tools/stage_host.sh
