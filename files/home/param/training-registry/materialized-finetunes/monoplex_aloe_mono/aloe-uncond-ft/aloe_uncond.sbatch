#!/bin/bash
#SBATCH -J aloeuncondft -p all -N8 --ntasks-per-node=8 --gres=gpu:8 -t 12:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-uncond-ft/logs/aloeuncondft-%j.log
# aloe-uncond-ft (10-16) GATE: submitted --dependency=afterany:22291 -w <22291's 8 nodes> so the nodes are held the moment the no-pose
# ablation frees them; the trainer (aloe_uncond_run.sh, read at run time) starts as soon as control/READY exists (written once the code is final).
Q=/data/home/amartyavarma/ppm/aloe-uncond-ft
echo "== aloe-uncond-ft gate: job $SLURM_JOB_ID allocated $SLURM_JOB_NODELIST $(TZ=America/Los_Angeles date '+%F %T PT'); waiting for $Q/control/READY"
while [ ! -f $Q/control/READY ]; do sleep 5; done
echo "== READY $(TZ=America/Los_Angeles date '+%T PT'): exec aloe_uncond_run.sh"
exec bash $Q/aloe_uncond_run.sh
