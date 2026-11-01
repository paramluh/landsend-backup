#!/bin/bash
#SBATCH -J npl3-hold -p all -N16 --ntasks-per-node=1 -c 16 --mem=28G --gres=none -t 8:00:00
#SBATCH -o /data/home/amartyavarma/ppm/code-811/fresh/append/npl3/logs/hold-%j.log
# 10-23 delivery-3 ingest: CPU-only holder allocation on the 16 PPM hosts (16 of 64 CPUs, 28G per host, no GPU: the hosts have ~590G schedulable, MemSpecLimit 156G; FT 500G + xfree 16G + hfb pin 16G + this 28G still fit), co-scheduled with
# the two Aloe FTs (32 CPUs / 500G each) without delaying them. The ingest steps run in it via srun --jobid <this> --overlap (npl3_ingest.sbatch).
# Exits when npl3/ctl/RELEASE exists (written at the end of the ingest, or by the resume waiter before it submits the main run, which needs whole nodes).
echo "hold $SLURM_JOB_ID on $SLURM_JOB_NODELIST $(TZ=America/Los_Angeles date '+%F %T PT')"
while [ ! -f /data/home/amartyavarma/ppm/code-811/fresh/append/npl3/ctl/RELEASE ]; do sleep 20; done
echo "released $(TZ=America/Los_Angeles date '+%F %T PT')"
