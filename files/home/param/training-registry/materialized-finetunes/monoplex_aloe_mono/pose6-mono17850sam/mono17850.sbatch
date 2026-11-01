#!/bin/bash
#SBATCH -J pose6-mono17850sam -p all -x slinky-9 --ntasks-per-node=8 --cpus-per-task=4 --gres=gpu:8 --mem=500G -t 2-00:00:00
#SBATCH -o /data/home/amartyavarma/ppm/pose6/mono17850sam/logs/ft-%j.log
# pose6 mono17850 (10-27): the aloe-pose-mono-8020 recipe (job 24272 -> export-357) with init = 8,1,1 dcp-17850 (local-17850, old arch, both mods
# off). NOT submitted by the kit: launch.sh submits it on --go only, with -N / -w / --mem / -t (DEFAULT 8 nodes = world 64: --mem=500G -t 2-00:00:00;
# FALLBACK 1 node: --mem=530G -t 1-06:00:00) and --export=ALL,FT_BASE,FT_LINK,FT_SAVES,FT_EXPORTS,FT_STAGE,FT_DISK_RULE,FT_RANK0. Never slinky-9.
exec bash /data/home/amartyavarma/ppm/pose6/mono17850sam/mono17850_run.sh
