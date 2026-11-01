#!/bin/bash
#SBATCH -J pose6-monoband -p all -x slinky-9 --ntasks-per-node=8 --cpus-per-task=4 --gres=gpu:8 --mem=500G -t 2-00:00:00
#SBATCH -o /data/home/amartyavarma/ppm/pose6/monoband/logs/ft-%j.log
# pose6 monoband (10-29, main -a8) = pose6/mono17850/mono17850.sbatch (job 31240) with this kit's name / log / run script. Submitted by
# launch.sh --go only (-N8 -w <hosts> --mem=500G -t 2-00:00:00 --export=ALL,FT_BASE,FT_LINK,FT_SAVES,FT_EXPORTS,FT_STAGE,FT_DISK_RULE,FT_RANK0).
exec bash /data/home/amartyavarma/ppm/pose6/monoband/monoband_run.sh
