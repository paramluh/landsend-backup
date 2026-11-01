#!/bin/bash
#SBATCH -J pose6-ft -p all -x slinky-9 --ntasks-per-node=8 --cpus-per-task=4 --gres=gpu:8 --mem=500G -t 2-00:00:00
#SBATCH -o /data/home/amartyavarma/ppm/pose6/kit/logs/ft-%j.log
# pose6 B3 (10-27) monoplex Aloe pose FT = the aloe-pose-mono-8020 recipe (job 24272 -> export-357) with s2p cb0 + G0COND, from a base of the
# 8,1,1 pose run 31101 (FT_BASE). NOT submitted by the kit: launch.sh submits it on --go only, with -N / -w / --mem / -t for the chosen layout
# (DEFAULT 8 nodes = world 64 = the recipe verbatim: --mem=500G -t 2-00:00:00, mono8020's; FALLBACK 1 node = world 8: --mem=530G
# -t 1-06:00:00, the Rosie FT's) and --export=ALL,FT_BASE=<base>,FT_LINK,FT_SAVES,FT_EXPORTS,FT_STAGE. Stop: touch control/STOP_AFTER_SAVE (world 8: no FULL save possible, see README). Never slinky-9.
exec bash /data/home/amartyavarma/ppm/pose6/kit/ft_s2pg0_run.sh
