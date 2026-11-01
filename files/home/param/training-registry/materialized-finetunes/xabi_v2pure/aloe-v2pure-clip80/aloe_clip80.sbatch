#!/bin/bash
#SBATCH -J aloeclip80 -p all -N3 -w slinky-0,slinky-1,slinky-2 --ntasks-per-node=8 --cpus-per-task=4 --gres=gpu:8 --mem=500G -t 2-00:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-v2pure-clip80/logs/aloeclip80-%j.log
# aloe-v2pure-clip80 (10-18): = job 23051 (aloe-v2pure-ft: PPM 32B monoplex Aloe pose FT on the release-v2 v2pure data) with ONE change, the
# system-prompt rule (no persona text; voice clip on a seeded 80% of items). 3 TTS-fleet nodes x 8 H100 (world 24, as 23051).
# 8 nodes (world 64, same 64 rows / step): sbatch -N8 -w <8 nodes> aloe_clip80.sbatch (command-line options override this header).
# Stop: touch control/STOP_AFTER_SAVE (or scancel -b -s USR1 <job>). Resume: sbatch this file again (same nodes).
exec bash /data/home/amartyavarma/ppm/aloe-v2pure-clip80/aloe_clip80_run.sh
