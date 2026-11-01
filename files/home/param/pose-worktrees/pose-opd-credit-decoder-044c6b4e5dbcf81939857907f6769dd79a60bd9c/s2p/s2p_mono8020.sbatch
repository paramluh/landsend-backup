#!/bin/bash
#SBATCH -J pose-s2p-ft -p all -N1 -w slinky-10 -x slinky-9 --ntasks-per-node=8 --cpus-per-task=4 --gres=gpu:8 --mem=530G -t 1-06:00:00
#SBATCH -o /data/home/amartyavarma/ppm/pose-s2p/logs/s2p-ft-%j.log
# pose-s2p (10-27) DRY CONFIG, NOT SUBMITTED (main / user decide): the aloe-pose-mono-8020 recipe (job 24272 -> export-357) verbatim with ONE
# change, PPM_S2P=1 (speech codes at T -> the pose tower at T, zero-init), from the same init (PPM local-11488), on ONE spare 8 x H100 node
# (world 8: the Rosie FT's AdamW host offload / no FULL saves). Stop: touch control/STOP_AFTER_SAVE (world 8: a FULL save OOMs the host;
# see TESTS.md (e)). Launch: sbatch /data/home/amartyavarma/ppm/pose-s2p/s2p_mono8020.sbatch   (override the node with -w slinky-18)
exec bash /data/home/amartyavarma/ppm/pose-s2p/s2p_mono8020_run.sh
