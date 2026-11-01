#!/bin/bash
#SBATCH -J npl5-lane -p all -x slinky-9 -N1 -n1 --gres=none -t 12:00:00
#SBATCH -o /data/home/amartyavarma/ppm/pose5/npl5/logs/lane-%j.out
# npl5 staging lane on a spare: sbatch -w <spare> -c <cpus> --mem=<G> stage_lane.sbatch <lane name> <RAW root>
# (lane A: slinky-10, RAW=/scratch/amartyavarma/npl5-stage; lane B: slinky-4, RAW=/data/home/amartyavarma/ppm/pose5/npl5/raw)
exec bash /data/home/amartyavarma/ppm/pose5/npl5/bin/${LANE_SCRIPT:-stage_lane.sh} "$@"
