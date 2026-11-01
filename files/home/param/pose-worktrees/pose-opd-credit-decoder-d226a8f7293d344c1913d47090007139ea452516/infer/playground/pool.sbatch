#!/bin/bash
#SBATCH -J genweb-poolctl
#SBATCH -x slinky-9
#SBATCH -c 1
#SBATCH --mem=1G
#SBATCH -t 72:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-posedup-live/genweb/logs/%x-%j.out
exec /data/home/amartyavarma/ppm/aloe-posedup-live/genweb/pool.sh
