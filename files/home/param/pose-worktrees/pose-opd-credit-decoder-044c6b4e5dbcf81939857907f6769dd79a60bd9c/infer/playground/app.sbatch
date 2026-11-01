#!/bin/bash
#SBATCH -J genweb-app
#SBATCH -x slinky-9
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 72:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-posedup-live/genweb/logs/%x-%j.out
# 10-26: the playground's always-up front (:9205) + supervised app server (9206/9207, zero-downtime deploys: touch genweb/DEPLOY).
exec bash /data/home/amartyavarma/ppm/aloe-posedup-live/genweb/app_sup.sh
