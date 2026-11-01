#!/bin/bash
#SBATCH -J genweb-app
#SBATCH -x slinky-9
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 72:00:00
#SBATCH -o logs/%x-%j.out
# 10-26: the playground's always-up front (:9205) + supervised app server (9206/9207, zero-downtime deploys: touch genweb/DEPLOY).
: "${CANOPY_ROOT:?export CANOPY_ROOT=<cluster home that holds ppm/ venvs/ models/; see README>}"
exec bash ${GENWEB:-$CANOPY_ROOT/ppm/aloe-posedup-live/genweb}/app_sup.sh
