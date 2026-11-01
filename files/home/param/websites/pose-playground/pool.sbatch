#!/bin/bash
#SBATCH -J genweb-poolctl
#SBATCH -x slinky-9
#SBATCH -c 1
#SBATCH --mem=1G
#SBATCH -t 72:00:00
#SBATCH -o logs/%x-%j.out
: "${CANOPY_ROOT:?export CANOPY_ROOT=<cluster home that holds ppm/ venvs/ models/; see README>}"
exec ${GENWEB:-$CANOPY_ROOT/ppm/aloe-posedup-live/genweb}/pool.sh
