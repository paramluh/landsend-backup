#!/bin/bash
# delete the PARTIAL node-local export-198 of si20k-vox-ppm11488 (crashed mid-write: no COMPLETE, 4 of 8 rank files per node) to free /scratch
source /data/home/amartyavarma/ppm/scratch_guard.sh
H=$(hostname -s); D=/scratch/amartyavarma/si20k-vox-ppm11488/exports/export-198
[ -d $D ] || { echo "$H: no $D"; exit 0; }
[ -f $D/COMPLETE ] && { echo "$H: $D has COMPLETE: NOT deleting"; exit 1; }
n=$(ls $D | wc -l); guard_rm $D && echo "$H: deleted partial $D ($n files); /scratch free $(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9) GB"
